/*
 *  See header file for a description of this class.
 *
 */

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "MagneticField/Engine/interface/MagneticField.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/InputTag.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMOffline/Alignment/interface/TkAlCaRecoMonitor.h"

//#include <TrackingTools/Records/interface/TransientRecHitRecord.h>

#include <DataFormats/JetReco/interface/CaloJet.h>
#include <DataFormats/Math/interface/deltaR.h>

#include <string>
#include "TLorentzVector.h"

TkAlCaRecoMonitor::TkAlCaRecoMonitor(const edm::ParameterSet& iConfig) {
  dqmStore_ = edm::Service<DQMStore>().operator->();
  conf_ = iConfig;
}

TkAlCaRecoMonitor::~TkAlCaRecoMonitor() { } 

void TkAlCaRecoMonitor::beginJob(edm::EventSetup const& iSetup) {
  using namespace edm;

  std::string histname;  //for naming the histograms according to algorithm used

  std::string AlgoName     = conf_.getParameter<std::string>("AlgoName");
  std::string MEFolderName = conf_.getParameter<std::string>("FolderName"); 

  maxJetPt_ = conf_.getParameter<double>("maxJetPt");

  dqmStore_->setCurrentFolder(MEFolderName);
  fillInvariantMass_ = conf_.getParameter<bool>("fillInvariantMass");
  runsOnReco_ = conf_.getParameter<bool>("runsOnReco");
  //    
  unsigned int MassBin = conf_.getParameter<unsigned int>("MassBin");
  double MassMin = conf_.getParameter<double>("MassMin");
  double MassMax = conf_.getParameter<double>("MassMax");

  if(fillInvariantMass_){
    histname = "InvariantMass_";
    invariantMass_ = dqmStore_->book1D(histname+AlgoName, histname+AlgoName, MassBin, MassMin, MassMax);
    invariantMass_->setAxisTitle("invariant Mass / GeV");
  }

  unsigned int    JetPtBin = conf_.getParameter<unsigned int>("JetPtBin");
  double JetPtMin = conf_.getParameter<double>("JetPtMin");
  double JetPtMax = conf_.getParameter<double>("JetPtMax");

  histname = "JetPt_";
  jetPt_ = dqmStore_->book1D(histname+AlgoName, histname+AlgoName, JetPtBin, JetPtMin, JetPtMax);
  jetPt_->setAxisTitle("jet p_{T} / GeV");

  unsigned int    MinJetDeltaRBin = conf_.getParameter<unsigned int>("MinJetDeltaRBin");
  double MinJetDeltaRMin = conf_.getParameter<double>("MinJetDeltaRMin");
  double MinJetDeltaRMax = conf_.getParameter<double>("MinJetDeltaRMax");

  histname = "MinJetDeltaR_";
  minJetDeltaR_ = dqmStore_->book1D(histname+AlgoName, histname+AlgoName, MinJetDeltaRBin, MinJetDeltaRMin, MinJetDeltaRMax);
  minJetDeltaR_->setAxisTitle("minimal Jet #DeltaR / rad");

  unsigned int    MinTrackDeltaRBin = conf_.getParameter<unsigned int>("MinTrackDeltaRBin");
  double MinTrackDeltaRMin = conf_.getParameter<double>("MinTrackDeltaRMin");
  double MinTrackDeltaRMax = conf_.getParameter<double>("MinTrackDeltaRMax");

  histname = "MinTrackDeltaR_";
  minTrackDeltaR_ = dqmStore_->book1D(histname+AlgoName, histname+AlgoName, MinTrackDeltaRBin, MinTrackDeltaRMin, MinTrackDeltaRMax);
  minTrackDeltaR_->setAxisTitle("minimal Track #DeltaR / rad");
  
  int zBin = 300;
  double zMax = 300.0; //cm
  
  int rBin = 120;
  double rMax = 120.0; //cm

  histname = "Hits_ZvsR_";
  Hits_ZvsR_ = dqmStore_->book2D(histname+AlgoName, histname+AlgoName, zBin, -zMax, zMax, rBin, 0.0, rMax);

  histname = "Hits_XvsY_";
  Hits_XvsY_ = dqmStore_->book2D(histname+AlgoName, histname+AlgoName, rBin, -rMax, rMax, rBin, -rMax, rMax);

}
//
// -- Analyse
//
void TkAlCaRecoMonitor::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;
  
  InputTag trackProducer = conf_.getParameter<edm::InputTag>("TrackProducer");
  
  Handle<reco::TrackCollection> trackCollection;
  iEvent.getByLabel(trackProducer, trackCollection);
  if (!trackCollection.isValid()){
    LogError("Alignment")<<"invalid trackcollection encountered!";
    return;
  }
  
//  std::string recHitBuilderName = "WithTrackAngle";
//  edm::ESHandle<TransientTrackingRecHitBuilder> recHitBuilderHandle;
//  
//  iSetup.get<TransientRecHitRecord>().get(recHitBuilderName,recHitBuilderHandle);

  edm::ESHandle<TrackerGeometry> geometry;
  iSetup.get<TrackerDigiGeometryRecord>().get(geometry);
  if(! geometry.isValid()){
    LogError("Alignment")<<"invalid geometry found in event setup!";
  } 

  InputTag jetCollection = conf_.getParameter<edm::InputTag>("CaloJetCollection");
  Handle<reco::CaloJetCollection> jets;
  if(runsOnReco_){
    iEvent.getByLabel(jetCollection  ,jets);
    if(! jets.isValid()){
      LogError("Alignment")<<"no jets found in event!";
    }
  }
    
  for( reco::TrackCollection::const_iterator track = (*trackCollection).begin(); track < (*trackCollection).end(); ++track ){
    double dR = 0;  
    if(runsOnReco_){
      double minJetDeltaR = 10; // some number > 2pi
      for(reco::CaloJetCollection::const_iterator itJet = jets->begin(); itJet != jets->end() ; ++itJet) {
	jetPt_->Fill( (*itJet).pt() );
	dR = deltaR((*track),(*itJet));
	if((*itJet).pt() > maxJetPt_ && dR < minJetDeltaR)
	  minJetDeltaR = dR; 
	
	//LogInfo("Alignment") <<">  isolated: "<< isolated << " jetPt "<< (*itJet).pt() <<" deltaR: "<< deltaR(*(*it),(*itJet)) ;
      }
      minJetDeltaR_->Fill( minJetDeltaR );
    }
    
    double minTrackDeltaR = 10; // some number > 2pi
    for( reco::TrackCollection::const_iterator track2 = (*trackCollection).begin(); track2 < (*trackCollection).end(); ++track2 ){
      dR = deltaR((*track),(*track2));
      if(dR < minTrackDeltaR && dR > 1e-6)
	minTrackDeltaR = dR;
    }
    minTrackDeltaR_->Fill( minTrackDeltaR );
    fillHitmaps( *track, *geometry );
    //, recHitBuilderHandle
  }
  if(fillInvariantMass_){
    if((*trackCollection).size() == 2){
      double theDaughterMass = 0.10565836;
      TLorentzVector track0((*trackCollection).at(0).px(),(*trackCollection).at(0).py(),(*trackCollection).at(0).pz(),
			    sqrt(((*trackCollection).at(0).p()*(*trackCollection).at(0).p())+theDaughterMass*theDaughterMass));
      TLorentzVector track1((*trackCollection).at(1).px(),(*trackCollection).at(1).py(),(*trackCollection).at(1).pz(),
			    sqrt(((*trackCollection).at(1).p()*(*trackCollection).at(1).p())+theDaughterMass*theDaughterMass));
      TLorentzVector mother = track0+track1;
      
      invariantMass_->Fill( mother.M() );
    }else{
      LogInfo("Alignment")<<"wrong number of tracks trackcollection encountered: "<<(*trackCollection).size();
    }
  }
}

void TkAlCaRecoMonitor::fillHitmaps(const reco::Track &track, const TrackerGeometry& geometry)
{
  for (trackingRecHit_iterator iHit = track.recHitsBegin(); iHit != track.recHitsEnd(); ++iHit) {
    
    //    TransientTrackingRecHit::ConstRecHitPointer tHit = 
    //  recHitBuilderHandle->build( (*iHit).get() );
    if( (*iHit)->isValid() ){
      const TrackingRecHit *hit = (*iHit).get();
      const DetId geoId(hit->geographicalId());
      const GeomDet* gd = geometry.idToDet(geoId);
      // since 2_1_X local hit positions are transient. taking center of the hit module for now.
      // The alternative would be the coarse estimation or a refit.
      //const GlobalPoint globP( gd->toGlobal( hit->localPosition() ) );
      const GlobalPoint globP( gd->toGlobal( Local3DPoint(0.,0.,0.) ) );
   
      Hits_ZvsR_->Fill( globP.z(), sqrt( globP.x()*globP.x() + globP.y()*globP.y() ) );
      Hits_XvsY_->Fill( globP.x(), globP.y() );

    }
    //me->Fill( tHit->globalPosition().z(), tHit->globalPosition().mag() );
  }
}



void TkAlCaRecoMonitor::endJob(void) {
  bool outputMEsInRootFile = conf_.getParameter<bool>("OutputMEsInRootFile");
  std::string outputFileName = conf_.getParameter<std::string>("OutputFileName");
  if(outputMEsInRootFile){
    dqmStore_->showDirStructure();
    dqmStore_->save(outputFileName);
  }
}

DEFINE_FWK_MODULE(TkAlCaRecoMonitor);
