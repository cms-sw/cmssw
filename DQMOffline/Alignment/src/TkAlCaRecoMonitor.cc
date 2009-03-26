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
//#include <sstream>
#include "TLorentzVector.h"

TkAlCaRecoMonitor::TkAlCaRecoMonitor(const edm::ParameterSet& iConfig) {
  dqmStore_ = edm::Service<DQMStore>().operator->();
  conf_ = iConfig;
}

TkAlCaRecoMonitor::~TkAlCaRecoMonitor() { } 

void TkAlCaRecoMonitor::beginJob(edm::EventSetup const& iSetup) {
  using namespace edm;

  std::string histname;  //for naming the histograms according to algorithm used

  trackProducer_ = conf_.getParameter<edm::InputTag>("TrackProducer");
  referenceTrackProducer_ = conf_.getParameter<edm::InputTag>("ReferenceTrackProducer");

  std::string AlgoName     = conf_.getParameter<std::string>("AlgoName");
  std::string MEFolderName = conf_.getParameter<std::string>("FolderName"); 

  daughterMass_ = conf_.getParameter<double>("daughterMass");

  maxJetPt_ = conf_.getParameter<double>("maxJetPt");

  dqmStore_->setCurrentFolder(MEFolderName);
  fillInvariantMass_ = conf_.getParameter<bool>("fillInvariantMass");
  runsOnReco_ = conf_.getParameter<bool>("runsOnReco");
  useSignedR_ = conf_.getParameter<bool>("useSignedR");
  fillRawIdMap_ = conf_.getParameter<bool>("fillRawIdMap");

  //    
  unsigned int MassBin = conf_.getParameter<unsigned int>("MassBin");
  double MassMin = conf_.getParameter<double>("MassMin");
  double MassMax = conf_.getParameter<double>("MassMax");

  if(fillInvariantMass_){
    histname = "InvariantMass_";
    invariantMass_ = dqmStore_->book1D(histname+AlgoName, histname+AlgoName, MassBin, MassMin, MassMax);
    invariantMass_->setAxisTitle("invariant Mass / GeV");
  }

  unsigned int TrackPtPositiveBin = conf_.getParameter<unsigned int>("TrackPtBin");
  double TrackPtPositiveMin = conf_.getParameter<double>("TrackPtMin");
  double TrackPtPositiveMax = conf_.getParameter<double>("TrackPtMax");

  histname = "TrackPtPositive_";
  TrackPtPositive_ = dqmStore_->book1D(histname+AlgoName, histname+AlgoName, TrackPtPositiveBin, TrackPtPositiveMin, TrackPtPositiveMax);
  TrackPtPositive_->setAxisTitle("p_{T} of tracks charge > 0");

  unsigned int TrackPtNegativeBin = conf_.getParameter<unsigned int>("TrackPtBin");
  double TrackPtNegativeMin = conf_.getParameter<double>("TrackPtMin");
  double TrackPtNegativeMax = conf_.getParameter<double>("TrackPtMax");

  histname = "TrackPtNegative_";
  TrackPtNegative_ = dqmStore_->book1D(histname+AlgoName, histname+AlgoName, TrackPtNegativeBin, TrackPtNegativeMin, TrackPtNegativeMax);
  TrackPtNegative_->setAxisTitle("p_{T} of tracks charge < 0");

  unsigned int TrackCurvatureBin = conf_.getParameter<unsigned int>("TrackCurvatureBin");
  double TrackCurvatureMin = conf_.getParameter<double>("TrackCurvatureMin");
  double TrackCurvatureMax = conf_.getParameter<double>("TrackCurvatureMax");

  histname = "TrackCurvature_";
  TrackCurvature_ = dqmStore_->book1D(histname+AlgoName, histname+AlgoName, TrackCurvatureBin, TrackCurvatureMin, TrackCurvatureMax);
  TrackCurvature_->setAxisTitle("#kappa track");


  histname = "TrackQuality_";
  TrackQuality_ = dqmStore_->book1D(histname+AlgoName, histname+AlgoName, 
				 reco::TrackBase::qualitySize-1, -0.5, reco::TrackBase::qualitySize-1.5);
  TrackQuality_->setAxisTitle("quality");
  for ( int i = 0; i<reco::TrackBase::qualitySize-1; ++i){
    TrackQuality_->getTH1()->GetXaxis()->SetBinLabel(i+1,
	            reco::TrackBase::qualityName( reco::TrackBase::TrackQuality(i) ).c_str());
  } 

  unsigned int SumChargeBin = conf_.getParameter<unsigned int>("SumChargeBin");
  double SumChargeMin = conf_.getParameter<double>("SumChargeMin");
  double SumChargeMax = conf_.getParameter<double>("SumChargeMax");

  histname = "SumCharge_";
  sumCharge_ = dqmStore_->book1D(histname+AlgoName, histname+AlgoName, SumChargeBin, SumChargeMin, SumChargeMax);
  sumCharge_->setAxisTitle("#SigmaCharge");

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

  unsigned int TrackEfficiencyBin = conf_.getParameter<unsigned int>("TrackEfficiencyBin");
  double TrackEfficiencyMin = conf_.getParameter<double>("TrackEfficiencyMin");
  double TrackEfficiencyMax = conf_.getParameter<double>("TrackEfficiencyMax");

  histname = "AlCaRecoTrackEfficiency_";
  AlCaRecoTrackEfficiency_ = dqmStore_->book1D(histname+AlgoName, histname+AlgoName, TrackEfficiencyBin, TrackEfficiencyMin, TrackEfficiencyMax);
  AlCaRecoTrackEfficiency_->setAxisTitle("n("+trackProducer_.label()+") / n("+referenceTrackProducer_.label()+")");

  int zBin =  conf_.getParameter<unsigned int>("HitMapsZBin"); //300
  double zMax = conf_.getParameter<double>("HitMapZMax"); //300.0; //cm
  
  int rBin = conf_.getParameter<unsigned int>("HitMapsRBin");//120;
  double rMax = conf_.getParameter<double>("HitMapRMax"); //120.0; //cm

  histname = "Hits_ZvsR_";
  double rMin = 0.0;
  if( useSignedR_ )
    rMin = -rMax;

  Hits_ZvsR_ = dqmStore_->book2D(histname+AlgoName, histname+AlgoName, zBin, -zMax, zMax, rBin, rMin, rMax);

  histname = "Hits_XvsY_";
  Hits_XvsY_ = dqmStore_->book2D(histname+AlgoName, histname+AlgoName, rBin, -rMax, rMax, rBin, -rMax, rMax);

  edm::ESHandle<TrackerGeometry> tkGeom;
  iSetup.get<TrackerDigiGeometryRecord>().get( tkGeom );
  TrackerGeometry tracker  = TrackerGeometry(*tkGeom);
  std::vector<int> sortedRawIds;
  for( std::vector<DetId>::const_iterator iDetId = tracker.detUnitIds().begin(); iDetId != tracker.detUnitIds().end(); ++iDetId){
    sortedRawIds.push_back( (*iDetId).rawId() );
  }
  std::sort(sortedRawIds.begin(),sortedRawIds.end());

  int i = 0;
  for( std::vector<int>::iterator iRawId = sortedRawIds.begin(); iRawId != sortedRawIds.end(); ++iRawId){
    binByRawId_[ (*iRawId) ] = i;
    i++;
  }
  if( fillRawIdMap_){
    histname = "Hits_perDetId_";
    
    //leads to differences in axsis between samples??
    //int nModules = binByRawId_.size();
    //Hits_perDetId_ = dqmStore_->book1D(histname+AlgoName, histname+AlgoName, nModules, static_cast<double>(nModules) -0.5, static_cast<double>(nModules) -0.5);
    Hits_perDetId_ = dqmStore_->book1D(histname+AlgoName, histname+AlgoName, 16601,-0.5, 16600.5 );
    Hits_perDetId_->setAxisTitle("rawId Bins");

    //// impossible takes too much memory :(  
    //  std::stringstream binLabel;
    //  for( std::map<int,int>::iterator it = binByRawId_.begin(); it != binByRawId_.end(); ++it ){
    //    binLabel.str() = "";
    //    binLabel << (*it).first;
    //    Hits_perDetId_->getTH1()->GetXaxis()->SetBinLabel( (*it).second +1, binLabel.str().c_str());
    //  }
  }

  iSetup.get<IdealMagneticFieldRecord>().get(magneticField_);
  if (!magneticField_.isValid()){
    LogError("Alignment")<<"invalid magnetic field configuration encountered!";
    return;
  }
}
//
// -- Analyse
//
void TkAlCaRecoMonitor::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;
  
  Handle<reco::TrackCollection> trackCollection;
  iEvent.getByLabel(trackProducer_, trackCollection);
  if (!trackCollection.isValid()){
    LogError("Alignment")<<"invalid trackcollection encountered!";
    return;
  }

  Handle<reco::TrackCollection> referenceTrackCollection;
  iEvent.getByLabel(referenceTrackProducer_, referenceTrackCollection);
  if (!trackCollection.isValid()){
    LogError("Alignment")<<"invalid reference track-collection encountered!";
    return;
  }

  edm::ESHandle<TrackerGeometry> geometry;
  iSetup.get<TrackerDigiGeometryRecord>().get(geometry);
  if(! geometry.isValid()){
    LogError("Alignment")<<"invalid geometry found in event setup!";
  } 

  InputTag jetCollection = conf_.getParameter<edm::InputTag>("CaloJetCollection");
  Handle<reco::CaloJetCollection> jets;
  if(runsOnReco_){
    iEvent.getByLabel(jetCollection, jets);
    if(! jets.isValid()){
      LogError("Alignment")<<"no jet collection found in event!";
    }
  }

  AlCaRecoTrackEfficiency_->Fill( static_cast<double>((*trackCollection).size()) / (*referenceTrackCollection).size() );
  
  double sumOfCharges = 0;
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

    if( (*track).charge() > 0 )
      TrackPtPositive_->Fill( (*track).pt() );
    if( (*track).charge() < 0 )
      TrackPtNegative_->Fill( (*track).pt() );

    GlobalPoint gPoint((*track).vx(), (*track).vy(), (*track).vz());
    double B = magneticField_->inTesla(gPoint).z();
    double curv = -(*track).charge()*0.002998*B/(*track).pt();
    //std::cout << "curv: "<<curv<<std::endl;
    TrackCurvature_->Fill( curv );
    TrackQuality_->Fill( (*track).qualityMask()  );
    minTrackDeltaR_->Fill( minTrackDeltaR );
    fillHitmaps( *track, *geometry );
    sumOfCharges += (*track).charge();
  }

  sumCharge_->Fill( sumOfCharges );

  if(fillInvariantMass_){
    if((*trackCollection).size() == 2){
      TLorentzVector track0((*trackCollection).at(0).px(),(*trackCollection).at(0).py(),(*trackCollection).at(0).pz(),
			    sqrt(((*trackCollection).at(0).p()*(*trackCollection).at(0).p())+daughterMass_*daughterMass_));
      TLorentzVector track1((*trackCollection).at(1).px(),(*trackCollection).at(1).py(),(*trackCollection).at(1).pz(),
			    sqrt(((*trackCollection).at(1).p()*(*trackCollection).at(1).p())+daughterMass_*daughterMass_));
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
      double r = sqrt( globP.x()*globP.x() + globP.y()*globP.y() );
      if( useSignedR_ )
	r*= globP.y() / fabs( globP.y() );
      Hits_ZvsR_->Fill( globP.z(), r );
      Hits_XvsY_->Fill( globP.x(), globP.y() );
      if( fillRawIdMap_)
	 Hits_perDetId_->Fill( binByRawId_[ geoId.rawId() ]);  
    }
    //me->Fill( tHit->globalPosition().z(), tHit->globalPosition().mag() );
  }
}



void TkAlCaRecoMonitor::endJob(void) {
  bool outputMEsInRootFile = conf_.getParameter<bool>("OutputMEsInRootFile");
  std::string outputFileName = conf_.getParameter<std::string>("OutputFileName");
  if(outputMEsInRootFile){
    //dqmStore_->showDirStructure();
    dqmStore_->save(outputFileName);
  }
}

DEFINE_FWK_MODULE(TkAlCaRecoMonitor);

