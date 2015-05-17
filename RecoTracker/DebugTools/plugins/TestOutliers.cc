// -*- C++ -*-
//
// Package:    TestOutliers
// Class:      TestOutliers
// 
/**\class TestOutliers TestOutliers.cc RecoTracker/DebugTools/plugins/TestOutliers.cc

   Description: <one line class summary>

   Implementation:
   <Notes on implementation>
*/
//
// Original Author:  Giuseppe Cerati
//         Created:  Mon Sep 17 10:31:30 CEST 2007
//
//


// system include files
#include <memory>
#include <vector>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "SimDataFormats/Associations/interface/TrackToTrackingParticleAssociator.h"
#include "SimTracker/TrackerHitAssociation/interface/TrackerHitAssociator.h"
#include <TH1.h>
#include <TH2.h>
#include <TFile.h>
#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"
#include "TrackingTools/PatternTools/interface/TSCPBuilderNoMaterial.h"
#include "MagneticField/Engine/interface/MagneticField.h" 
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h" 
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/ErrorFrameTransformer.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"

//
// class decleration
//

class TestOutliers : public edm::EDAnalyzer {
public:
  explicit TestOutliers(const edm::ParameterSet&);
  ~TestOutliers();


private:
  virtual void beginRun(edm::Run & run, const edm::EventSetup&) ;
  virtual void analyze(const edm::Event&, const edm::EventSetup&) override;
  virtual void endJob() override ;

  // ----------member data ---------------------------
  edm::InputTag trackTagsOut_; //used to select what tracks to read from configuration file
  edm::InputTag trackTagsOld_; //used to select what tracks to read from configuration file
  edm::InputTag tpTags_; //used to select what tracks to read from configuration file
  TrackerHitAssociator::Config trackerHitAssociatorConfig_;
  edm::EDGetTokenT<reco::TrackToTrackingParticleAssociator> theAssociatorOldToken;
  edm::EDGetTokenT<reco::TrackToTrackingParticleAssociator> theAssociatorOutToken;
  edm::ESHandle<TrackerGeometry> theG;
  std::string out;
  TFile * file;
  TH1F *histoPtOut, *histoPtOld ;
  TH1F *histoQoverpOut,*histoQoverpOld;
  TH1F *histoPhiOut,*histoPhiOld;
  TH1F *histoD0Out,*histoD0Old;
  TH1F *histoDzOut,*histoDzOld;
  TH1F *histoLambdaOut,*histoLambdaOld;
  TH1F *deltahits,*deltahitsAssocGained,*deltahitsAssocLost,*hitsPerTrackLost,*hitsPerTrackAssocLost,*hitsPerTrackGained,*hitsPerTrackAssocGained,*deltahitsOK,*deltahitsNO;
  TH1F *okcutsOut,*okcutsOld;
  TH1F *tracks, *goodbadhits, *process, *goodcluster, *goodprocess, *badcluster, *badprocess;
  TH1F *goodhittype, *goodlayer, *goodhittype_clgteq4, *goodlayer_clgteq4, *goodhittype_cllt4, *goodlayer_cllt4, *badhittype, *badlayer;
  TH2F *posxy, *poszr;
  TH1F *goodpixgteq4_simvecsize, *goodpixlt4_simvecsize;
  TH1F *goodst1gteq4_simvecsize, *goodst1lt4_simvecsize;
  TH1F *goodst2gteq4_simvecsize, *goodst2lt4_simvecsize;
  TH1F *goodprjgteq4_simvecsize, *goodprjlt4_simvecsize;
  TH1F *goodpix_clustersize;
  TH1F *goodst1_clustersize;
  TH1F *goodst2_clustersize;
  TH1F *goodprj_clustersize;
  TH1F *goodpix_simvecsize;
  TH1F *goodst1_simvecsize;
  TH1F *goodst2_simvecsize;
  TH1F *goodprj_simvecsize;
  TH1F *goodhittype_simvecsmall, *goodlayer_simvecsmall, *goodhittype_simvecbig, *goodlayer_simvecbig, *goodbadmerged, *goodbadmergedLost, *goodbadmergedGained;
  TH1F *energyLoss,*energyLossMax,*energyLossRatio, *nOfTrackIds, *mergedPull, *mergedlayer, *mergedcluster, *mergedhittype;
  TH1F *sizeOut, *sizeOld, *sizeOutT, *sizeOldT;
  TH1F *countOutA, *countOutT, *countOldT;
  TH1F *gainedhits,*gainedhits2;
  TH1F *probXgood,*probXbad,*probXdelta,*probXshared,*probXnoshare;
  TH1F *probYgood,*probYbad,*probYdelta,*probYshared,*probYnoshare;
};
//
// constants, enums and typedefs
//

//
// static data member definitions
//

using namespace edm;

//
// constructors and destructor
//
TestOutliers::TestOutliers(const edm::ParameterSet& iConfig)
  :
  trackTagsOut_(iConfig.getUntrackedParameter<edm::InputTag>("tracksOut")),
  trackTagsOld_(iConfig.getUntrackedParameter<edm::InputTag>("tracksOld")),
  tpTags_(iConfig.getUntrackedParameter<edm::InputTag>("tp")),
  trackerHitAssociatorConfig_(consumesCollector()),
  theAssociatorOldToken(consumes<reco::TrackToTrackingParticleAssociator>(iConfig.getUntrackedParameter<edm::InputTag>("TrackAssociatorByHitsOld"))),
  theAssociatorOutToken(consumes<reco::TrackToTrackingParticleAssociator>(iConfig.getUntrackedParameter<edm::InputTag>("TrackAssociatorByHitsOut"))),
  out(iConfig.getParameter<std::string>("out"))
{
  LogTrace("TestOutliers") <<"constructor";
//   ParameterSet cuts = iConfig.getParameter<ParameterSet>("RecoTracksCuts");
//   selectRecoTracks = RecoTrackSelector(cuts.getParameter<double>("ptMin"),
// 				       cuts.getParameter<double>("minRapidity"),
// 				       cuts.getParameter<double>("maxRapidity"),
// 				       cuts.getParameter<double>("tip"),
// 				       cuts.getParameter<double>("lip"),
// 				       cuts.getParameter<int>("minHit"),
// 				       cuts.getParameter<double>("maxChi2"));
  
  LogTrace("TestOutliers") <<"end constructor";
}


TestOutliers::~TestOutliers()
{
 
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to for each event  ------------
void
TestOutliers::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  //Retrieve tracker topology from geometry
  edm::ESHandle<TrackerTopology> tTopo;
  iSetup.get<TrackerTopologyRcd>().get(tTopo);



  using namespace edm;
  using namespace std;
  using reco::TrackCollection;

  LogTrace("TestOutliers") <<"analyze event #" << iEvent.id();
  
  edm::Handle<edm::View<reco::Track> > tracksOut;
  iEvent.getByLabel(trackTagsOut_,tracksOut);
  edm::Handle<edm::View<reco::Track> > tracksOld;
  iEvent.getByLabel(trackTagsOld_,tracksOld);
  Handle<TrackingParticleCollection> tps;
  iEvent.getByLabel(tpTags_,tps);
  edm::Handle<reco::BeamSpot> beamSpot;
  iEvent.getByLabel("offlineBeamSpot",beamSpot); 

  TrackerHitAssociator hitAssociator(iEvent, trackerHitAssociatorConfig_);

  edm::Handle<reco::TrackToTrackingParticleAssociator> hAssociatorOld;
  iEvent.getByToken(theAssociatorOldToken, hAssociatorOld);
  const reco::TrackToTrackingParticleAssociator *theAssociatorOld = hAssociatorOld.product();

  edm::Handle<reco::TrackToTrackingParticleAssociator> hAssociatorOut;
  iEvent.getByToken(theAssociatorOutToken, hAssociatorOut);
  const reco::TrackToTrackingParticleAssociator *theAssociatorOut = hAssociatorOut.product();

  reco::RecoToSimCollection recSimCollOut=theAssociatorOut->associateRecoToSim(tracksOut, tps);
  reco::RecoToSimCollection recSimCollOld=theAssociatorOld->associateRecoToSim(tracksOld, tps);
  sizeOut->Fill(recSimCollOut.size());
  sizeOld->Fill(recSimCollOld.size());
  sizeOutT->Fill(tracksOut->size());
  sizeOldT->Fill(tracksOld->size());

  LogTrace("TestOutliers") << "tOld size=" << tracksOld->size() << " tOut size=" << tracksOut->size() 
			   << " aOld size=" << recSimCollOld.size() << " aOut size=" << recSimCollOut.size();

#if 0
  LogTrace("TestOutliers") << "recSimCollOld.size()=" << recSimCollOld.size() ;
  for(reco::TrackCollection::size_type j=0; j<tracksOld->size(); ++j){
    reco::TrackRef trackOld(tracksOld, j);
    //if ( !selectRecoTracks( *trackOld,beamSpot.product() ) ) continue;
    LogTrace("TestOutliers") << "trackOld->pt()=" << trackOld->pt() << " trackOld->numberOfValidHits()=" << trackOld->numberOfValidHits();
    std::vector<pair<TrackingParticleRef, double> > tpOld;
    if(recSimCollOld.find(trackOld) != recSimCollOld.end()){
      tpOld = recSimCollOld[trackOld];
      if (tpOld.size()!=0) LogTrace("TestOutliers") << " associated" ;
      else LogTrace("TestOutliers") << " NOT associated" ;
    } else LogTrace("TestOutliers") << " NOT associated" ;
  }
  LogTrace("TestOutliers") << "recSimCollOut.size()=" << recSimCollOut.size() ;
  for(reco::TrackCollection::size_type j=0; j<tracksOut->size(); ++j){
    reco::TrackRef trackOut(tracksOut, j);
    //if ( !selectRecoTracks( *trackOut,beamSpot.product() ) ) continue;
    LogTrace("TestOutliers") << "trackOut->pt()=" << trackOut->pt() << " trackOut->numberOfValidHits()=" << trackOut->numberOfValidHits();
    std::vector<pair<TrackingParticleRef, double> > tpOut;
    if(recSimCollOut.find(trackOut) != recSimCollOut.end()){
      tpOut = recSimCollOut[trackOut];
      if (tpOut.size()!=0) LogTrace("TestOutliers") << " associated" ;
      else LogTrace("TestOutliers") << " NOT associated" ;
    } else LogTrace("TestOutliers") << " NOT associated" ;
  }
#endif

  LogTrace("TestOutliers") <<"tracksOut->size()="<<tracksOut->size();
  LogTrace("TestOutliers") <<"tracksOld->size()="<<tracksOld->size();

  std::vector<unsigned int> outused;
  for(unsigned int j=0; j<tracksOld->size(); ++j) {
    countOldT->Fill(1);
    edm::RefToBase<reco::Track> trackOld(tracksOld, j);
    LogTrace("TestOutliers") << "now track old with id=" << j << " seed ref=" << trackOld->seedRef().get() << " pt=" << trackOld->pt()
			      << " eta=" <<  trackOld->eta() << " chi2=" <<  trackOld->normalizedChi2()
			      << " tip= " << fabs(trackOld->dxy(beamSpot->position()))
			      << " lip= " << fabs(trackOld->dsz(beamSpot->position()));

    //     if (i==tracksOut->size()) {
    //       if(recSimCollOld.find(trackOld) != recSimCollOld.end()){      
    // 	vector<pair<TrackingParticleRef, double> > tpOld;
    // 	tpOld = recSimCollOld[trackOld];
    // 	if (tpOld.size()!=0) { 
    // 	  LogTrace("TestOutliers") <<"no match: old associated and out lost! old has #hits=" << trackOld->numberOfValidHits() 
    // 				   << " and fraction=" << tpOld.begin()->second;
    // 	  if (tpOld.begin()->second>0.5) hitsPerTrackLost->Fill(trackOld->numberOfValidHits());
    // 	}
    //       }
    //       continue;
    //     }

    std::vector<unsigned int> outtest;//all the out tracks with the same seed ref
    for(unsigned int k=0; k<tracksOut->size(); ++k) {
      edm::RefToBase<reco::Track> tmpOut = edm::RefToBase<reco::Track>(tracksOut, k);
      if ( tmpOut->seedRef() == trackOld->seedRef() ) {
	outtest.push_back(k);
      }      
    }

    edm::RefToBase<reco::Track> trackOut;
    if (outtest.size()==1) {// if only one that's it
      trackOut = edm::RefToBase<reco::Track>(tracksOut, outtest[0]);
      LogTrace("TestOutliers") << "now track out with id=" << outtest[0] << " seed ref=" << trackOut->seedRef().get() << " pt=" << trackOut->pt();	
      outused.push_back(outtest[0]);
    } else if (outtest.size()>1) {//if > 1 take the one that shares all the hits with the old track
      for(unsigned int p=0; p<outtest.size(); ++p) {
	edm::RefToBase<reco::Track> tmpOut = edm::RefToBase<reco::Track>(tracksOut, outtest[p]);	
	bool allhits = true;
	for (trackingRecHit_iterator itOut = tmpOut->recHitsBegin(); itOut!=tmpOut->recHitsEnd(); itOut++) {
	  if ((*itOut)->isValid()) { 
	    bool thishit = false;
	    for (trackingRecHit_iterator itOld = trackOld->recHitsBegin();  itOld!=trackOld->recHitsEnd(); itOld++) {
	      if ((*itOld)->isValid()) {
		const TrackingRecHit* kt = &(**itOld);
		if ( (*itOut)->sharesInput(kt,TrackingRecHit::some) ) { 
		  thishit = true;
		  break;
		}
	      }
	    }
	    if (!thishit) allhits = false;
	  }
	}
	if (allhits) {
	  trackOut = edm::RefToBase<reco::Track>(tracksOut, outtest[p]);
	  LogTrace("TestOutliers") << "now track out with id=" << outtest[p] << " seed ref=" << trackOut->seedRef().get() << " pt=" << trackOut->pt();	
	  outused.push_back(outtest[p]);	  
	}
      }
    } 

    if (outtest.size()==0 || trackOut.get()==0 ) {//no out track found for the old track
      if(recSimCollOld.find(trackOld) != recSimCollOld.end()){      
	vector<pair<TrackingParticleRef, double> > tpOld;
	tpOld = recSimCollOld[trackOld];
	if (tpOld.size()!=0) { 
	  LogTrace("TestOutliers") <<"no match: old associated and out lost! old has #hits=" << trackOld->numberOfValidHits() 
				   << " and fraction=" << tpOld.begin()->second;
	  if (tpOld.begin()->second>0.5) hitsPerTrackLost->Fill(trackOld->numberOfValidHits());
	}
      }      
      LogTrace("TestOutliers") <<"...skip to next old track";
      continue;
    }

    //look if old and out are associated    
    LogTrace("TestOutliers") <<"trackOut->seedRef()=" << trackOut->seedRef().get() << " trackOld->seedRef()=" << trackOld->seedRef().get();
    bool oldAssoc = recSimCollOld.find(trackOld) != recSimCollOld.end();
    bool outAssoc = recSimCollOut.find(trackOut) != recSimCollOut.end();
    LogTrace("TestOutliers") <<"outAssoc=" << outAssoc <<" oldAssoc=" << oldAssoc;

    //     if ( trackOut->seedRef()!= trackOld->seedRef() ||  
    // 	 (trackOut->seedRef() == trackOld->seedRef() && trackOut->numberOfValidHits()>trackOld->numberOfValidHits()) ) {
    //       LogTrace("TestOutliers") <<"out and old tracks does not match...";
    //       LogTrace("TestOutliers") <<"old has #hits=" << trackOld->numberOfValidHits();
    //       std::vector<pair<TrackingParticleRef, double> > tpOld;
    //       if(recSimCollOld.find(trackOld) != recSimCollOld.end()) {
    // 	tpOld = recSimCollOld[trackOld];
    // 	if (tpOld.size()!=0) { 
    // 	  LogTrace("TestOutliers") <<"old was associated with fraction=" << tpOld.begin()->second;
    // 	  if (tpOld.begin()->second>0.5) hitsPerTrackLost->Fill(trackOld->numberOfValidHits());
    // 	}
    //       }
    //       LogTrace("TestOutliers") <<"...skip to next old track";
    //       continue;
    //     }
    //     //++i;
    countOutT->Fill(1);

    //if ( !selectRecoTracks( *trackOld,beamSpot.product() ) ) continue;//no more cuts

    tracks->Fill(0);//FIXME

    TrackingParticleRef tpr;
    TrackingParticleRef tprOut;
    TrackingParticleRef tprOld;
    double fracOut;
    std::vector<unsigned int> tpids;
    std::vector<std::pair<TrackingParticleRef, double> > tpOut;
    std::vector<pair<TrackingParticleRef, double> > tpOld;
    //contare outliers delle tracce che erano associate e non lo sono piu!!!!

    if(outAssoc) {//save the ids od the tp associate to the out track
      tpOut = recSimCollOut[trackOut];
      if (tpOut.size()!=0) {
	countOutA->Fill(1);
	tprOut = tpOut.begin()->first;
	fracOut = tpOut.begin()->second;
        for (TrackingParticle::g4t_iterator g4T=tprOut->g4Track_begin(); g4T!=tprOut->g4Track_end(); ++g4T) {
          tpids.push_back(g4T->trackId());
        }
      }
    }

    if(oldAssoc){//save the ids od the tp associate to the old track
      tpOld = recSimCollOld[trackOld];
      if (tpOld.size()!=0) { 
	tprOld = tpOld.begin()->first;
	// 	LogTrace("TestOutliers") <<"old associated and out not! old has #hits=" << trackOld->numberOfValidHits() 
	// 				 << " and fraction=" << tpOld.begin()->second;
	// 	if (tpOld.begin()->second>0.5) hitsPerTrackLost->Fill(trackOld->numberOfValidHits());//deve essere un plot diverso tipo LostAssoc
	if (tpOut.size()==0) {
	  for (TrackingParticle::g4t_iterator g4T=tprOld->g4Track_begin(); g4T!=tprOld->g4Track_end(); ++g4T) {
	    tpids.push_back(g4T->trackId());
	  }
	}
      }
    }

    if (tprOut.get()!=0 || tprOld.get()!=0) { //at least one of the tracks has to be associated

      tpr = tprOut.get()!=0 ? tprOut : tprOld;

      const SimTrack * assocTrack = &(*tpr->g4Track_begin());
      
      //if ( trackOut->numberOfValidHits() < trackOld->numberOfValidHits() ) {
      if ( trackOut->numberOfValidHits() != trackOld->numberOfValidHits() || 
	   !(*trackOut->recHitsBegin())->sharesInput((*trackOld->recHitsBegin()),TrackingRecHit::some) ||
	   !(*(trackOut->recHitsEnd()-1))->sharesInput((*(trackOld->recHitsEnd()-1)),TrackingRecHit::some)  ) 
	{ //there are outliers if the number of valid hits is != or if the first and last hit does not match
	LogTrace("TestOutliers") << "outliers for track with #hits=" << trackOut->numberOfValidHits();
	tracks->Fill(1);
	LogTrace("TestOutliers") << "Out->pt=" << trackOut->pt() << " Old->pt=" << trackOld->pt() 
				 << " tp->pt=" << sqrt(tpr->momentum().perp2()) 
	  //<< " trackOut->ptError()=" << trackOut->ptError() << " trackOld->ptError()=" << trackOld->ptError() 
				 << " Old->validHits=" << trackOld->numberOfValidHits() << " Out->validHits=" << trackOut->numberOfValidHits()
	  /*<< " fracOld=" << fracOld*/ << " fracOut=" << fracOut
				 << " deltaHits=" << trackOld->numberOfValidHits()-trackOut->numberOfValidHits();

	//compute all the track parameters	  
	double PtPullOut = (trackOut->pt()-sqrt(tpr->momentum().perp2()))/trackOut->ptError(); 
	double PtPullOld = (trackOld->pt()-sqrt(tpr->momentum().perp2()))/trackOld->ptError();
	histoPtOut->Fill( PtPullOut );
	histoPtOld->Fill( PtPullOld );

	//LogTrace("TestOutliers") << "MagneticField";		  
	edm::ESHandle<MagneticField> theMF;
	iSetup.get<IdealMagneticFieldRecord>().get(theMF);
	FreeTrajectoryState 
	  ftsAtProduction(GlobalPoint(tpr->vertex().x(),tpr->vertex().y(),tpr->vertex().z()),
			  GlobalVector(assocTrack->momentum().x(),assocTrack->momentum().y(),assocTrack->momentum().z()),
			  TrackCharge(trackOld->charge()),
			  theMF.product());
	TSCPBuilderNoMaterial tscpBuilder;
	TrajectoryStateClosestToPoint tsAtClosestApproach 
	  = tscpBuilder(ftsAtProduction,GlobalPoint(0,0,0));//as in TrackProducerAlgorithm
	GlobalPoint v = tsAtClosestApproach.theState().position();
	GlobalVector p = tsAtClosestApproach.theState().momentum();
		  
	//LogTrace("TestOutliers") << "qoverpSim";		  
	double qoverpSim = tsAtClosestApproach.charge()/p.mag();
	double lambdaSim = M_PI/2-p.theta();
	double phiSim    = p.phi();
	double dxySim    = (-v.x()*sin(p.phi())+v.y()*cos(p.phi()));
	double dszSim    = v.z()*p.perp()/p.mag() - (v.x()*p.x()+v.y()*p.y())/p.perp() * p.z()/p.mag();
	double d0Sim     = -dxySim;
	double dzSim     = dszSim*p.mag()/p.perp();
		  
	//LogTrace("TestOutliers") << "qoverpPullOut";		  
	double qoverpPullOut=(trackOut->qoverp()-qoverpSim)/trackOut->qoverpError();
	double qoverpPullOld=(trackOld->qoverp()-qoverpSim)/trackOld->qoverpError();
	double lambdaPullOut=(trackOut->lambda()-lambdaSim)/trackOut->thetaError();
	double lambdaPullOld=(trackOld->lambda()-lambdaSim)/trackOld->thetaError();
	double phi0PullOut=(trackOut->phi()-phiSim)/trackOut->phiError();
	double phi0PullOld=(trackOld->phi()-phiSim)/trackOld->phiError();
	double d0PullOut=(trackOut->d0()-d0Sim)/trackOut->d0Error();
	double d0PullOld=(trackOld->d0()-d0Sim)/trackOld->d0Error();
	double dzPullOut=(trackOut->dz()-dzSim)/trackOut->dzError();
	double dzPullOld=(trackOld->dz()-dzSim)/trackOld->dzError();
		  
	//LogTrace("TestOutliers") << "histoQoverpOut";		  
	histoQoverpOut->Fill(qoverpPullOut);
	histoQoverpOld->Fill(qoverpPullOld);
	histoPhiOut->Fill(phi0PullOut);  
	histoPhiOld->Fill(phi0PullOld);  
	histoD0Out->Fill(d0PullOut);  
	histoD0Old->Fill(d0PullOld);  
	histoDzOut->Fill(dzPullOut);  
	histoDzOld->Fill(dzPullOld);  
	histoLambdaOut->Fill(lambdaPullOut);
	histoLambdaOld->Fill(lambdaPullOld);

	//delta number of valid hits
	LogTrace("TestOutliers") << "deltahits=" << trackOld->numberOfValidHits()-trackOut->numberOfValidHits();		  
	deltahits->Fill(trackOld->numberOfValidHits()-trackOut->numberOfValidHits());

	if(tprOut.get()!=0 && tprOld.get()==0) { //out associated and old not: gained track
	  if (tpOld.size()!=0 && tpOld.begin()->second<=0.5) {
	    deltahitsAssocGained->Fill(trackOld->numberOfValidHits()-trackOut->numberOfValidHits());
	    hitsPerTrackAssocGained->Fill(trackOut->numberOfValidHits());
	    LogTrace("TestOutliers") << "a) gained (assoc) track out #hits==" << trackOut->numberOfValidHits() << " old #hits=" << trackOld->numberOfValidHits();    
	  } else {
	    deltahitsAssocGained->Fill(trackOld->numberOfValidHits()-trackOut->numberOfValidHits());
	    hitsPerTrackAssocGained->Fill(trackOut->numberOfValidHits());
	    LogTrace("TestOutliers") << "b) gained (assoc) track out #hits==" << trackOut->numberOfValidHits() << " old #hits=" << trackOld->numberOfValidHits();    
	  }
	} else if(tprOut.get()==0 && tprOld.get()!=0) { //old associated and out not: lost track
	  LogTrace("TestOutliers") <<"old associated and out not! old has #hits=" << trackOld->numberOfValidHits() 
				   << " and fraction=" << tpOld.begin()->second;
	  if (tpOld.begin()->second>0.5) {      
	    hitsPerTrackAssocLost->Fill(trackOld->numberOfValidHits());
	    deltahitsAssocLost->Fill(trackOld->numberOfValidHits()-trackOut->numberOfValidHits());
	  }
	}
	
	if ( fabs(PtPullOut) < fabs(PtPullOld) ) 
	  deltahitsOK->Fill(trackOld->numberOfValidHits()-trackOut->numberOfValidHits());
	else 
	  deltahitsNO->Fill(trackOld->numberOfValidHits()-trackOut->numberOfValidHits());
	
	//LogTrace("TestOutliers") << "RecoTrackSelector";		  
	//if (selectRecoTracks(*trackOut,beamSpot.product())) okcutsOut->Fill(1); else okcutsOut->Fill(0);
	//if (selectRecoTracks(*trackOld,beamSpot.product())) okcutsOld->Fill(1); else okcutsOld->Fill(0);

	LogTrace("TestOutliers") << "track old";
	for (trackingRecHit_iterator itOld = trackOld->recHitsBegin(); itOld!=trackOld->recHitsEnd() ; itOld++){
	  LogTrace("TestOutliers") << (*itOld)->isValid() << " " << (*itOld)->geographicalId().rawId();
	}
	LogTrace("TestOutliers") << "track out";
	for (trackingRecHit_iterator itOut = trackOut->recHitsBegin(); itOut!=trackOut->recHitsEnd() ; itOut++){
	  LogTrace("TestOutliers") << (*itOut)->isValid() << " " << (*itOut)->geographicalId().rawId();
	}
	//LogTrace("TestOutliers") << "itOut";		  


	vector<pair<int, trackingRecHit_iterator> > gainedlostoutliers;
	//look for gained hits
	for (trackingRecHit_iterator itOut = trackOut->recHitsBegin(); itOut!=trackOut->recHitsEnd(); itOut++){
	  bool gained = true;
	  if ((*itOut)->isValid()) {
	    for (trackingRecHit_iterator itOld = trackOld->recHitsBegin(); itOld!=trackOld->recHitsEnd() ; itOld++){
	      if ( (*itOld)->geographicalId().rawId()==(*itOut)->geographicalId().rawId() ) gained = false;
	    }
	    if (gained) {
	      gainedlostoutliers.push_back(pair<int, trackingRecHit_iterator>(1,itOut));
	      LogTrace("TestOutliers") << "broken trajectory during old fit... gained hit " << (*itOut)->geographicalId().rawId();
	      gainedhits->Fill(1);
	    }
	  }
	}

	//look for outliers and lost hits
	for (trackingRecHit_iterator itOld = trackOld->recHitsBegin(); itOld!=trackOld->recHitsEnd() ; itOld++){

	  bool outlier = false;
	  bool lost = true;

	  for (trackingRecHit_iterator itOut = trackOut->recHitsBegin(); itOut!=trackOut->recHitsEnd(); itOut++){
	    if ( (*itOld)->geographicalId().rawId()==(*itOut)->geographicalId().rawId() ) {
	      lost=false;
	      if ( (*itOld)->isValid() && !(*itOut)->isValid() && (*itOld)->geographicalId().rawId()==(*itOut)->geographicalId().rawId() ) {
		LogTrace("TestOutliers") << (*itOld)->isValid() << " " << (*itOut)->isValid() << " " 
					 << (*itOld)->geographicalId().rawId() << " " << (*itOut)->geographicalId().rawId();
		outlier=true;
	      }
	    }
	  }
	  if (lost) gainedlostoutliers.push_back(pair<int, trackingRecHit_iterator>(2,itOld));
	  if (lost) LogTrace("TestOutliers") << "lost";
	  else if (outlier) gainedlostoutliers.push_back(pair<int, trackingRecHit_iterator>(3,itOld));
	}

	for (std::vector<pair<int, trackingRecHit_iterator> >::iterator it = gainedlostoutliers.begin(); it!=gainedlostoutliers.end();++it) {
	  LogTrace("TestOutliers") << "type of processed hit:" <<it->first;
	  trackingRecHit_iterator itHit = it->second;
	  bool gained = false;
	  bool lost = false;
	  bool outlier = false;
	  if (it->first==1) gained = true;
	  else if (it->first==2) lost = true;
	  else if (it->first==3) outlier = true;

	  if (outlier||lost||gained) {
	  //if (1) {

	    if (lost && (*itHit)->isValid()==false) {
	      goodbadmergedLost->Fill(0);
	      LogTrace("TestOutliers") << "lost invalid";		  
	      continue;
	    }
	    else if (gained && (*itHit)->isValid()==false) {
	      goodbadmergedGained->Fill(0);
	      LogTrace("TestOutliers") << "gained invalid";
	      continue;
	    }
	      
	    //LogTrace("TestOutliers") << "vector<SimHitIdpr>";		  
	    //look if the hit comes from a correct sim track
	    std::vector<SimHitIdpr> simTrackIds = hitAssociator.associateHitId(**itHit);
	    bool goodhit = false;
	    for(size_t j=0; j<simTrackIds.size(); j++){
	      for (size_t jj=0; jj<tpids.size(); jj++){
		if (simTrackIds[j].first == tpids[jj]) goodhit = true;
		break;
	      }
	    }

	    //find what kind of hit is it
	    int clustersize = 0;
	    int hittypeval = 0;
	    int layerval = 0 ;
	    if (dynamic_cast<const SiPixelRecHit*>(&**itHit)){	
	      LogTrace("TestOutliers") << "SiPixelRecHit";		  
	      clustersize =  ((const SiPixelRecHit*)(&**itHit))->cluster()->size() ;
	      hittypeval  = 1;
	    }
	    else if (dynamic_cast<const SiStripRecHit2D*>(&**itHit)){
	      LogTrace("TestOutliers") << "SiStripRecHit2D";		  
	      clustersize =  ((const SiStripRecHit2D*)(&**itHit))->cluster()->amplitudes().size() ;
	      hittypeval  = 2;
	    }
	    else if (dynamic_cast<const SiStripMatchedRecHit2D*>(&**itHit)){
	      LogTrace("TestOutliers") << "SiStripMatchedRecHit2D";		  
	      int clsize1 = ((const SiStripMatchedRecHit2D*)(&**itHit))->monoCluster().amplitudes().size();
	      int clsize2 =  ((const SiStripMatchedRecHit2D*)(&**itHit))->stereoCluster().amplitudes().size();
	      if (clsize1>clsize2) clustersize = clsize1;
	      else clustersize = clsize2;
	      hittypeval  = 3;
	    }
	    else if (dynamic_cast<const ProjectedSiStripRecHit2D*>(&**itHit)){
	      LogTrace("TestOutliers") << "ProjectedSiStripRecHit2D";		  
	      clustersize =  ((const ProjectedSiStripRecHit2D*)(&**itHit))->originalHit().cluster()->amplitudes().size();
	      hittypeval  = 4;
	    }
	      
	    //find the layer of the hit
	    int subdetId = (*itHit)->geographicalId().subdetId();
	    DetId id = (*itHit)->geographicalId();
	    int layerId  = tTopo->layer(id);
	    layerval = subdetId*10+layerId;
      
	    //LogTrace("TestOutliers") << "gpos";		  
	    GlobalPoint gpos = theG->idToDet((*itHit)->geographicalId())->surface().toGlobal((*itHit)->localPosition());


	    //get the vector of sim hit associated and choose the one with the largest energy loss
	    //double delta = 99999;
	    //LocalPoint rhitLPv = (*itHit)->localPosition();
	    //vector<PSimHit> assSimHits = hitAssociator.associateHit(**itHit);
	    //if (assSimHits.size()==0) continue;
	    //PSimHit shit;
	    //for(std::vector<PSimHit>::const_iterator m=assSimHits.begin(); m<assSimHits.end(); m++){
	    //if ((m->localPosition()-rhitLPv).mag()<delta) {
	    //  shit=*m;
	    //  delta = (m->localPosition()-rhitLPv).mag();
	    // }
	    //}
	    //LogTrace("TestOutliers") << "energyLoss_";	   
	    double energyLoss_ = 0.;
	    unsigned int monoId = 0;
	    std::vector<double> energyLossM;
	    std::vector<double> energyLossS;
	    std::vector<PSimHit> assSimHits = hitAssociator.associateHit(**itHit);
	    if (assSimHits.size()==0) continue;
	    PSimHit shit;
	    std::vector<unsigned int> trackIds;
	    energyLossS.clear();
	    energyLossM.clear();
	    //LogTrace("TestOutliers") << "energyLossM";		  
	    for(std::vector<PSimHit>::const_iterator m=assSimHits.begin(); m<assSimHits.end(); m++){
	      if (outlier) energyLoss->Fill(m->energyLoss());
	      unsigned int tId = m->trackId();
	      if (find(trackIds.begin(),trackIds.end(),tId)==trackIds.end()) trackIds.push_back(tId);
	      LogTrace("TestOutliers") << "id=" << tId;
	      if (m->energyLoss()>energyLoss_) {
		shit=*m;
		energyLoss_ = m->energyLoss();
	      }
	      if (hittypeval==3) {
		if (monoId==0) monoId = m->detUnitId();
		if (monoId == m->detUnitId()){
		  energyLossM.push_back(m->energyLoss());
		}
		else {
		  energyLossS.push_back(m->energyLoss());
		}
		//std::cout << "detUnitId="  << m->detUnitId() << " trackId=" << m->trackId() << " energyLoss=" << m->energyLoss() << std::endl;
	      } else {
		energyLossM.push_back(m->energyLoss());
	      }
	    }
	    unsigned int nIds = trackIds.size();

	    if (outlier) {
	      goodbadhits->Fill(goodhit);
	      posxy->Fill(fabs(gpos.x()),fabs(gpos.y()));
	      poszr->Fill(fabs(gpos.z()),sqrt(gpos.x()*gpos.x()+gpos.y()*gpos.y()));
	      process->Fill(shit.processType());		      
	      energyLossMax->Fill(energyLoss_);
	    }
	      
	    //look if the hit is shared and if is produced only by ionization processes
	    bool shared = true;
	    bool ioniOnly = true;
	    unsigned int idc = 0;
	    for (size_t jj=0; jj<tpids.size(); jj++){
	      idc += std::count(trackIds.begin(),trackIds.end(),tpids[jj]);
	    }
	    if (idc==trackIds.size()) {
	      shared = false;
	    }
	    for(std::vector<PSimHit>::const_iterator m=assSimHits.begin()+1; m<assSimHits.end(); m++){
	      if ((m->processType()!=7&&m->processType()!=8&&m->processType()!=9)&&abs(m->particleType())!=11){
		ioniOnly = false;
		break;
	      }
	    }
	    if (ioniOnly&&!shared){
	      LogTrace("TestOutliers") << "delta";
	    }
	    
	    if (goodhit) { 
	      if (outlier) {
		goodprocess->Fill(shit.processType());
		if (clustersize>=4) {
		  goodhittype_clgteq4->Fill(hittypeval);
		  goodlayer_clgteq4->Fill(layerval);
		} else {
		  goodhittype_cllt4->Fill(hittypeval);
		  goodlayer_cllt4->Fill(layerval);
		}
		//LogTrace("TestOutliers") << "hittypeval && clustersize";		  
		if (hittypeval==1 && clustersize>=4) goodpixgteq4_simvecsize->Fill(assSimHits.size());
		if (hittypeval==1 && clustersize<4 ) goodpixlt4_simvecsize->Fill(assSimHits.size());
		if (hittypeval==2 && clustersize>=4) goodst1gteq4_simvecsize->Fill(assSimHits.size());
		if (hittypeval==2 && clustersize<4 ) goodst1lt4_simvecsize->Fill(assSimHits.size());
		if (hittypeval==3 && clustersize>=4) goodst2gteq4_simvecsize->Fill(assSimHits.size());
		if (hittypeval==3 && clustersize<4 ) goodst2lt4_simvecsize->Fill(assSimHits.size());
		if (hittypeval==4 && clustersize>=4) goodprjgteq4_simvecsize->Fill(assSimHits.size());
		if (hittypeval==4 && clustersize<4 ) goodprjlt4_simvecsize->Fill(assSimHits.size());
		  
		//LogTrace("TestOutliers") << "hittypeval";		  
		if (hittypeval==1) goodpix_clustersize->Fill(clustersize);
		if (hittypeval==2) goodst1_clustersize->Fill(clustersize);
		if (hittypeval==3) goodst2_clustersize->Fill(clustersize);
		if (hittypeval==4) goodprj_clustersize->Fill(clustersize);
		if (hittypeval==1) goodpix_simvecsize->Fill(assSimHits.size());
		if (hittypeval==2) goodst1_simvecsize->Fill(assSimHits.size());
		if (hittypeval==3) goodst2_simvecsize->Fill(assSimHits.size());
		if (hittypeval==4) goodprj_simvecsize->Fill(assSimHits.size());
				  
		//LogTrace("TestOutliers") << "nOfTrackIds";		  
		nOfTrackIds->Fill(nIds);
		if (hittypeval!=3) { 
		  if (energyLossM.size()>1) {
		    sort(energyLossM.begin(),energyLossM.end(),greater<double>());
		    energyLossRatio->Fill(energyLossM[1]/energyLossM[0]);
		  }
		} else {
		  //LogTrace("TestOutliers") << "hittypeval==3";		  
		  if (energyLossM.size()>1&&energyLossS.size()<=1) {
		    sort(energyLossM.begin(),energyLossM.end(),greater<double>());
		    energyLossRatio->Fill(energyLossM[1]/energyLossM[0]);
		  }
		  else if (energyLossS.size()>1&&energyLossM.size()<=1) {
		    sort(energyLossS.begin(),energyLossS.end(),greater<double>());
		    energyLossRatio->Fill(energyLossS[1]/energyLossS[0]);
		  }
		  else if (energyLossS.size()>1&&energyLossM.size()>1) {
		    sort(energyLossM.begin(),energyLossM.end(),greater<double>());
		    sort(energyLossS.begin(),energyLossS.end(),greater<double>());
		    energyLossM[1]/energyLossM[0] > energyLossS[1]/energyLossS[0] 
		      ? energyLossRatio->Fill(energyLossM[1]/energyLossM[0]) 
		      : energyLossRatio->Fill(energyLossS[1]/energyLossS[0]);
		  }
		}
		  
		LogTrace("TestOutliers") << "before merged";		  
		const SiStripMatchedRecHit2D* tmp = dynamic_cast<const SiStripMatchedRecHit2D*>(&**itHit);
		LogTrace("TestOutliers") << "tmp=" << tmp; 
		LogTrace("TestOutliers") << "assSimHits.size()=" << assSimHits.size(); 
		if ( (assSimHits.size()>1 && tmp==0) || 
		     (assSimHits.size()>2 && tmp!=0) ) {
		  //std::cout << "MERGED HIT" << std::endl;
		  //LogTrace("TestOutliers") << "merged";		  
		  mergedlayer->Fill(layerval);
		  mergedcluster->Fill(clustersize);
		  mergedhittype->Fill(hittypeval);

		  for(std::vector<PSimHit>::const_iterator m=assSimHits.begin(); m<assSimHits.end(); m++){
		    unsigned int tId = m->trackId();
		    LogTrace("TestOutliers") << "component with id=" << tId <<  " eLoss=" << m->energyLoss() 
					     << " proc=" <<  m->processType() << " part=" <<  m->particleType();
		    if (find(tpids.begin(),tpids.end(),tId)==tpids.end()) continue;
		    if (m->processType()==2) {
		      //GlobalPoint gpos = theG->idToDet((*itHit)->geographicalId())->surface().toGlobal((*itHit)->localPosition());
		      //GlobalPoint gpr = rhit->globalPosition();
		      AlgebraicSymMatrix33 ger = 
			ErrorFrameTransformer().transform((*itHit)->localPositionError(),theG->idToDet((*itHit)->geographicalId())->surface() ).matrix();
		      //AlgebraicSymMatrix ger = rhit->globalPositionError().matrix();
		      GlobalPoint gps = theG->idToDet((*itHit)->geographicalId())->surface().toGlobal(m->localPosition());
		      //LogTrace("TestOutliers") << gpr << " " << gps << " " << ger;
		      ROOT::Math::SVector<double,3> delta;
		      delta[0]=gpos.x()-gps.x();
		      delta[1]=gpos.y()-gps.y();
		      delta[2]=gpos.z()-gps.z();
		      LogTrace("TestOutliers") << delta << " " << ger ;
		      double mpull = sqrt(delta[0]*delta[0]/ger[0][0]+delta[1]*delta[1]/ger[1][1]+delta[2]*delta[2]/ger[2][2]);
		      LogTrace("TestOutliers") << "hit pull=" << mpull;//ger.similarity(delta);
		      mergedPull->Fill(mpull);
		    }
		  }
		  LogTrace("TestOutliers") << "end merged";
		} else {//not merged=>good
		  //LogTrace("TestOutliers") << "goodlayer";		  
		  goodlayer->Fill(layerval);		  
		  goodcluster->Fill(clustersize);
		  goodhittype->Fill(hittypeval);
		}	
	      }//if (outlier)

	      const SiPixelRecHit* pix = dynamic_cast<const SiPixelRecHit*>(&**itHit);
	      if ((hittypeval!=3 && assSimHits.size()<2)||(hittypeval==3 && assSimHits.size()<3)){
		if (outlier) {
		  goodhittype_simvecsmall->Fill(hittypeval);
		  goodlayer_simvecsmall->Fill(layerval);
		  goodbadmerged->Fill(1);
		  if (pix) {
		    probXgood->Fill(pix->probabilityX());
		    probYgood->Fill(pix->probabilityY());
		  }
		  LogTrace("TestOutliers") << "out good";		  
		}
		else if (lost){
		  goodbadmergedLost->Fill(1);
		  LogTrace("TestOutliers") << "lost good";		  
		}
		else if (gained){
		  goodbadmergedGained->Fill(1);
		  LogTrace("TestOutliers") << "gained good";		  
		}
		LogTrace("TestOutliers") << "good"; 
	      } else {
		if (outlier) {
		  goodhittype_simvecbig->Fill(hittypeval);
		  goodlayer_simvecbig->Fill(layerval);
		  if (ioniOnly&&!shared) { 
		    goodbadmerged->Fill(3);
		    if (pix) {
		      probXdelta->Fill(pix->probabilityX());
		      probYdelta->Fill(pix->probabilityY());
		    }
		  } else if(!ioniOnly&&!shared) {
		    goodbadmerged->Fill(4);
		    if (pix) {
		      probXnoshare->Fill(pix->probabilityX());
		      probYnoshare->Fill(pix->probabilityY());
		    }
		  } else {
		    goodbadmerged->Fill(5);
		    if (pix) {
		      probXshared->Fill(pix->probabilityX());
		      probYshared->Fill(pix->probabilityY());
		    }
		  }
		  LogTrace("TestOutliers") << "out merged, ioniOnly=" << ioniOnly << " shared=" << shared;
		}
		else if (lost) {
		  if (ioniOnly&&!shared) goodbadmergedLost->Fill(3);
		  else if(!ioniOnly&&!shared) goodbadmergedLost->Fill(4);
		  else goodbadmergedLost->Fill(5);
		  LogTrace("TestOutliers") << "lost merged, ioniOnly=" << ioniOnly << " shared=" << shared;
		}
		else if (gained) {
		  if (ioniOnly&&!shared) goodbadmergedGained->Fill(3);
		  else if(!ioniOnly&&!shared) goodbadmergedGained->Fill(4);
		  else goodbadmergedGained->Fill(5);
		  LogTrace("TestOutliers") << "gained merged, ioniOnly=" << ioniOnly << " shared=" << shared;
		}
		LogTrace("TestOutliers") << "merged, ioniOnly=" << ioniOnly << " shared=" << shared;
	      }
	    } //if (goodhit)
	    else {//badhit
	      //LogTrace("TestOutliers") << "badhit";	
	      if (outlier) {
		badcluster->Fill(clustersize);
		badhittype->Fill(hittypeval);
		badlayer->Fill(layerval);
		badprocess->Fill(shit.processType());
		goodbadmerged->Fill(2);
		const SiPixelRecHit* pix = dynamic_cast<const SiPixelRecHit*>(&**itHit);
		if (pix) {
		  probXbad->Fill(pix->probabilityX());
		  probYbad->Fill(pix->probabilityY());
		}
		LogTrace("TestOutliers") << "out bad";		  
	      }
	      else if (lost) {
		goodbadmergedLost->Fill(2);
		LogTrace("TestOutliers") << "lost bad";		  
	      }
	      else if (gained) {
		goodbadmergedGained->Fill(2);
		LogTrace("TestOutliers") << "gained bad";		  
	      }
	      LogTrace("TestOutliers") << "bad"; 
	    }
	  }
	}
      } 
      //else if ( trackOut->numberOfValidHits() > trackOld->numberOfValidHits() ) {
      else if ( 0 ) {
	LogTrace("TestOutliers") << "outliers for track with #hits=" << trackOut->numberOfValidHits();
	tracks->Fill(1);
	LogTrace("TestOutliers") << "Out->pt=" << trackOut->pt() << " Old->pt=" << trackOld->pt() 
				 << " tp->pt=" << sqrt(tpr->momentum().perp2()) 
	  //<< " trackOut->ptError()=" << trackOut->ptError() << " trackOld->ptError()=" << trackOld->ptError() 
				 << " Old->validHits=" << trackOld->numberOfValidHits() << " Out->validHits=" << trackOut->numberOfValidHits()
	  /*<< " fracOld=" << fracOld*/ << " fracOut=" << fracOut
				 << " deltaHits=" << trackOld->numberOfValidHits()-trackOut->numberOfValidHits();
	LogTrace("TestOutliers") << "track with gained hits";
	gainedhits2->Fill(trackOut->numberOfValidHits()-trackOld->numberOfValidHits());	
      } else {
	LogTrace("TestOutliers") << "no outliers for track with #hits=" << trackOut->numberOfValidHits();
      }
    }
    LogTrace("TestOutliers") << "end track old #" << j;
  }  

  for (unsigned int k=0;k<tracksOut->size();++k) {
    if ( find(outused.begin(),outused.end(),k)==outused.end() ) {
      edm::RefToBase<reco::Track> trackOut(tracksOut, k);      
      bool outAssoc = recSimCollOut.find(trackOut) != recSimCollOut.end();
      if (outAssoc) {
	hitsPerTrackGained->Fill(trackOut->numberOfValidHits());
	LogTrace("TestOutliers") << "gained track out id=" << k << " #hits==" << trackOut->numberOfValidHits();    

      }
    }    
  }
}


// ------------ method called once each job just before starting event loop  ------------
void 
TestOutliers::beginRun(edm::Run & run, const edm::EventSetup& es)
{
  es.get<TrackerDigiGeometryRecord>().get(theG);
  const bool oldAddDir = TH1::AddDirectoryStatus();
  TH1::AddDirectory(true);
  file = new TFile(out.c_str(),"recreate");
  histoPtOut = new TH1F("histoPtOut","histoPtOut",100,-10,10);
  histoPtOld = new TH1F("histoPtOld","histoPtOld",100,-10,10);
  histoQoverpOut = new TH1F("histoQoverpOut","histoQoverpOut",250,-25,25);
  histoQoverpOld = new TH1F("histoQoverpOld","histoQoverpOld",250,-25,25);
  histoPhiOut = new TH1F("histoPhiOut","histoPhiOut",250,-25,25);
  histoPhiOld = new TH1F("histoPhiOld","histoPhiOld",250,-25,25);
  histoD0Out = new TH1F("histoD0Out","histoD0Out",250,-25,25);
  histoD0Old = new TH1F("histoD0Old","histoD0Old",250,-25,25);
  histoDzOut = new TH1F("histoDzOut","histoDzOut",250,-25,25);
  histoDzOld = new TH1F("histoDzOld","histoDzOld",250,-25,25);
  histoLambdaOut = new TH1F("histoLambdaOut","histoLambdaOut",250,-25,25);
  histoLambdaOld = new TH1F("histoLambdaOld","histoLambdaOld",250,-25,25);
  deltahits = new TH1F("deltahits","deltahits",80,-40,40);
  deltahitsOK = new TH1F("deltahitsOK","deltahitsOK",20,0,20);
  deltahitsNO = new TH1F("deltahitsNO","deltahitsNO",20,0,20);
  okcutsOut = new TH1F("okcutsOut","okcutsOut",2,-0.5,1.5);
  okcutsOld = new TH1F("okcutsOld","okcutsOld",2,-0.5,1.5);
  tracks = new TH1F("tracks_","tracks_",2,-0.5,1.5);
  goodbadhits = new TH1F("goodbadhits","goodbadhits",2,-0.5,1.5);
  process = new TH1F("process","process",20,-0.5,19.5);
  goodcluster = new TH1F("goodcluster","goodcluster",40,-0.5,39.5);
  goodprocess = new TH1F("goodprocess","goodprocess",20,-0.5,19.5);
  badcluster = new TH1F("badcluster","badcluster",40,-0.5,39.5);
  badprocess = new TH1F("badprocess","badprocess",20,-0.5,19.5);
  goodhittype = new TH1F("goodhittype","goodhittype",5,-0.5,4.5);
  goodlayer = new TH1F("goodlayer","goodlayer",70,-0.5,69.5);
  goodhittype_clgteq4 = new TH1F("goodhittype_clgteq4","goodhittype_clgteq4",5,-0.5,4.5);
  goodlayer_clgteq4 = new TH1F("goodlayer_clgteq4","goodlayer_clgteq4",70,-0.5,69.5);
  goodhittype_cllt4 = new TH1F("goodhittype_cllt4","goodhittype_cllt4",5,-0.5,4.5);
  goodlayer_cllt4 = new TH1F("goodlayer_cllt4","goodlayer_cllt4",70,-0.5,69.5);
  badhittype = new TH1F("badhittype","badhittype",5,-0.5,4.5);
  badlayer = new TH1F("badlayer","badlayer",70,-0.5,69.5);
  posxy = new TH2F("posxy","posxy",1200,0,120,1200,0,120);
  poszr = new TH2F("poszr","poszr",3000,0,300,1200,0,120);
  goodpixgteq4_simvecsize = new TH1F("goodpixgteq4_simvecsize","goodpixgteq4_simvecsize",40,-0.5,39.5);
  goodpixlt4_simvecsize  = new TH1F("goodpixlt4_simvecsize","goodpixlt4_simvecsize",40,-0.5,39.5);
  goodst1gteq4_simvecsize = new TH1F("goodst1gteq4_simvecsize","goodst1gteq4_simvecsize",40,-0.5,39.5);
  goodst1lt4_simvecsize  = new TH1F("goodst1lt4_simvecsize","goodst1lt4_simvecsize",40,-0.5,39.5);
  goodst2gteq4_simvecsize = new TH1F("goodst2gteq4_simvecsize","goodst2gteq4_simvecsize",40,-0.5,39.5);
  goodst2lt4_simvecsize  = new TH1F("goodst2lt4_simvecsize","goodst2lt4_simvecsize",40,-0.5,39.5);
  goodprjgteq4_simvecsize = new TH1F("goodprjgteq4_simvecsize","goodprjgteq4_simvecsize",40,-0.5,39.5);
  goodprjlt4_simvecsize  = new TH1F("goodprjlt4_simvecsize","goodprjlt4_simvecsize",40,-0.5,39.5);
  goodpix_clustersize = new TH1F("goodpix_clustersize","goodpix_clustersize",40,-0.5,39.5);
  goodst1_clustersize = new TH1F("goodst1_clustersize","goodst1_clustersize",40,-0.5,39.5);
  goodst2_clustersize = new TH1F("goodst2_clustersize","goodst2_clustersize",40,-0.5,39.5);
  goodprj_clustersize = new TH1F("goodprj_clustersize","goodprj_clustersize",40,-0.5,39.5);
  goodpix_simvecsize = new TH1F("goodpix_simvecsize","goodpix_simvecsize",40,-0.5,39.5);
  goodst1_simvecsize = new TH1F("goodst1_simvecsize","goodst1_simvecsize",40,-0.5,39.5);
  goodst2_simvecsize = new TH1F("goodst2_simvecsize","goodst2_simvecsize",40,-0.5,39.5);
  goodprj_simvecsize = new TH1F("goodprj_simvecsize","goodprj_simvecsize",40,-0.5,39.5);
  goodhittype_simvecsmall = new TH1F("goodhittype_simvecsmall","goodhittype_simvecsmall",5,-0.5,4.5);
  goodlayer_simvecsmall = new TH1F("goodlayer_simvecsmall","goodlayer_simvecsmall",70,-0.5,69.5);
  goodhittype_simvecbig = new TH1F("goodhittype_simvecbig","goodhittype_simvecbig",5,-0.5,4.5);
  goodlayer_simvecbig = new TH1F("goodlayer_simvecbig","goodlayer_simvecbig",70,-0.5,69.5);
  goodbadmerged = new TH1F("goodbadmerged","goodbadmerged",5,0.5,5.5);
  goodbadmergedLost = new TH1F("goodbadmergedLost","goodbadmergedLost",5,0.5,5.5);
  goodbadmergedGained = new TH1F("goodbadmergedGained","goodbadmergedGained",5,0.5,5.5);
  energyLoss = new TH1F("energyLoss","energyLoss",1000,0,0.1);
  energyLossMax = new TH1F("energyLossMax","energyLossMax",1000,0,0.1);
  energyLossRatio = new TH1F("energyLossRatio","energyLossRatio",100,0,1);
  nOfTrackIds = new TH1F("nOfTrackIds","nOfTrackIds",10,0,10);
  mergedPull = new TH1F("mergedPull","mergedPull",100,0,10);
  mergedlayer = new TH1F("mergedlayer","mergedlayer",70,-0.5,69.5);
  mergedhittype = new TH1F("mergedhittype","mergedhittype",5,-0.5,4.5);
  mergedcluster = new TH1F("mergedcluster","mergedcluster",40,-0.5,39.5);
  deltahitsAssocGained = new TH1F("deltahitsAssocGained","deltahitsAssocGained",80, -40, 40);
  deltahitsAssocLost = new TH1F("deltahitsAssocLost","deltahitsAssocLost",80, -40, 40);
  hitsPerTrackLost = new TH1F("hitsPerTrackLost","hitsPerTrackLost",40, -0.5, 39.5);
  hitsPerTrackAssocLost = new TH1F("hitsPerTrackAssocLost","hitsPerTrackAssocLost",40, -0.5, 39.5);
  hitsPerTrackGained = new TH1F("hitsPerTrackGained","hitsPerTrackGained",40, -0.5, 39.5);
  hitsPerTrackAssocGained = new TH1F("hitsPerTrackAssocGained","hitsPerTrackAssocGained",40, -0.5, 39.5);
  sizeOut = new TH1F("sizeOut","sizeOut",900,-0.5,899.5);
  sizeOld = new TH1F("sizeOld","sizeOld",900,-0.5,899.5);
  sizeOutT = new TH1F("sizeOutT","sizeOutT",900,-0.5,899.5);
  sizeOldT = new TH1F("sizeOldT","sizeOldT",900,-0.5,899.5);
  countOutA = new TH1F("countOutA","countOutA",2,0,2);
  countOutT = new TH1F("countOutT","countOutT",2,0,2);
  countOldT = new TH1F("countOldT","countOldT",2,0,2);
  gainedhits = new TH1F("gainedhits","gainedhits",2,0,2);
  gainedhits2 = new TH1F("gainedhits2","gainedhits2",30,-0.5,29.5);
  probXgood = new TH1F("probXgood","probXgood",110,0,1.1);
  probXbad = new TH1F("probXbad","probXbad",110,0,1.1);
  probXdelta = new TH1F("probXdelta","probXdelta",110,0,1.1);
  probXshared = new TH1F("probXshared","probXshared",110,0,1.1);
  probXnoshare = new TH1F("probXnoshare","probXnoshare",110,0,1.1);
  probYgood = new TH1F("probYgood","probYgood",110,0,1.1);
  probYbad = new TH1F("probYbad","probYbad",110,0,1.1);
  probYdelta = new TH1F("probYdelta","probYdelta",110,0,1.1);
  probYshared = new TH1F("probYshared","probYshared",110,0,1.1);
  probYnoshare = new TH1F("probYnoshare","probYnoshare",110,0,1.1);
  TH1::AddDirectory(oldAddDir);
}
// ------------ method called once each job just after ending the event loop  ------------
void 
TestOutliers::endJob() {
  LogTrace("TestOutliers") << "TestOutliers::endJob";
  file->Write();
  LogTrace("TestOutliers") << "outfile written";
  file->Close();
  LogTrace("TestOutliers") << "oufile closed";
  LogTrace("TestOutliers") << "exiting TestOutliers::endJob";
}

//define this as a plug-in
DEFINE_FWK_MODULE(TestOutliers);
