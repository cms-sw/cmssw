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
// $Id: TestOutliers.cc,v 1.2 2008/02/13 16:04:10 cerati Exp $
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
#include "FWCore/ParameterSet/interface/InputTag.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "SimTracker/TrackAssociation/interface/TrackAssociatorByHits.h"
#include "SimTracker/TrackerHitAssociation/interface/TrackerHitAssociator.h"
#include "SimTracker/Records/interface/TrackAssociatorRecord.h"
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
#include "DataFormats/SiStripDetId/interface/TIBDetId.h"
#include "DataFormats/SiStripDetId/interface/TOBDetId.h"
#include "DataFormats/SiStripDetId/interface/TIDDetId.h"
#include "DataFormats/SiStripDetId/interface/TECDetId.h"
#include "DataFormats/SiPixelDetId/interface/PXBDetId.h"
#include "DataFormats/SiPixelDetId/interface/PXFDetId.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/ErrorFrameTransformer.h"
#include "PhysicsTools/RecoAlgos/interface/RecoTrackSelector.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"

//
// class decleration
//

class TestOutliers : public edm::EDAnalyzer {
public:
  explicit TestOutliers(const edm::ParameterSet&);
  ~TestOutliers();


private:
  virtual void beginJob(const edm::EventSetup&) ;
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void endJob() ;

  // ----------member data ---------------------------
  edm::InputTag trackTagsOut_; //used to select what tracks to read from configuration file
  edm::InputTag trackTagsOld_; //used to select what tracks to read from configuration file
  edm::InputTag tpTags_; //used to select what tracks to read from configuration file
  TrackAssociatorBase * theAssociatorOld;
  TrackAssociatorBase * theAssociatorOut;
  RecoTrackSelector selectRecoTracks;
  TrackerHitAssociator* hitAssociator;
  edm::ESHandle<TrackerGeometry> theG;
  std::string out;
  TFile * file;
  TH1F *histoPtOut, *histoPtOld ;
  TH1F *histoQoverpOut,*histoQoverpOld;
  TH1F *histoPhiOut,*histoPhiOld;
  TH1F *histoD0Out,*histoD0Old;
  TH1F *histoDzOut,*histoDzOld;
  TH1F *histoLambdaOut,*histoLambdaOld;
  TH1F *deltahits,*deltahitsOK,*deltahitsNO;
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
  TH1F *goodhittype_simvecsmall, *goodlayer_simvecsmall, *goodhittype_simvecbig, *goodlayer_simvecbig, *goodbadmerged;
  TH1F *energyLoss,*energyLossMax,*energyLossRatio, *nOfTrackIds, *mergedPull, *mergedlayer, *mergedcluster, *mergedhittype;
  edm::ParameterSet psetold,psetout;
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
  out(iConfig.getParameter<std::string>("out"))
{
  LogTrace("TestOutliers") <<"constructor";
  ParameterSet cuts = iConfig.getParameter<ParameterSet>("RecoTracksCuts");
  selectRecoTracks = RecoTrackSelector(cuts.getParameter<double>("ptMin"),
				       cuts.getParameter<double>("minRapidity"),
				       cuts.getParameter<double>("maxRapidity"),
				       cuts.getParameter<double>("tip"),
				       cuts.getParameter<double>("lip"),
				       cuts.getParameter<int>("minHit"),
				       cuts.getParameter<double>("maxChi2"),
				       cuts.getParameter<std::string>("quality"));
  
  psetold = iConfig.getParameter<ParameterSet>("TrackAssociatorByHitsPSetOld");
  psetout = iConfig.getParameter<ParameterSet>("TrackAssociatorByHitsPSetOut");
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

  hitAssociator = new TrackerHitAssociator::TrackerHitAssociator(iEvent);

  theAssociatorOld = new TrackAssociatorByHits(psetold);
  theAssociatorOut = new TrackAssociatorByHits(psetout);
  reco::RecoToSimCollection recSimCollOut=theAssociatorOut->associateRecoToSim(tracksOut, tps, &iEvent);
  reco::RecoToSimCollection recSimCollOld=theAssociatorOld->associateRecoToSim(tracksOld, tps, &iEvent);

#if 0
  LogTrace("TestOutliers") << "recSimCollOld.size()=" << recSimCollOld.size() ;
  for(reco::TrackCollection::size_type j=0; j<tracksOld->size(); ++j){
    reco::TrackRef trackOld(tracksOld, j);
    if ( !selectRecoTracks( *trackOld,beamSpot.product() ) ) continue;
    LogTrace("TestOutliers") << "trackOld->pt()=" << trackOld->pt() << " trackOld->numberOfValidHits()=" << trackOld->numberOfValidHits();
    vector<pair<TrackingParticleRef, double> > tpOld;
    if(recSimCollOld.find(trackOld) != recSimCollOld.end()){
      tpOld = recSimCollOld[trackOld];
      if (tpOld.size()!=0) LogTrace("TestOutliers") << " associated" ;
      else LogTrace("TestOutliers") << " NOT associated" ;
    } else LogTrace("TestOutliers") << " NOT associated" ;
  }
  LogTrace("TestOutliers") << "recSimCollOut.size()=" << recSimCollOut.size() ;
  for(reco::TrackCollection::size_type j=0; j<tracksOut->size(); ++j){
    reco::TrackRef trackOut(tracksOut, j);
    if ( !selectRecoTracks( *trackOut,beamSpot.product() ) ) continue;
    LogTrace("TestOutliers") << "trackOut->pt()=" << trackOut->pt() << " trackOut->numberOfValidHits()=" << trackOut->numberOfValidHits();
    vector<pair<TrackingParticleRef, double> > tpOut;
    if(recSimCollOut.find(trackOut) != recSimCollOut.end()){
      tpOut = recSimCollOut[trackOut];
      if (tpOut.size()!=0) LogTrace("TestOutliers") << " associated" ;
      else LogTrace("TestOutliers") << " NOT associated" ;
    } else LogTrace("TestOutliers") << " NOT associated" ;
  }
#endif

  unsigned int i=0;
  for(unsigned int j=0; j<tracksOld->size(); ++j) {
    edm::RefToBase<reco::Track> trackOld(tracksOld, j);
    edm::RefToBase<reco::Track> trackOut(tracksOut, i);
    if ( (trackOut->numberOfValidHits()+trackOut->numberOfLostHits()) 
	 != (trackOld->numberOfValidHits()+trackOld->numberOfLostHits()) ) continue;
    ++i;

    if ( !selectRecoTracks( *trackOld,beamSpot.product() ) ) continue;//????

    tracks->Fill(0);//FIXME

    bool hasOut = false;
	
    std::vector<std::pair<TrackingParticleRef, double> > tpOut;
    if(recSimCollOut.find(trackOut) != recSimCollOut.end()) {
      tpOut = recSimCollOut[trackOut];
      if (tpOut.size()!=0) {
	TrackingParticleRef tprOut = tpOut.begin()->first;
	double fracOut = tpOut.begin()->second;
	vector<unsigned int> tpids;
        for (TrackingParticle::g4t_iterator g4T=tprOut->g4Track_begin(); g4T!=tprOut->g4Track_end(); ++g4T) {
          tpids.push_back(g4T->trackId());
        }
 	const SimTrack * assocTrack = &(*tprOut->g4Track_begin());
	      
	// 		LogTrace("TestOutliers") << trackOut->numberOfValidHits()+trackOut->numberOfLostHits() << " " 
	// 				  << trackOld->numberOfValidHictfWithMaterialTracks::redots()+trackOld->numberOfLostHits() << " \t" 
	// 				  << trackOut->numberOfValidHits() << " " << trackOld->numberOfValidHits();
	if ( trackOut->numberOfValidHits() < trackOld->numberOfValidHits() ) {
	  //== (foundOld+lostOld) && trackOut->numberOfValidHits() < foundOld) {
	  hasOut=true;
	  tracks->Fill(1);
	  TrackingParticleRef tpr = tprOut;
	  LogTrace("TestOutliers") << "Out->pt=" << trackOut->pt() << " Old->pt=" << trackOld->pt() 
			    << " tp->pt=" << sqrt(tpr->momentum().perp2()) 
	    //<< " trackOut->ptError()=" << trackOut->ptError() << " trackOld->ptError()=" << trackOld->ptError() 
			    << " Old->validHits=" << trackOld->numberOfValidHits() << " Out->validHits=" << trackOut->numberOfValidHits()
	    /*<< " fracOld=" << fracOld*/ << " fracOut=" << fracOut
			    << " deltaHits=" << trackOld->numberOfValidHits()-trackOut->numberOfValidHits();
	  
	  double PtPullOut = (trackOut->pt()-sqrt(tpr->momentum().perp2()))/trackOut->ptError(); 
	  double PtPullOld = (trackOld->pt()-sqrt(tpr->momentum().perp2()))/trackOld->ptError();
	  histoPtOut->Fill( PtPullOut );
	  histoPtOld->Fill( PtPullOld );
		  
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
		  
	  double qoverpSim = tsAtClosestApproach.charge()/p.mag();
	  double lambdaSim = M_PI/2-p.theta();
	  double phiSim    = p.phi();
	  double dxySim    = (-v.x()*sin(p.phi())+v.y()*cos(p.phi()));
	  double dszSim    = v.z()*p.perp()/p.mag() - (v.x()*p.x()+v.y()*p.y())/p.perp() * p.z()/p.mag();
	  double d0Sim     = -dxySim;
	  double dzSim     = dszSim*p.mag()/p.perp();
		  
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

	  deltahits->Fill(trackOld->numberOfValidHits()-trackOut->numberOfValidHits());
	  //deltahits->Fill(foundOld-trackOut->numberOfValidHits());
	  if ( fabs(PtPullOut) < fabs(PtPullOld) ) 
	    deltahitsOK->Fill(trackOld->numberOfValidHits()-trackOut->numberOfValidHits());
	  //deltahitsOK->Fill(foundOld-trackOut->numberOfValidHits());
	  else 
	    deltahitsNO->Fill(trackOld->numberOfValidHits()-trackOut->numberOfValidHits());
	  //deltahitsNO->Fill(foundOld-trackOut->numberOfValidHits());

	  RecoTrackSelector select(0.8, -2.5, 2.5, 3.5, 30, 5, 10000, "loose");
	  if (select(*trackOut,beamSpot.product())) okcutsOut->Fill(1); else okcutsOut->Fill(0);
	  if (select(*trackOld,beamSpot.product())) okcutsOld->Fill(1); else okcutsOld->Fill(0);

	  trackingRecHit_iterator itOut = trackOut->recHitsBegin();
	  for (trackingRecHit_iterator itOld = trackOld->recHitsBegin(); itOld!=trackOld->recHitsEnd(); itOld++,itOut++){
	    if ( (*itOld)->isValid() && !(*itOut)->isValid() ) {

	      std::vector<SimHitIdpr> simTrackIds = hitAssociator->associateHitId(**itOld);
	      bool goodhit = false;
	      for(size_t j=0; j<simTrackIds.size(); j++){
		for (size_t jj=0; jj<tpids.size(); jj++){
		  if (simTrackIds[j].first == tpids[jj]) goodhit = true;
		  break;
		}
	      }
	      goodbadhits->Fill(goodhit);
	      int clustersize = 0;
	      int hittypeval = 0;
	      int layerval = 0 ;
	      if (dynamic_cast<const SiPixelRecHit*>(&**itOld)){	
		clustersize =  ((const SiPixelRecHit*)(&**itOld))->cluster()->size() ;
		hittypeval  = 1;
	      }
	      else if (dynamic_cast<const SiStripRecHit2D*>(&**itOld)){
		clustersize =  ((const SiStripRecHit2D*)(&**itOld))->cluster()->amplitudes().size() ;
		hittypeval  = 2;
	      }
	      else if (dynamic_cast<const SiStripMatchedRecHit2D*>(&**itOld)){
		int clsize1 = ((const SiStripMatchedRecHit2D*)(&**itOld))->monoHit()->cluster()->amplitudes().size();
		int clsize2 =  ((const SiStripMatchedRecHit2D*)(&**itOld))->stereoHit()->cluster()->amplitudes().size();
		if (clsize1>clsize2) clustersize = clsize1;
		else clustersize = clsize2;
		hittypeval  = 3;
	      }
	      else if (dynamic_cast<const ProjectedSiStripRecHit2D*>(&**itOld)){
		clustersize =  ((const ProjectedSiStripRecHit2D*)(&**itOld))->originalHit().cluster()->amplitudes().size();
		hittypeval  = 4;
	      }
		      
	      int subdetId = (*itOld)->geographicalId().subdetId();
	      int layerId  = 0;
	      DetId id = (*itOld)->geographicalId();
	      if (id.subdetId()==3) layerId = ((TIBDetId)(id)).layer();
	      if (id.subdetId()==5) layerId = ((TOBDetId)(id)).layer();
	      if (id.subdetId()==1) layerId = ((PXBDetId)(id)).layer();
	      if (id.subdetId()==4) layerId = ((TIDDetId)(id)).wheel();
	      if (id.subdetId()==6) layerId = ((TECDetId)(id)).wheel();
	      if (id.subdetId()==2) layerId = ((PXFDetId)(id)).disk();
	      layerval = subdetId*10+layerId;
		      
	      GlobalPoint gpos = theG->idToDet((*itOld)->geographicalId())->surface().toGlobal((*itOld)->localPosition());
	      posxy->Fill(fabs(gpos.x()),fabs(gpos.y()));
	      poszr->Fill(fabs(gpos.z()),sqrt(gpos.x()*gpos.x()+gpos.y()*gpos.y()));
		      
	      //double delta = 99999;
	      //LocalPoint rhitLPv = (*itOld)->localPosition();
	      //vector<PSimHit> assSimHits = hitAssociator->associateHit(**itOld);
	      //if (assSimHits.size()==0) continue;
	      //PSimHit shit;
	      //for(vector<PSimHit>::const_iterator m=assSimHits.begin(); m<assSimHits.end(); m++){
	      //if ((m->localPosition()-rhitLPv).mag()<delta) {
	      //  shit=*m;
	      //  delta = (m->localPosition()-rhitLPv).mag();
	      // }
	      //}
	      double energyLoss_ = 0.;
	      unsigned int monoId = 0;
	      vector<double> energyLossM;
	      vector<double> energyLossS;
	      vector<PSimHit> assSimHits = hitAssociator->associateHit(**itOld);
	      if (assSimHits.size()==0) continue;
	      PSimHit shit;
	      vector<unsigned int> trackIds;
	      energyLossS.clear();
	      energyLossM.clear();
	      for(vector<PSimHit>::const_iterator m=assSimHits.begin(); m<assSimHits.end(); m++){
		unsigned int tId = m->trackId();
		if (find(trackIds.begin(),trackIds.end(),tId)==trackIds.end()) trackIds.push_back(tId);
		energyLoss->Fill(m->energyLoss());
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
		  //cout << "detUnitId="  << m->detUnitId() << " trackId=" << m->trackId() << " energyLoss=" << m->energyLoss() << endl;
		} else {
		  energyLossM.push_back(m->energyLoss());
		}
	      }
	      unsigned int nIds = trackIds.size();
	      process->Fill(shit.processType());		      
	      energyLossMax->Fill(energyLoss_);

	      if (goodhit) { 
		goodprocess->Fill(shit.processType());
		if (clustersize>=4) {
		  goodhittype_clgteq4->Fill(hittypeval);
		  goodlayer_clgteq4->Fill(layerval);
		} else {
		  goodhittype_cllt4->Fill(hittypeval);
		  goodlayer_cllt4->Fill(layerval);
		}
		if (hittypeval==1 && clustersize>=4) goodpixgteq4_simvecsize->Fill(assSimHits.size());
		if (hittypeval==1 && clustersize<4 ) goodpixlt4_simvecsize->Fill(assSimHits.size());
		if (hittypeval==2 && clustersize>=4) goodst1gteq4_simvecsize->Fill(assSimHits.size());
		if (hittypeval==2 && clustersize<4 ) goodst1lt4_simvecsize->Fill(assSimHits.size());
		if (hittypeval==3 && clustersize>=4) goodst2gteq4_simvecsize->Fill(assSimHits.size());
		if (hittypeval==3 && clustersize<4 ) goodst2lt4_simvecsize->Fill(assSimHits.size());
		if (hittypeval==4 && clustersize>=4) goodprjgteq4_simvecsize->Fill(assSimHits.size());
		if (hittypeval==4 && clustersize<4 ) goodprjlt4_simvecsize->Fill(assSimHits.size());

		if (hittypeval==1) goodpix_clustersize->Fill(clustersize);
		if (hittypeval==2) goodst1_clustersize->Fill(clustersize);
		if (hittypeval==3) goodst2_clustersize->Fill(clustersize);
		if (hittypeval==4) goodprj_clustersize->Fill(clustersize);
		if (hittypeval==1) goodpix_simvecsize->Fill(assSimHits.size());
		if (hittypeval==2) goodst1_simvecsize->Fill(assSimHits.size());
		if (hittypeval==3) goodst2_simvecsize->Fill(assSimHits.size());
		if (hittypeval==4) goodprj_simvecsize->Fill(assSimHits.size());

		if ((hittypeval!=3 && assSimHits.size()<2)||(hittypeval==3 && assSimHits.size()<3)){
		  goodhittype_simvecsmall->Fill(hittypeval);
		  goodlayer_simvecsmall->Fill(layerval);
		  goodbadmerged->Fill(1);
		} else {
		  goodhittype_simvecbig->Fill(hittypeval);
		  goodlayer_simvecbig->Fill(layerval);
		  goodbadmerged->Fill(3);
		}

		nOfTrackIds->Fill(nIds);
		if (hittypeval!=3) { 
		  if (energyLossM.size()>1) {
		    sort(energyLossM.begin(),energyLossM.end(),greater<double>());
		    energyLossRatio->Fill(energyLossM[1]/energyLossM[0]);
		  }
		} else {
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

		if ( (assSimHits.size()>1 && dynamic_cast<const SiStripMatchedRecHit2D*>(&**itOld)==0) || 
		     (assSimHits.size()>2 && dynamic_cast<const SiStripMatchedRecHit2D*>(&**itOld)!=0) ) {
		  //cout << "MERGED HIT" << endl;
		  mergedlayer->Fill(layerval);
		  mergedcluster->Fill(clustersize);
		  mergedhittype->Fill(hittypeval);
		  for(vector<PSimHit>::const_iterator m=assSimHits.begin(); m<assSimHits.end(); m++){
		    unsigned int tId = m->trackId();
		    //cout << "component with id=" << tId <<  " eLoss=" << m->energyLoss() << " pType=" <<  m->processType() << endl;
		    if (find(tpids.begin(),tpids.end(),tId)==tpids.end()) continue;
		    if (m->processType()==2) {
		      //GlobalPoint gpos = theG->idToDet((*itOld)->geographicalId())->surface().toGlobal((*itOld)->localPosition());
		      //GlobalPoint gpr = rhit->globalPosition();
		      AlgebraicSymMatrix ger = 
			ErrorFrameTransformer().transform((*itOld)->localPositionError(),theG->idToDet((*itOld)->geographicalId())->surface() ).matrix();
		      //AlgebraicSymMatrix ger = rhit->globalPositionError().matrix();
		      GlobalPoint gps = theG->idToDet((*itOld)->geographicalId())->surface().toGlobal(m->localPosition());
		      //LogVerbatim("TestTrackHits") << gpr << " " << gps << " " << ger;
		      HepVector delta(3);
		      delta[0]=gpos.x()-gps.x();
		      delta[1]=gpos.y()-gps.y();
		      delta[2]=gpos.z()-gps.z();
		      //LogVerbatim("TestTrackHits") << delta << " " << ger ;
		      double mpull = sqrt(delta[0]*delta[0]/ger[0][0]+delta[1]*delta[1]/ger[1][1]+delta[2]*delta[2]/ger[2][2]);
		      //cout << "hit pull=" << mpull << endl;//ger.similarity(delta);
		      mergedPull->Fill(mpull);
		    }
		  }
		} else {
		  goodlayer->Fill(layerval);		  
		  goodcluster->Fill(clustersize);
		  goodhittype->Fill(hittypeval);
		}			
	      } else {//badhit
		badcluster->Fill(clustersize);
		badhittype->Fill(hittypeval);
		badlayer->Fill(layerval);
		badprocess->Fill(shit.processType());
		goodbadmerged->Fill(2);
	      }
	    }
	  }
	  continue;
	} 
      }
      //     }
      //   }
      //       if (!hasOut && fracOld>=0.5)  LogTrace("TestOutliers") << "No Out track for Old track with hits #" << trackOld->numberOfValidHits() << " fracOld=" << fracOld;
    }
  }  
  delete hitAssociator;
}


// ------------ method called once each job just before starting event loop  ------------
void 
TestOutliers::beginJob(const edm::EventSetup& es)
{
  es.get<TrackerDigiGeometryRecord>().get(theG);
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
  deltahits = new TH1F("deltahits","deltahits",20,0,20);
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
  goodbadmerged = new TH1F("goodbadmerged","goodbadmerged",3,0.5,3.5);
  energyLoss = new TH1F("energyLoss","energyLoss",1000,0,0.1);
  energyLossMax = new TH1F("energyLossMax","energyLossMax",1000,0,0.1);
  energyLossRatio = new TH1F("energyLossRatio","energyLossRatio",100,0,1);
  nOfTrackIds = new TH1F("nOfTrackIds","nOfTrackIds",10,0,10);
  mergedPull = new TH1F("mergedPull","mergedPull",100,0,10);
  mergedlayer = new TH1F("mergedlayer","mergedlayer",70,-0.5,69.5);
  mergedhittype = new TH1F("mergedhittype","mergedhittype",5,-0.5,4.5);
  mergedcluster = new TH1F("mergedcluster","mergedcluster",40,-0.5,39.5);
}
// ------------ method called once each job just after ending the event loop  ------------
void 
TestOutliers::endJob() {
  file->Write();
  file->Close();
}

//define this as a plug-in
DEFINE_FWK_MODULE(TestOutliers);
