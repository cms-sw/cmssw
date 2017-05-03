// -*- C++ -*-
//
// Package:    TrackAnalyzer
// Class:      TrackAnalyzer
//
/**\class TrackAnalyzer TrackAnalyzer.cc MitHig/TrackAnalyzer/src/TrackAnalyzer.cc

   Description: <one line class summary>

   Implementation:
   Prepare the Track Tree for analysis
*/
//
// Original Author:  Yilmaz Yetkin, Yen-Jie Lee
// Updated: Frank Ma, Matt Nguyen
//         Created:  Tue Sep 30 15:14:28 CEST 2008
// $Id: TrackAnalyzer.cc,v 1.55 2013/06/11 20:58:09 yjlee Exp $

// system include files
#include <memory>
#include <iostream>
#include <vector>
#include <string>
#include <map>
#include <functional>

// CMSSW user include files
#include "DataFormats/Common/interface/DetSetAlgorithm.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/SiPixelDetId/interface/PXBDetId.h"
#include "DataFormats/SiPixelDetId/interface/PXFDetId.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHitCollection.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetUnit.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
//#include "Geometry/TrackerGeometryBuilder/interface/TrackerLayerIdAccessor.h"
#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingVertex.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingVertexContainer.h"
#include "SimDataFormats/TrackingHit/interface/PSimHit.h"
#include "SimGeneral/HepPDTRecord/interface/ParticleDataTable.h"
// #include "SimTracker/TrackerHitAssociation/interface/TrackerHitAssociator.h"
#include "DataFormats/SiPixelDetId/interface/PixelEndcapName.h"

// #include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutSetup.h"
// #include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutRecord.h"
// #include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerObjectMapRecord.h"
// #include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerObjectMap.h"
// #include "L1Trigger/GlobalTrigger/interface/L1GlobalTrigger.h"

#include "DataFormats/Math/interface/Point3D.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"

#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticleFwd.h"
#include "SimTracker/Records/interface/TrackAssociatorRecord.h"
#include "DataFormats/RecoCandidate/interface/TrackAssociation.h"
// #include "SimTracker/TrackAssociation/interface/TrackAssociatorByHits.h"
// #include "SimTracker/TrackAssociation/interface/TrackAssociatorByChi2.h"
#include "DataFormats/TrackReco/interface/DeDxData.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"

#include "HeavyIonsAnalysis/TrackAnalysis/interface/TrkAnalyzerUtils.h"

// Particle Flow
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlock.h"
#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"
#include "DataFormats/ParticleFlowReco/interface/PFClusterFwd.h"

// Vertex significance
#include "RecoBTag/SecondaryVertex/interface/SecondaryVertex.h"

// Root include files
#include "TTree.h"

using namespace std;
using namespace edm;
using namespace reco;

//
// class decleration
//

#define PI 3.14159265358979

#define MAXTRACKS 60000
#define MAXVTX 100
#define MAXQUAL 5
#define MAXMATCH 5

const HepMC::GenParticle * getGpMother(const HepMC::GenParticle *gp) {
  if (gp != 0) {
    const HepMC::GenVertex *vtx = gp->production_vertex();
    if (vtx != 0 && vtx->particles_in_size() > 0) {
      return *vtx->particles_in_const_begin();
    }
  }
  return 0;
}
struct TrackEvent{

  // event information
  int nRun;
  int nEv;
  int nLumi;
  int nBX;
  int N; // multiplicity variable

  // Vertex information
  int nVtx;

  int nTrkVtx[MAXVTX];
  float normChi2Vtx[MAXVTX];
  float sumPtVtx[MAXVTX];
  //int nTrkVtxHard[MAXVTX];
  int maxVtx;
  int maxPtVtx;
  int maxMultVtx;
  //int maxVtxHard;

  float xVtx[MAXVTX];
  float yVtx[MAXVTX];
  float zVtx[MAXVTX];
  float xVtxErr[MAXVTX];
  float yVtxErr[MAXVTX];
  float zVtxErr[MAXVTX];

  float vtxDist2D[MAXVTX];
  float vtxDist2DErr[MAXVTX];
  float vtxDist2DSig[MAXVTX];
  float vtxDist3D[MAXVTX];
  float vtxDist3DErr[MAXVTX];
  float vtxDist3DSig[MAXVTX];

  int nVtxSim;
  float xVtxSim[MAXVTX];
  float yVtxSim[MAXVTX];
  float zVtxSim[MAXVTX];

  // -- rec tracks --
  int nTrk;
  float trkEta[MAXTRACKS];
  float trkPhi[MAXTRACKS];
  float trkPt[MAXTRACKS];
  float trkPtError[MAXTRACKS];
  unsigned char trkNHit[MAXTRACKS];
  unsigned char trkNlayer[MAXTRACKS];
  int trkNlayer3D[MAXTRACKS];
  bool trkQual[MAXQUAL][MAXTRACKS];
  float trkChi2[MAXTRACKS];
  float trkChi2hit1D[MAXTRACKS];
  unsigned char trkNdof[MAXTRACKS];
  float trkDz[MAXTRACKS];
  float trkDz1[MAXTRACKS];               // dZ to the highest pt vertex
  float trkDz2[MAXTRACKS];               // dZ to the highest mult vertex
  float trkDzError[MAXTRACKS];
  float trkDzError1[MAXTRACKS];
  float trkDzError2[MAXTRACKS];
  float trkDzOverDzError[MAXTRACKS*MAXVTX];
  float trkDxy[MAXTRACKS];
  float trkDxyBS[MAXTRACKS];
  float trkDxy1[MAXTRACKS];              // d0 to the highest pt vertex
  float trkDxy2[MAXTRACKS];              // d0 to the highest mult vertex
  float trkDxyError[MAXTRACKS];
  float trkDxyErrorBS[MAXTRACKS];
  float trkDxyError1[MAXTRACKS];
  float trkDxyError2[MAXTRACKS];
  float trkDxyOverDxyError[MAXTRACKS*MAXVTX];
  float trkVx[MAXTRACKS];
  float trkVy[MAXTRACKS];
  float trkVz[MAXTRACKS];
  bool  trkFake[MAXTRACKS];
  unsigned char trkAlgo[MAXTRACKS];
  unsigned char trkOriginalAlgo[MAXTRACKS];
  float trkMVA[MAXTRACKS];
  bool trkMVALoose[MAXTRACKS];
  bool trkMVATight[MAXTRACKS];
  float dedx[MAXTRACKS];
  int trkCharge[MAXTRACKS];
  unsigned char trkNVtx[MAXTRACKS];
  unsigned int trkVtxIndex[MAXTRACKS];
  bool trkAssocVtx[MAXTRACKS*MAXVTX];
  int nTrkTimesnVtx;

  float trkExpHit1Eta[MAXTRACKS];
  float trkExpHit2Eta[MAXTRACKS];
  float trkExpHit3Eta[MAXTRACKS];
  float trkStatus[MAXTRACKS];
  float trkPId[MAXTRACKS];
  float trkMPId[MAXTRACKS];
  float trkGMPId[MAXTRACKS];

  //matched PF Candidate Info
  int pfType[MAXTRACKS];
  float pfCandPt[MAXTRACKS];
  float pfEcal[MAXTRACKS];
  float pfHcal[MAXTRACKS];

  // -- sim tracks --
  int   nParticle;
  int pStatus[MAXTRACKS];
  int pPId[MAXTRACKS];
  float pEta[MAXTRACKS];
  float pPhi[MAXTRACKS];
  float pPt[MAXTRACKS];
  float pAcc[MAXTRACKS];
  float pAccPair[MAXTRACKS];
  int pCharge[MAXTRACKS];

  int pNRec[MAXTRACKS];
  int   pNHit[MAXTRACKS];
  // matched track info (if matched)
  float mtrkPt[MAXTRACKS];
  float mtrkPtError[MAXTRACKS];
  float mtrkEta[MAXTRACKS];
  float mtrkPhi[MAXTRACKS];
  int   mtrkNHit[MAXTRACKS];
  int   mtrkNlayer[MAXTRACKS];
  int   mtrkNlayer3D[MAXTRACKS];
  bool   mtrkQual[MAXQUAL][MAXTRACKS];
  float mtrkChi2[MAXTRACKS];
  int mtrkNdof[MAXTRACKS];
  float mtrkDz1[MAXTRACKS];
  float mtrkDzError1[MAXTRACKS];
  float mtrkDzOverDzError[MAXTRACKS*MAXVTX];
  float mtrkDxy1[MAXTRACKS];
  float mtrkDxyError1[MAXTRACKS];
  float mtrkDz2[MAXTRACKS];
  float mtrkDzError2[MAXTRACKS];
  float mtrkDxy2[MAXTRACKS];
  float mtrkDxyError2[MAXTRACKS];
  float mtrkDxyOverDxyError[MAXTRACKS*MAXVTX];
  int mtrkAlgo[MAXTRACKS];
  int mtrkOriginalAlgo[MAXTRACKS];
  float mtrkMVA[MAXTRACKS];
  int nParticleTimesnVtx;
  bool mtrkMVATight[MAXTRACKS];
  bool mtrkMVALoose[MAXTRACKS];

  // calo compatibility
  int mtrkPfType[MAXTRACKS];
  float mtrkPfCandPt[MAXTRACKS];
  float mtrkPfEcal[MAXTRACKS];
  float mtrkPfHcal[MAXTRACKS];

  int matchedGenID[MAXTRACKS][MAXMATCH];
};

class TrackAnalyzer : public edm::EDAnalyzer {

public:
  explicit TrackAnalyzer(const edm::ParameterSet&);
  ~TrackAnalyzer();

private:
  virtual void beginJob() ;
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void endJob() ;

  void fillVertices(const edm::Event& iEvent);
  void fillTracks(const edm::Event& iEvent, const edm::EventSetup& iSetup);
  void fillSimTracks(const edm::Event& iEvent, const edm::EventSetup& iSetup);
  void matchPFCandToTrack(const edm::Event& iEvent, const edm::EventSetup& iSetup, unsigned it, int & cand_type, float & cand_pt, float & mEcalSum, float & mHcalSum);

  int getLayerId(const PSimHit&);
  bool hitDeadPXF(const reco::Track& tr);
  //const TrackingParticle* doRecoToTpMatch(reco::RecoToSimCollection recSimColl, const reco::TrackRef &in);
  //vector<int> matchTpToGen(const edm::Event& iEvent, const TrackingParticle* tparticle);

  int associateSimhitToTrackingparticle(unsigned int trid );
  bool checkprimaryparticle(const TrackingParticle* tp);

  // ----------member data ---------------------------

  bool doTrack_;
  bool doTrackExtra_;
  bool doSimTrack_;
  bool doSimVertex_;
  bool fillSimTrack_;
  bool doPFMatching_;
  bool useQuality_;
  bool doDeDx_;
  bool doDebug_;
  bool doMVA_;
  // bool associateChi2_;
  bool doHighestPtVertex_;
  bool doTrackVtxWImpPar_;

  double trackPtMin_;
  double trackVtxMaxDistance_;
  std::vector<std::string> qualityStrings_;
  std::string qualityString_;

  double simTrackPtMin_;
  bool fiducialCut_;
  edm::InputTag trackSrcLabel_;
  edm::EDGetTokenT<edm::View<reco::Track> > trackSrcView_;
  edm::EDGetTokenT<vector<Track> > trackSrc_;
  edm::InputTag mvaSrcLabel_;
  edm::EDGetTokenT<edm::ValueMap<float> > mvaSrc_;
  edm::EDGetTokenT<reco::GenParticleCollection> particleSrc_;
  //edm::EDGetTokenT<> tpFakeSrc_;
  edm::EDGetTokenT<TrackingParticleCollection> tpEffSrc_;
  edm::EDGetTokenT<PFCandidateCollection> pfCandSrc_;
  edm::EDGetTokenT<DeDxDataValueMap> DeDxSrc_;
  edm::EDGetTokenT<reco::SimToRecoCollection> associatorMapSR_;
  edm::EDGetTokenT<reco::RecoToSimCollection> associatorMapRS_;

  vector<edm::EDGetTokenT<reco::VertexCollection> > vertexSrc_;
  edm::EDGetTokenT<TrackingVertexCollection> simVertexSrc_;

  const TrackerGeometry* geo_;
  edm::Service<TFileService> fs;
  edm::ESHandle < ParticleDataTable > pdt;
  edm::Handle<TrackingParticleCollection> trackingParticles;

  edm::EDGetTokenT<reco::BeamSpot> beamSpotProducer_;

  // Root object
  TTree* trackTree_;

  TrackEvent pev_;

  // Acceptance
  enum { BPix1=0, BPix2=1, BPix3=2,
	 FPix1_neg=3, FPix2_neg=4,
	 FPix1_pos=5, FPix2_pos=6,
	 nLayers=7};

};

//--------------------------------------------------------------------------------------------------
TrackAnalyzer::TrackAnalyzer(const edm::ParameterSet& iConfig)

{

  doTrack_             = iConfig.getUntrackedParameter<bool>  ("doTrack",true);
  doTrackExtra_             = iConfig.getUntrackedParameter<bool>  ("doTrackExtra",false);
  doSimTrack_             = iConfig.getUntrackedParameter<bool>  ("doSimTrack",false);
  fillSimTrack_             = iConfig.getUntrackedParameter<bool>  ("fillSimTrack",false);
  doSimVertex_             = iConfig.getUntrackedParameter<bool>  ("doSimVertex",false);
  doHighestPtVertex_             = iConfig.getUntrackedParameter<bool>  ("doHighestPtVertex",true);

  if(!doSimTrack_){
    fillSimTrack_ = 0;
    doSimVertex_ = 0;
  }
  
  // else{
    // associateChi2_             = iConfig.getParameter<bool>  ("associateChi2");
  // }

  doDeDx_             = iConfig.getUntrackedParameter<bool>  ("doDeDx",false);
  doDebug_             = iConfig.getUntrackedParameter<bool>  ("doDebug",false);
  doMVA_             = iConfig.getUntrackedParameter<bool>  ("doMVA",false);

  doPFMatching_             = iConfig.getUntrackedParameter<bool>  ("doPFMatching",false);
  doTrackVtxWImpPar_             = iConfig.getUntrackedParameter<bool>  ("doTrackVtxWImpPar",true);
  useQuality_ = iConfig.getUntrackedParameter<bool>("useQuality",false);

  trackPtMin_             = iConfig.getUntrackedParameter<double>  ("trackPtMin",0.40);
  trackVtxMaxDistance_             = iConfig.getUntrackedParameter<double>  ("trackVtxMaxDistance",3.0);
  qualityString_ = iConfig.getUntrackedParameter<std::string>("qualityString","highPurity");

  qualityStrings_ = iConfig.getUntrackedParameter<std::vector<std::string> >("qualityStrings",std::vector<std::string>(0));
  if(qualityStrings_.size() == 0) qualityStrings_.push_back(qualityString_);

  simTrackPtMin_             = iConfig.getUntrackedParameter<double>  ("simTrackPtMin",0.40);
  fiducialCut_ = (iConfig.getUntrackedParameter<bool>("fiducialCut",false));
  trackSrcLabel_ = iConfig.getParameter<edm::InputTag>("trackSrc");
  trackSrc_ = consumes<vector<Track> > (trackSrcLabel_);
  trackSrcView_ = consumes<edm::View<reco::Track> >(trackSrcLabel_);
  if(doMVA_){
    mvaSrcLabel_ = iConfig.getParameter<edm::InputTag>("mvaSrc");
    mvaSrc_ = consumes<edm::ValueMap<float> > (mvaSrcLabel_);
  }
  if(doSimTrack_){
    particleSrc_ = consumes<reco::GenParticleCollection>(iConfig.getParameter<edm::InputTag>("particleSrc"));
    //tpFakeSrc_ =  consumes<>(iConfig.getUntrackedParameter<edm::InputTag>("tpFakeSrc",edm::InputTag("mix","MergedTrackTruth")));
    tpEffSrc_ =  consumes<TrackingParticleCollection>(iConfig.getUntrackedParameter<edm::InputTag>("tpEffSrc",edm::InputTag("mix","MergedTrackTruth")));
    associatorMapSR_ = consumes<reco::SimToRecoCollection>(iConfig.getParameter<edm::InputTag>("associatorMap"));
    associatorMapRS_ = consumes<reco::RecoToSimCollection>(iConfig.getParameter<edm::InputTag>("associatorMap"));
  }

  std::vector<std::string> vertexSrcString_ = iConfig.getParameter<vector<string> >("vertexSrc");
  for(unsigned i = 0; i < vertexSrcString_.size(); i++)
  {
    vertexSrc_.push_back(consumes<reco::VertexCollection>(edm::InputTag(vertexSrcString_[i])));
  }
  if(doSimVertex_){
    simVertexSrc_ =  consumes<TrackingVertexCollection>(iConfig.getUntrackedParameter<edm::InputTag>("tpVtxSrc",edm::InputTag("mix","MergedTrackTruth")));
  }
  beamSpotProducer_  = consumes<reco::BeamSpot>(iConfig.getUntrackedParameter<edm::InputTag>("beamSpotSrc",edm::InputTag("offlineBeamSpot")));
  if(doPFMatching_){
    pfCandSrc_ = consumes<PFCandidateCollection>(iConfig.getParameter<edm::InputTag>("pfCandSrc"));
  }
  if(doDeDx_){
    DeDxSrc_ = consumes<DeDxDataValueMap> (iConfig.getUntrackedParameter<edm::InputTag>("DeDxMap"));
  }

}

//--------------------------------------------------------------------------------------------------
TrackAnalyzer::~TrackAnalyzer()
{
}

//--------------------------------------------------------------------------------------------------
void
TrackAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  // Get tracker geometry
  //  cout <<"StartFill"<<endl;

  edm::ESHandle<TrackerGeometry> tGeo;
  iSetup.get<TrackerDigiGeometryRecord>().get(tGeo);
  geo_ = tGeo.product();
  iSetup.getData(pdt);

  //  cout <<"Got data"<<endl;
  pev_.nEv = (int)iEvent.id().event();
  pev_.nRun = (int)iEvent.id().run();
  pev_.nLumi = (int)iEvent.luminosityBlock();
  pev_.nBX = (int)iEvent.bunchCrossing();
  pev_.N = 0;

  // pev_.nv = 0;
  pev_.nParticle = 0;
  pev_.nTrk = 0;

  //cout <<"Fill Vtx"<<endl;
  fillVertices(iEvent);

  //cout <<"Fill Tracks"<<endl;
  if (doTrack_) fillTracks(iEvent, iSetup);
  //cout <<"Tracks filled!"<<endl;
  if (doSimTrack_){
    fillSimTracks(iEvent, iSetup);
    pev_.nParticleTimesnVtx = pev_.nParticle*pev_.nVtx;
  }
  //cout <<"SimTracks filled!"<<endl;
  pev_.nTrkTimesnVtx=pev_.nTrk*pev_.nVtx;
  trackTree_->Fill();
  //cout <<"Tree filled!"<<endl;
  memset(&pev_,0,sizeof pev_);

}

//--------------------------------------------------------------------------------------------------
void
TrackAnalyzer::fillVertices(const edm::Event& iEvent){

  // Vertex 0 : pev_vz[0] MC information from TrackingVertexCollection
  // Vertex 1 - n : Reconstructed Vertex from various of algorithms


  if(doSimVertex_){
    Handle<TrackingVertexCollection> vertices;
    iEvent.getByToken(simVertexSrc_, vertices);
    pev_.nVtxSim=vertices->size();
    for (unsigned int i = 0 ; i< vertices->size(); ++i){
      pev_.zVtxSim[i] = (*vertices)[i].position().z();
      pev_.xVtxSim[i] = (*vertices)[i].position().z();
      pev_.yVtxSim[i] = (*vertices)[i].position().z();
    }
  }

  // Fill reconstructed vertices.
  for(unsigned int iv = 0; iv < vertexSrc_.size(); ++iv){
    const reco::VertexCollection * recoVertices;
    edm::Handle<reco::VertexCollection> vertexCollection;
    //cout <<vertexSrc_[iv]<<endl;
    iEvent.getByToken(vertexSrc_[iv],vertexCollection);
    recoVertices = vertexCollection.product();
    // unsigned int daughter = 0;
    int nVertex = 0;
    unsigned int greatestNtrkVtx = 0;
    unsigned int greatestPtVtx = 0;

    nVertex = recoVertices->size();
    pev_.nVtx = nVertex;
    for (int i = 0 ; i< nVertex; ++i){
      pev_.xVtx[i] = (*recoVertices)[i].position().x();
      pev_.yVtx[i] = (*recoVertices)[i].position().y();
      pev_.zVtx[i] = (*recoVertices)[i].position().z();
      pev_.xVtxErr[i] = (*recoVertices)[i].xError();
      pev_.yVtxErr[i] = (*recoVertices)[i].yError();
      pev_.zVtxErr[i] = (*recoVertices)[i].zError();
      pev_.normChi2Vtx[i] = (*recoVertices)[i].normalizedChi2();


      math::XYZPoint vtx_temp(pev_.xVtx[i],pev_.yVtx[i], pev_.zVtx[i]);


      float vtxSumPt=0.;
      int vtxMult=0;

      Handle<vector<Track> > etracks;
      iEvent.getByToken(trackSrc_, etracks);
      if(doTrackVtxWImpPar_){
        int trkCount = 0;
	for(unsigned it=0; it<etracks->size(); ++it){
	  if(i==0) pev_.trkNVtx[it]=0;
	  pev_.trkAssocVtx[it*pev_.nVtx+i]=false;
	  const reco::Track & etrk = (*etracks)[it];
	  if (etrk.pt()<trackPtMin_) continue;
	  if(fiducialCut_ && hitDeadPXF(etrk)) continue; // if track hits the dead region, igonore it;
	  float Dz=etrk.dz(vtx_temp);
	  float DzError=sqrt(etrk.dzError()*etrk.dzError()+pev_.zVtxErr[i]*pev_.zVtxErr[i]);
	  float Dxy=etrk.dxy(vtx_temp);
	  float DxyError=sqrt(etrk.dxyError()*etrk.dxyError()+pev_.xVtxErr[i]*pev_.yVtxErr[i]);
          pev_.trkDzOverDzError[trkCount*pev_.nVtx+i]=fabs(rndSF(Dz/DzError,4));
          pev_.trkDxyOverDxyError[trkCount*pev_.nVtx+i]=fabs(rndSF(Dxy/DxyError,4));
          trkCount++;
	  
          if(qualityStrings_.size()>0 && !etrk.quality(reco::TrackBase::qualityByName(qualityStrings_[0].data()))) continue;
	  if(fabs(Dz/DzError) < trackVtxMaxDistance_ && fabs(Dxy/DxyError)< trackVtxMaxDistance_ && etrk.ptError()/etrk.pt() < 0.3 && fabs(etrk.eta())<2.4){
	    vtxSumPt+=etrk.pt();
	    vtxMult++;
	    pev_.trkAssocVtx[trkCount*pev_.nVtx+i]=true;
	    pev_.trkNVtx[trkCount]++;
	  }
	}
      }
      else{
	for (reco::Vertex::trackRef_iterator it = (*recoVertices)[i].tracks_begin(); it != (*recoVertices)[i].tracks_end(); it++) {
	  vtxSumPt += (**it).pt();
	  Handle<vector<Track> > etracks;
	  iEvent.getByToken(trackSrc_, etracks);

	  for(unsigned itrack=0; itrack<etracks->size(); ++itrack){
	    reco::TrackRef trackRef=reco::TrackRef(etracks,itrack);
	    //cout<<" trackRef.key() "<<trackRef.key()<< " it->key() "<<it->key()<<endl;
	    if(trackRef.key()==it->key()){
	      pev_.trkVtxIndex[itrack] = i;  // note that index starts from 1
	      //cout<< " matching track "<<itrack<<endl;
	    }
	  }
	}
      }

      pev_.sumPtVtx[i] = vtxSumPt;

      if(doTrackVtxWImpPar_) pev_.nTrkVtx[i] = vtxMult;
      else pev_.nTrkVtx[i] = (*recoVertices)[i].tracksSize();

      if( vtxMult > pev_.nTrkVtx[greatestNtrkVtx]) greatestNtrkVtx = i;
      if( vtxSumPt > pev_.sumPtVtx[greatestPtVtx]) greatestPtVtx = i;
    }

    pev_.maxMultVtx = greatestNtrkVtx;
    pev_.maxPtVtx = greatestPtVtx;

    //loop over vertices again to get the significance wrt the leading vertex -Matt
    for (unsigned int i = 0 ; i< recoVertices->size(); ++i){
      if(i==greatestPtVtx) continue;
      GlobalVector direction = GlobalVector(pev_.xVtx[i]-pev_.xVtx[greatestPtVtx],pev_.xVtx[i]-pev_.xVtx[greatestPtVtx],pev_.xVtx[i]-pev_.xVtx[greatestPtVtx]);
      Measurement1D vtxDist2D = reco::SecondaryVertex::computeDist2d((*recoVertices)[greatestPtVtx], (*recoVertices)[i], direction, true);
      Measurement1D vtxDist3D = reco::SecondaryVertex::computeDist3d((*recoVertices)[greatestPtVtx], (*recoVertices)[i], direction, true);
      pev_.vtxDist2D[i]=vtxDist2D.value();
      pev_.vtxDist2DErr[i]=vtxDist2D.error();
      pev_.vtxDist2DSig[i]=vtxDist2D.significance();
      pev_.vtxDist3D[i]=vtxDist3D.value();
      pev_.vtxDist3DErr[i]=vtxDist3D.error();
      pev_.vtxDist3DSig[i]=vtxDist3D.significance();
    }
  }
}

//--------------------------------------------------------------------------------------------------
void
TrackAnalyzer::fillTracks(const edm::Event& iEvent, const edm::EventSetup& iSetup){
  Handle<vector<Track> > etracks;
  iEvent.getByToken(trackSrc_, etracks);
  reco::BeamSpot beamSpot;
  edm::Handle<reco::BeamSpot> recoBeamSpotHandle;
  iEvent.getByToken(beamSpotProducer_,recoBeamSpotHandle);
  beamSpot = *recoBeamSpotHandle;

  // do reco-to-sim association
  // Handle<TrackingParticleCollection>  TPCollectionHfake;
  Handle<edm::View<reco::Track> >  trackCollection;
  iEvent.getByToken(trackSrcView_, trackCollection);
  // ESHandle<TrackAssociatorBase> theAssociator;
  reco::RecoToSimCollection recSimColl;

  edm::Handle<reco::RecoToSimCollection > recotosimCollectionH;

  Handle<DeDxDataValueMap> DeDxMap;
  if(doDeDx_){
    iEvent.getByToken(DeDxSrc_, DeDxMap);
  }
  
  Handle<edm::ValueMap<float> > mvaoutput;
  if(doMVA_){
   iEvent.getByToken(mvaSrc_, mvaoutput);
  }
  if(doSimTrack_) {
   iEvent.getByToken(associatorMapRS_,recotosimCollectionH);
   recSimColl= *(recotosimCollectionH.product());
  }

  pev_.nTrk=0;
  pev_.N=0;
  for(unsigned it=0; it<etracks->size(); ++it){
    const reco::Track & etrk = (*etracks)[it];
    reco::TrackRef trackRef=reco::TrackRef(etracks,it);

    if (etrk.pt()<trackPtMin_) continue;
    if(fiducialCut_ && hitDeadPXF(etrk)) continue; // if track hits the dead region, igonore it;


    for(unsigned int iq = 0; iq < qualityStrings_.size(); ++iq){
      pev_.trkQual[iq][pev_.nTrk]=0;
      if(etrk.quality(reco::TrackBase::qualityByName(qualityStrings_[iq].data()))) pev_.trkQual[iq][pev_.nTrk]=1;
    }

    if(useQuality_ && etrk.quality(reco::TrackBase::qualityByName(qualityString_)) != 1) continue;

    if(doDeDx_){
      pev_.dedx[pev_.nTrk]=(*DeDxMap)[trackRef].dEdx();

    }
	
    if(doDebug_){
     int count1dhits=0;
     trackingRecHit_iterator edh = etrk.recHitsEnd();
     for (trackingRecHit_iterator ith = etrk.recHitsBegin(); ith != edh; ++ith) {
      // const TrackingRecHit * hit = ith->get();

      if ((*ith)->isValid()) {
       if (typeid(*ith) == typeid(SiStripRecHit1D)) ++count1dhits;
       }
     }
	 pev_.trkChi2hit1D[pev_.nTrk]=(etrk.chi2()+count1dhits)/double(etrk.ndof()+count1dhits);

	}
 
    pev_.trkEta[pev_.nTrk]=rndDP(etrk.eta(),3);
    pev_.trkPhi[pev_.nTrk]=rndDP(etrk.phi(),3);
    pev_.trkPt[pev_.nTrk]=rndSF(etrk.pt(),4);
    pev_.trkPtError[pev_.nTrk]=rndSF(etrk.ptError(),4);
    pev_.trkCharge[pev_.nTrk]=etrk.charge();
    pev_.trkNHit[pev_.nTrk]=etrk.numberOfValidHits();
    pev_.trkDxy[pev_.nTrk]=etrk.dxy();
    pev_.trkDxyError[pev_.nTrk]=etrk.dxyError();
    pev_.trkDz[pev_.nTrk]=etrk.dz();
    pev_.trkDzError[pev_.nTrk]=etrk.dzError();
    pev_.trkChi2[pev_.nTrk]=rndSF(etrk.chi2(),4);
    pev_.trkNdof[pev_.nTrk]=etrk.ndof();
    pev_.trkVx[pev_.nTrk]=etrk.vx();
    pev_.trkVy[pev_.nTrk]=etrk.vy();
    pev_.trkVz[pev_.nTrk]=etrk.vz();

    math::XYZPoint v1(pev_.xVtx[pev_.maxPtVtx],pev_.yVtx[pev_.maxPtVtx], pev_.zVtx[pev_.maxPtVtx]);
    pev_.trkDz1[pev_.nTrk]=rndSF(etrk.dz(v1),4);
    pev_.trkDzError1[pev_.nTrk]=rndSF(sqrt(etrk.dzError()*etrk.dzError()+pev_.zVtxErr[pev_.maxPtVtx]*pev_.zVtxErr[pev_.maxPtVtx]),4);
    pev_.trkDxy1[pev_.nTrk]=rndSF(etrk.dxy(v1),4);
    pev_.trkDxyError1[pev_.nTrk]=rndSF(sqrt(etrk.dxyError()*etrk.dxyError()+pev_.xVtxErr[pev_.maxPtVtx]*pev_.yVtxErr[pev_.maxPtVtx]),4);

    math::XYZPoint v2(pev_.xVtx[pev_.maxMultVtx],pev_.yVtx[pev_.maxMultVtx], pev_.zVtx[pev_.maxMultVtx]);
    pev_.trkDz2[pev_.nTrk]=rndSF(etrk.dz(v2),4);
    pev_.trkDzError2[pev_.nTrk]=rndSF(sqrt(etrk.dzError()*etrk.dzError()+pev_.zVtxErr[pev_.maxMultVtx]*pev_.zVtxErr[pev_.maxMultVtx]),4);
    pev_.trkDxy2[pev_.nTrk]=rndSF(etrk.dxy(v2),4);
    pev_.trkDxyError2[pev_.nTrk]=rndSF(sqrt(etrk.dxyError()*etrk.dxyError()+pev_.xVtxErr[pev_.maxMultVtx]*pev_.yVtxErr[pev_.maxMultVtx]),4);

    pev_.trkDxyBS[pev_.nTrk]=etrk.dxy(beamSpot.position());
    pev_.trkDxyErrorBS[pev_.nTrk]=sqrt(etrk.dxyError()*etrk.dxyError()+beamSpot.BeamWidthX()*beamSpot.BeamWidthY());

    pev_.trkNlayer[pev_.nTrk] = etrk.hitPattern().trackerLayersWithMeasurement();
    pev_.trkNlayer3D[pev_.nTrk] = etrk.hitPattern().pixelLayersWithMeasurement() + etrk.hitPattern().numberOfValidStripLayersWithMonoAndStereo();

    pev_.trkAlgo[pev_.nTrk] = etrk.algo();
    pev_.trkOriginalAlgo[pev_.nTrk] = etrk.originalAlgo();

    if(doMVA_){
     if(etrk.algo() == 11){ //sets jet-core iteration tracks to MVA of +/-1 based on their highPurity bit (even though no MVA is used) 
       if(etrk.quality(reco::TrackBase::qualityByName(qualityStrings_[0].data()))) pev_.trkMVA[pev_.nTrk] = 1;
       else pev_.trkMVA[pev_.nTrk] = -1;
     }
     else pev_.trkMVA[pev_.nTrk] = rndDP((*mvaoutput)[trackRef],3);//non algo=11 behavior
	 if(mvaSrcLabel_.label() == "generalTracks")
	 {
	   pev_.trkMVALoose[pev_.nTrk] = (!((pev_.trkAlgo[pev_.nTrk] == 4 && pev_.trkMVA[pev_.nTrk] < -0.7) || (pev_.trkAlgo[pev_.nTrk] == 5 && pev_.trkMVA[pev_.nTrk] < -0.1) || (pev_.trkAlgo[pev_.nTrk] == 6 && pev_.trkMVA[pev_.nTrk] < 0.3) || (pev_.trkAlgo[pev_.nTrk] == 7 && pev_.trkMVA[pev_.nTrk] < 0.4) || (pev_.trkAlgo[pev_.nTrk] == 8 && pev_.trkMVA[pev_.nTrk] < -0.2) || (pev_.trkAlgo[pev_.nTrk] == 9 && pev_.trkMVA[pev_.nTrk] < 0.0) ||(pev_.trkAlgo[pev_.nTrk] == 10 && pev_.trkMVA[pev_.nTrk] < -0.3)) || pev_.trkMVA[pev_.nTrk] == -99) &&  etrk.quality(reco::TrackBase::qualityByName("highPurity"));
	 
	   pev_.trkMVATight[pev_.nTrk] = (!((pev_.trkAlgo[pev_.nTrk] == 4 && pev_.trkMVA[pev_.nTrk] < -0.7) || (pev_.trkAlgo[pev_.nTrk] == 5 && pev_.trkMVA[pev_.nTrk] < -0.1) || (pev_.trkAlgo[pev_.nTrk] == 6 && pev_.trkMVA[pev_.nTrk] < 0.3) || (pev_.trkAlgo[pev_.nTrk] == 7 && pev_.trkMVA[pev_.nTrk] < 0.5) || (pev_.trkAlgo[pev_.nTrk] == 8 && pev_.trkMVA[pev_.nTrk] < 0.5) || (pev_.trkAlgo[pev_.nTrk] == 9 && pev_.trkMVA[pev_.nTrk] < 0.4) ||(pev_.trkAlgo[pev_.nTrk] == 10 && pev_.trkMVA[pev_.nTrk] < 0)) || pev_.trkMVA[pev_.nTrk] == -99) &&  etrk.quality(reco::TrackBase::qualityByName("highPurity"));
     }
	 if(mvaSrcLabel_.label() == "hiGeneralTracks")
	 {
	   pev_.trkMVATight[pev_.nTrk] = (!((pev_.trkAlgo[pev_.nTrk] == 4 && pev_.trkMVA[pev_.nTrk] < -0.77) || (pev_.trkAlgo[pev_.nTrk] == 5 && pev_.trkMVA[pev_.nTrk] < 0.35) || (pev_.trkAlgo[pev_.nTrk] == 6 && pev_.trkMVA[pev_.nTrk] < 0.77) || (pev_.trkAlgo[pev_.nTrk] == 7 && pev_.trkMVA[pev_.nTrk] < 0.35)) || pev_.trkMVA[pev_.nTrk] == -99) &&  etrk.quality(reco::TrackBase::qualityByName("highPurity"));
	 }
    }
    // multiplicity variable
    if (pev_.trkQual[0][pev_.nTrk]&&
        (fabs(pev_.trkDz1[pev_.nTrk]/pev_.trkDzError1[pev_.nTrk]) < trackVtxMaxDistance_)&&
        (fabs(pev_.trkDxy1[pev_.nTrk]/pev_.trkDxyError1[pev_.nTrk]) < trackVtxMaxDistance_)&&
        (pev_.trkPtError[pev_.nTrk]/pev_.trkPt[pev_.nTrk]<0.3)&&
        (fabs(pev_.trkEta[pev_.nTrk]) < 2.4)&&
        (pev_.trkPt[pev_.nTrk] > trackPtMin_)
      ) pev_.N++;


    if (doSimTrack_) {
      pev_.trkFake[pev_.nTrk]=0;
      pev_.trkStatus[pev_.nTrk]=-999;
      pev_.trkPId[pev_.nTrk]=-999;
      pev_.trkMPId[pev_.nTrk]=-999;
      pev_.trkGMPId[pev_.nTrk]=-999;

      //match tracking particle to the reco particle
      const TrackingParticle* tparticle = doRecoToTpMatch(recSimColl, trackRef);//matchedSim->val[0].first.get();
      if(!tparticle) pev_.trkFake[pev_.nTrk]=1;
      else{
        pev_.trkStatus[pev_.nTrk]=tparticle->status();
	pev_.trkPId[pev_.nTrk]=tparticle->pdgId();
	if (tparticle->parentVertex().isNonnull() && !tparticle->parentVertex()->sourceTracks().empty()) {
	  pev_.trkMPId[pev_.nTrk]=tparticle->parentVertex()->sourceTracks()[0]->pdgId();
	} else {
	  pev_.trkMPId[pev_.nTrk]=-999;
	}

	// if (!tparticle->genParticle().empty()) {
	  // const HepMC::GenParticle * genMom = getGpMother(tparticle->genParticle()[0].get());
	  // if (genMom) {
	    // pev_.trkGMPId[pev_.nTrk] = genMom->pdg_id();
	  // }
        // }

        //now match the tracking particle to the gen-level particle
        vector<int> tempBarcode = matchTpToGen(iEvent, tparticle, particleSrc_);
        for(unsigned int ibarcode=0; ibarcode<tempBarcode.size(); ibarcode++){
            pev_.matchedGenID[pev_.nTrk][ibarcode] = tempBarcode.at(ibarcode);
        }
      }
    }

    if (doTrackExtra_) {
      // Very rough estimation of the expected position of the Pixel Hit
      double r = 4.4; // averaged first layer rho
      double x = r*cos(etrk.phi())+etrk.vx();
      double y = r*sin(etrk.eta())+etrk.vy();
      double z = r/tan(atan(exp(-etrk.eta()))*2)+etrk.vz();
      ROOT::Math::XYZVector tmpVector(x-pev_.xVtx[1],y-pev_.yVtx[1],z-pev_.zVtx[1]);
      double eta1 = tmpVector.eta();
      //double phi1 = etrk.phi();

      double r2 = 7.29; // averaged 2nd layer rho
      x = r2*cos(etrk.phi())+etrk.vx();
      y = r2*sin(etrk.eta())+etrk.vy();
      z = r2/tan(atan(exp(-etrk.eta()))*2)+etrk.vz();
      ROOT::Math::XYZVector tmpVector2(x-pev_.xVtx[1],y-pev_.yVtx[1],z-pev_.zVtx[1]);
      double eta2 = tmpVector2.eta();


      double r3 = 10.16; // averaged 3rd layer rho
      x = r3*cos(etrk.phi())+etrk.vx();
      y = r3*sin(etrk.eta())+etrk.vy();
      z = r3/tan(atan(exp(-etrk.eta()))*2)+etrk.vz();
      ROOT::Math::XYZVector tmpVector3(x-pev_.xVtx[1],y-pev_.yVtx[1],z-pev_.zVtx[1]);
      double eta3 = tmpVector3.eta();


      pev_.trkExpHit1Eta[pev_.nTrk]=eta1;
      pev_.trkExpHit2Eta[pev_.nTrk]=eta2;
      pev_.trkExpHit3Eta[pev_.nTrk]=eta3;
    }
    //pev_.trkNhit[pev_.nTrk]=tr.numberOfValidHits();
    if(doPFMatching_) matchPFCandToTrack(iEvent, iSetup, it,
					 // output to the following vars
					 pev_.pfType[pev_.nTrk],
					 pev_.pfCandPt[pev_.nTrk],
					 pev_.pfEcal[pev_.nTrk],
					 pev_.pfHcal[pev_.nTrk]);

    pev_.nTrk++;
  }

}

//-------------- Fill mother particle info via the production vertex -----------------
const HepMC::GenParticle * getGpMother2(const HepMC::GenParticle *gp) {
        if ( gp != 0 && gp->production_vertex()) {
                    //         cout << "get production vertex" << endl;
                    const HepMC::GenVertex *vtx = gp->production_vertex();
                            if (vtx != 0 && vtx->particles_in_size() > 0) {
                                            return *vtx->particles_in_const_begin();
                                                    }
                                }
            return 0;
}

//--------------------------------------------------------------------------------------------------
void
TrackAnalyzer::fillSimTracks(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  // edm::ESHandle<TrackAssociatorBase> theAssociator;
  edm::Handle<reco::SimToRecoCollection > simtorecoCollectionH;
  edm::Handle<TrackingParticleCollection>  TPCollectionHeff;

  iEvent.getByToken(tpEffSrc_,TPCollectionHeff);
  reco::SimToRecoCollection simRecColl;
  edm::Handle<vector<reco::Track> > etracks;
  Handle<edm::ValueMap<float> > mvaoutput;
  if(doMVA_){
   iEvent.getByToken(trackSrc_,etracks);
   iEvent.getByToken(mvaSrc_, mvaoutput);
  }
  iEvent.getByToken(associatorMapSR_,simtorecoCollectionH);
  simRecColl= *(simtorecoCollectionH.product());


  // Loop through sim tracks
  pev_.nParticle = 0;
  for(TrackingParticleCollection::size_type i=0; i<TPCollectionHeff->size(); i++) {
    TrackingParticleRef tpr(TPCollectionHeff, i);
    TrackingParticle* tp=const_cast<TrackingParticle*>(tpr.get());

    if (tp->pt() < simTrackPtMin_) continue;
    if (tp->status() < 0 || tp->charge()==0) continue; //only charged primaries

    // Fill sim track info
    pev_.pStatus[pev_.nParticle] = tp->status();
    pev_.pPId[pev_.nParticle] = tp->pdgId();
    pev_.pEta[pev_.nParticle] = tp->eta();
    pev_.pPhi[pev_.nParticle] = tp->phi();
    pev_.pPt[pev_.nParticle] = tp->pt();

    // Look up association map
    std::vector<std::pair<edm::RefToBase<reco::Track>, double> > rt;
    const reco::Track* mtrk=0;
    size_t nrec=0;
    if(simRecColl.find(tpr) != simRecColl.end()){
      math::XYZPoint v1(pev_.xVtx[pev_.maxPtVtx],pev_.yVtx[pev_.maxPtVtx], pev_.zVtx[pev_.maxPtVtx]);
      math::XYZPoint v2(pev_.xVtx[pev_.maxMultVtx],pev_.yVtx[pev_.maxMultVtx], pev_.zVtx[pev_.maxMultVtx]);
      rt = (std::vector<std::pair<edm::RefToBase<reco::Track>, double> >) simRecColl[tpr];
      std::vector<std::pair<edm::RefToBase<reco::Track>, double> >::const_iterator rtit;
      for (rtit = rt.begin(); rtit != rt.end(); ++rtit)
      {
	const reco::Track* tmtr = rtit->first.get();
	if (!(tmtr->quality(reco::TrackBase::qualityByName(qualityString_)))) continue;
	if(doHighestPtVertex_){
	  if ((fabs(tmtr->dz(v1)/sqrt(tmtr->dzError()*tmtr->dzError()+pev_.zVtxErr[pev_.maxPtVtx]*pev_.zVtxErr[pev_.maxPtVtx])) < trackVtxMaxDistance_)&&
	      (fabs( tmtr->dxy(v1)/sqrt(tmtr->dxyError()*tmtr->dxyError()+pev_.xVtxErr[pev_.maxPtVtx]*pev_.yVtxErr[pev_.maxPtVtx])) < trackVtxMaxDistance_)&&
	      (tmtr->ptError()/tmtr->pt()<0.3)&&
	      (fabs(tmtr->eta()) < 2.4) &&
	      (tmtr->pt() > trackPtMin_)) nrec++;
	}
	else{
	  if ((fabs(tmtr->dz(v2)/sqrt(tmtr->dzError()*tmtr->dzError()+pev_.zVtxErr[pev_.maxMultVtx]*pev_.zVtxErr[pev_.maxMultVtx])) < trackVtxMaxDistance_)&&
	      (fabs( tmtr->dxy(v2)/sqrt(tmtr->dxyError()*tmtr->dxyError()+pev_.xVtxErr[pev_.maxMultVtx]*pev_.yVtxErr[pev_.maxMultVtx])) < trackVtxMaxDistance_)&&
	      (tmtr->ptError()/tmtr->pt()<0.3)&&
	      (fabs(tmtr->eta()) < 2.4)&&
	      (tmtr->pt() > trackPtMin_)) nrec++;
	}
      }

      // remove the association if the track hits the bed region in FPIX
      // nrec>0 since we don't need it for nrec=0 case
      mtrk = rt.begin()->first.get();
      if(fiducialCut_ && nrec>0 && hitDeadPXF(*mtrk)) nrec=0;

      // Fill matched rec track info
      pev_.pNRec[pev_.nParticle] = nrec;

      pev_.mtrkPt[pev_.nParticle] = mtrk->pt();
      pev_.mtrkPtError[pev_.nParticle] = mtrk->ptError();
      pev_.mtrkNHit[pev_.nParticle] = mtrk->numberOfValidHits();
      pev_.mtrkNlayer[pev_.nParticle] = mtrk->hitPattern().trackerLayersWithMeasurement();
      pev_.mtrkNlayer3D[pev_.nParticle] = mtrk->hitPattern().pixelLayersWithMeasurement() + mtrk->hitPattern().numberOfValidStripLayersWithMonoAndStereo();
      // if (mtrk->quality(reco::TrackBase::qualityByName(qualityString_))) pev_.mtrkQual[pev_.nParticle] = 1;
      
      for(unsigned int iq = 0; iq < qualityStrings_.size(); ++iq){
        pev_.mtrkQual[iq][pev_.nParticle]=0;
        if(mtrk->quality(reco::TrackBase::qualityByName(qualityStrings_[iq].data()))) pev_.mtrkQual[iq][pev_.nParticle]=1;
      }
      
      pev_.mtrkChi2[pev_.nParticle]=mtrk->chi2();
      pev_.mtrkNdof[pev_.nParticle]=mtrk->ndof();
      pev_.mtrkDz1[pev_.nParticle] = mtrk->dz(v1);
      pev_.mtrkDzError1[pev_.nParticle] = sqrt(mtrk->dzError()*mtrk->dzError()+pev_.zVtxErr[pev_.maxPtVtx]*pev_.zVtxErr[pev_.maxPtVtx]);
      pev_.mtrkDxy1[pev_.nParticle] = mtrk->dxy(v1);
      pev_.mtrkDxyError1[pev_.nParticle] = sqrt(mtrk->dxyError()*mtrk->dxyError()+pev_.xVtxErr[pev_.maxPtVtx]*pev_.yVtxErr[pev_.maxPtVtx]);
      pev_.mtrkDz2[pev_.nParticle] = mtrk->dz(v2);
      pev_.mtrkDzError2[pev_.nParticle] = sqrt(mtrk->dzError()*mtrk->dzError()+pev_.zVtxErr[pev_.maxMultVtx]*pev_.zVtxErr[pev_.maxMultVtx]);
      pev_.mtrkDxy2[pev_.nParticle] = mtrk->dxy(v2);
      pev_.mtrkDxyError2[pev_.nParticle] = sqrt(mtrk->dxyError()*mtrk->dxyError()+pev_.xVtxErr[pev_.maxMultVtx]*pev_.yVtxErr[pev_.maxMultVtx]);
      pev_.mtrkAlgo[pev_.nParticle] = mtrk->algo();
      pev_.mtrkOriginalAlgo[pev_.nParticle] = mtrk->originalAlgo();
      if(doTrackVtxWImpPar_){
        for(int vtxNum = 0; vtxNum<pev_.nVtx; vtxNum++){
          math::XYZPoint pt(pev_.xVtx[vtxNum],pev_.yVtx[vtxNum], pev_.zVtx[vtxNum]);
          pev_.mtrkDzOverDzError[pev_.nParticle*pev_.nVtx+vtxNum] = fabs(rndSF(mtrk->dz(pt)/(sqrt(mtrk->dzError()*mtrk->dzError()+pev_.zVtxErr[vtxNum]*pev_.zVtxErr[vtxNum])),4)); 
          pev_.mtrkDxyOverDxyError[pev_.nParticle*pev_.nVtx+vtxNum] = fabs(rndSF(mtrk->dxy(pt)/(sqrt(mtrk->dxyError()*mtrk->dxyError()+pev_.xVtxErr[vtxNum]*pev_.yVtxErr[vtxNum])),4));
        } 
      }
      if(doMVA_){
        pev_.mtrkMVA[pev_.nParticle] = -99;
	     if (pev_.mtrkPt[pev_.nParticle]>0) {
        unsigned ind = mtrk - &((*etracks)[0]);                                                                   
        reco::TrackRef trackRef=reco::TrackRef(etracks,ind);
                                                                                    
        if(mtrk->algo() == 11){ //sets jet-core iteration tracks to MVA of +/-1 based on their highPurity bit (even though no MVA is used) 
          if(mtrk->quality(reco::TrackBase::qualityByName(qualityStrings_[0].data()))) pev_.mtrkMVA[pev_.nParticle] = 1;
          else pev_.mtrkMVA[pev_.nParticle] = -1;
        }
        else pev_.mtrkMVA[pev_.nParticle] = rndDP((*mvaoutput)[trackRef],3);//non algo=11 behavior

	    if(mvaSrcLabel_.label() == "generalTracks")
	    {
	     pev_.mtrkMVALoose[pev_.nParticle] = (!((pev_.mtrkAlgo[pev_.nParticle] == 4 && pev_.mtrkMVA[pev_.nParticle] < -0.7) || (pev_.mtrkAlgo[pev_.nParticle] == 5 && pev_.mtrkMVA[pev_.nParticle] < -0.1) || (pev_.mtrkAlgo[pev_.nParticle] == 6 && pev_.mtrkMVA[pev_.nParticle] < 0.3) || (pev_.mtrkAlgo[pev_.nParticle] == 7 && pev_.mtrkMVA[pev_.nParticle] < 0.4) || (pev_.mtrkAlgo[pev_.nParticle] == 8 && pev_.mtrkMVA[pev_.nParticle] < -0.2) || (pev_.mtrkAlgo[pev_.nParticle] == 9 && pev_.mtrkMVA[pev_.nParticle] < 0.0) ||(pev_.mtrkAlgo[pev_.nParticle] == 10 && pev_.mtrkMVA[pev_.nParticle] < -0.3)) || pev_.mtrkMVA[pev_.nParticle] == -99) &&  mtrk->quality(reco::TrackBase::qualityByName("highPurity"));
	 
	     pev_.mtrkMVATight[pev_.nParticle] = (!((pev_.mtrkAlgo[pev_.nParticle] == 4 && pev_.mtrkMVA[pev_.nParticle] < -0.7) || (pev_.mtrkAlgo[pev_.nParticle] == 5 && pev_.mtrkMVA[pev_.nParticle] < -0.1) || (pev_.mtrkAlgo[pev_.nParticle] == 6 && pev_.mtrkMVA[pev_.nParticle] < 0.3) || (pev_.mtrkAlgo[pev_.nParticle] == 7 && pev_.mtrkMVA[pev_.nParticle] < 0.5) || (pev_.mtrkAlgo[pev_.nParticle] == 8 && pev_.mtrkMVA[pev_.nParticle] < 0.5) || (pev_.mtrkAlgo[pev_.nParticle] == 9 && pev_.mtrkMVA[pev_.nParticle] < 0.4) ||(pev_.mtrkAlgo[pev_.nParticle] == 10 && pev_.mtrkMVA[pev_.nParticle] < 0)) || pev_.mtrkMVA[pev_.nParticle] == -99) &&  mtrk->quality(reco::TrackBase::qualityByName("highPurity"));
        }
	    if(mvaSrcLabel_.label() == "hiGeneralTracks")
	    {
	     pev_.mtrkMVATight[pev_.nParticle] = (!((pev_.mtrkAlgo[pev_.nParticle] == 4 && pev_.mtrkMVA[pev_.nParticle] < -0.77) || (pev_.mtrkAlgo[pev_.nParticle] == 5 && pev_.mtrkMVA[pev_.nParticle] < 0.35) || (pev_.mtrkAlgo[pev_.nParticle] == 6 && pev_.mtrkMVA[pev_.nParticle] < 0.77) || (pev_.mtrkAlgo[pev_.nParticle] == 7 && pev_.mtrkMVA[pev_.nParticle] < 0.35)) || pev_.mtrkMVA[pev_.nParticle] == -99) &&  mtrk->quality(reco::TrackBase::qualityByName("highPurity"));
	    }
	   }
      }
      // calo matching info for the matched track
      if(doPFMatching_) {
	size_t mtrkkey = rt.begin()->first.key();
	matchPFCandToTrack(iEvent, iSetup, mtrkkey,
			   // output to the following vars
			   pev_.mtrkPfType[pev_.nParticle],
			   pev_.mtrkPfCandPt[pev_.nParticle],
			   pev_.mtrkPfEcal[pev_.nParticle],
			   pev_.mtrkPfHcal[pev_.nParticle]);
      }
    }
    // remove the association if the track hits the bed region in FPIX
    // nrec>0 since we don't need it for nrec=0 case
    if(fiducialCut_ && nrec>0 && hitDeadPXF(*mtrk)) nrec=0;
    //cout << "simtrk: " << tp->pdgId() << " pt: " << tp->pt() << " nrec: " << nrec << endl;

    // Fill matched rec track info

    ++pev_.nParticle;
  }

}


//--------------------------------------------------------------------------------------------------
void
TrackAnalyzer::matchPFCandToTrack(const edm::Event& iEvent, const edm::EventSetup& iSetup, unsigned it, int & cand_type, float & cand_pt, float & mEcalSum, float & mHcalSum)
{

  // get PF candidates
  Handle<PFCandidateCollection> pfCandidates;
  bool isPFThere = iEvent.getByToken(pfCandSrc_, pfCandidates);

  if (!isPFThere){
    //cout<<" NO PF Candidates Found"<<endl;
    return;  // if no PFCand in an event, skip it
  }

  // double sum_ecal=0.0, sum_hcal=0.0;
  double ecalEnergy=0.0, hcalEnergy=0.0;


  // loop over pfCandidates to find track

  // int cand_index = -999;
  cand_pt = -999.0;
  cand_type =-1;

  for( unsigned ic=0; ic<pfCandidates->size(); ic++ ) {

    const reco::PFCandidate& cand = (*pfCandidates)[ic];

    int type = cand.particleId();

    // only charged hadrons and leptons can be asscociated with a track
    if(!(type == PFCandidate::h ||     //type1
	 type == PFCandidate::e ||     //type2
	 type == PFCandidate::mu      //type3
	 )
      ) continue;


    reco::TrackRef trackRef = cand.trackRef();

    if(it==trackRef.key()) {
      // cand_index = ic;
      cand_type = type;
      cand_pt = cand.pt();
      ecalEnergy = cand.ecalEnergy();
      hcalEnergy = cand.hcalEnergy();
      break;

    }
  }

  /*
  if(cand_index>=0){

    const reco::PFCandidate& cand = (*pfCandidates)[cand_index];

    for(unsigned ib=0; ib<cand.elementsInBlocks().size(); ib++) {

      PFBlockRef blockRef = cand.elementsInBlocks()[ib].first;


      unsigned indexInBlock = cand.elementsInBlocks()[ib].second;
      const edm::OwnVector<  reco::PFBlockElement>&  elements = (*blockRef).elements();

      //This tells you what type of element it is:
      //cout<<" block type"<<elements[indexInBlock].type()<<endl;

      switch (elements[indexInBlock].type()) {

      case PFBlockElement::ECAL: {
	reco::PFClusterRef clusterRef = elements[indexInBlock].clusterRef();
	double eet = clusterRef->energy()/cosh(clusterRef->eta());
	sum_ecal+=eet;
	break;
      }

      case PFBlockElement::HCAL: {
	reco::PFClusterRef clusterRef = elements[indexInBlock].clusterRef();
	double eet = clusterRef->energy()/cosh(clusterRef->eta());
	sum_hcal+=eet;
	break;
      }
      case PFBlockElement::TRACK: {
	//Do nothing since track are not normally linked to other tracks
	break;
      }
      default:
	break;
      }

    } // end of elementsInBlocks()
  }  // end of if(cand_index >= 0)
*/

  cand_type=cand_type;
  cand_pt=cand_pt;
  mEcalSum=ecalEnergy;
  mHcalSum=hcalEnergy;

  return;

}



// ------------
int
TrackAnalyzer::getLayerId(const PSimHit & simHit)
{
  unsigned int id = simHit.detUnitId();

  if(geo_->idToDetUnit(id)->subDetector() ==
     GeomDetEnumerators::PixelBarrel)
  {
    PXBDetId pid(id);
    return pid.layer() - 1; // 0, 1, 2
  }

  if(geo_->idToDetUnit(id)->subDetector() ==
     GeomDetEnumerators::PixelEndcap)
  {
    PXFDetId pid(id);
    return BPix3 + ((pid.side()-1) << 1) + pid.disk(); // 3 -
  }

  // strip
  return -1;
}

// ---------------
bool
TrackAnalyzer::hitDeadPXF(const reco::Track& tr){

  //-----------------------------------------------
  // For a given track, check whether this contains
  // hits on the dead region in the forward pixel
  //-----------------------------------------------

  bool hitDeadRegion = false;

  for(trackingRecHit_iterator recHit = tr.recHitsBegin();recHit!= tr.recHitsEnd(); recHit++){

    if((*recHit)->isValid()){

      DetId detId = (*recHit)->geographicalId();
      if(!geo_->idToDet(detId)) continue;

      Int_t diskLayerNum=0, bladeLayerNum=0, hcylLayerNum=0;

      unsigned int subdetId = static_cast<unsigned int>(detId.subdetId());

      if (subdetId == PixelSubdetector::PixelEndcap){

	PixelEndcapName pxfname(detId.rawId());
	diskLayerNum = pxfname.diskName();
	bladeLayerNum = pxfname.bladeName();
	hcylLayerNum = pxfname.halfCylinder();

	// hard-coded now based on /UserCode/Appeltel/PixelFiducialRemover/pixelfiducialremover_cfg.py
	if((bladeLayerNum==4 || bladeLayerNum==5 || bladeLayerNum==6) &&
	   (diskLayerNum==2) && (hcylLayerNum==4)) hitDeadRegion = true;
      }

    }// end of isValid
  }

  return hitDeadRegion;
}

// ------------ method called once each job just before starting event loop  ------------
void
TrackAnalyzer::beginJob()
{

  trackTree_ = fs->make<TTree>("trackTree","v1");

  // event
  trackTree_->Branch("nEv",&pev_.nEv,"nEv/I");
  trackTree_->Branch("nLumi",&pev_.nLumi,"nLumi/I");
  trackTree_->Branch("nBX",&pev_.nBX,"nBX/I");
  trackTree_->Branch("nRun",&pev_.nRun,"nRun/I");
  trackTree_->Branch("N",&pev_.N,"N/I");

  // vertex

  trackTree_->Branch("nVtx",&pev_.nVtx,"nVtx/I");
  trackTree_->Branch("nTrk",&pev_.nTrk,"nTrk/I");

  trackTree_->Branch("maxPtVtx",&pev_.maxPtVtx,"maxPtVtx/I");
  trackTree_->Branch("maxMultVtx",&pev_.maxMultVtx,"maxMultVtx/I");
  //  trackTree_->Branch("maxVtxHard",&pev_.maxVtxHard,"maxVtxHard/I");

  trackTree_->Branch("nTrkVtx",pev_.nTrkVtx,"nTrkVtx[nVtx]/I");
  trackTree_->Branch("normChi2Vtx",pev_.normChi2Vtx,"normChi2Vtx[nVtx]/F");
  trackTree_->Branch("sumPtVtx",pev_.sumPtVtx,"sumPtVtx[nVtx]/F");
  //  trackTree_->Branch("nTrkVtxHard",pev_.nTrkVtxHard,"nTrkVtxHard[nVtx]/I");

  trackTree_->Branch("xVtx",pev_.xVtx,"xVtx[nVtx]/F");
  trackTree_->Branch("yVtx",pev_.yVtx,"yVtx[nVtx]/F");
  trackTree_->Branch("zVtx",pev_.zVtx,"zVtx[nVtx]/F");
  trackTree_->Branch("xVtxErr",pev_.xVtxErr,"xVtxErr[nVtx]/F");
  trackTree_->Branch("yVtxErr",pev_.yVtxErr,"yVtxErr[nVtx]/F");
  trackTree_->Branch("zVtxErr",pev_.zVtxErr,"zVtxErr[nVtx]/F");

  trackTree_->Branch("vtxDist2D",pev_.vtxDist2D,"vtxDist2D[nVtx]/F");
  trackTree_->Branch("vtxDist2DErr",pev_.vtxDist2DErr,"vtxDist2DErr[nVtx]/F");
  trackTree_->Branch("vtxDist2DSig",pev_.vtxDist2DSig,"vtxDist2DSig[nVtx]/F");
  trackTree_->Branch("vtxDist3D",pev_.vtxDist3D,"vtxDist3D[nVtx]/F");
  trackTree_->Branch("vtxDist3DErr",pev_.vtxDist3DErr,"vtxDist3DErr[nVtx]/F");
  trackTree_->Branch("vtxDist3DSig",pev_.vtxDist3DSig,"vtxDist3DSig[nVtx]/F");

  trackTree_->Branch("nVtxSim",&pev_.nVtxSim,"nVtxSim/I");
  trackTree_->Branch("xVtxSim",pev_.xVtxSim,"xVtx[nVtxSim]/F");
  trackTree_->Branch("yVtxSim",pev_.yVtxSim,"yVtx[nVtxSim]/F");
  trackTree_->Branch("zVtxSim",pev_.zVtxSim,"zVtx[nVtxSim]/F");

  // Tracks
  trackTree_->Branch("trkPt",&pev_.trkPt,"trkPt[nTrk]/F");
  trackTree_->Branch("trkPtError",&pev_.trkPtError,"trkPtError[nTrk]/F");
  trackTree_->Branch("trkNHit",&pev_.trkNHit,"trkNHit[nTrk]/b");
  trackTree_->Branch("trkNlayer",&pev_.trkNlayer,"trkNlayer[nTrk]/b");
  trackTree_->Branch("trkEta",&pev_.trkEta,"trkEta[nTrk]/F");
  trackTree_->Branch("trkPhi",&pev_.trkPhi,"trkPhi[nTrk]/F");
  trackTree_->Branch("trkCharge",&pev_.trkCharge,"trkCharge[nTrk]/I");
  if(doTrackVtxWImpPar_){
    trackTree_->Branch("trkNVtx",&pev_.trkNVtx,"trkNVtx[nTrk]/b");
    trackTree_->Branch("nTrkTimesnVtx",&pev_.nTrkTimesnVtx,"nTrkTimesnVtx/I");
    trackTree_->Branch("trkAssocVtx",&pev_.trkAssocVtx,"trkAssocVtx[nTrkTimesnVtx]/O");
    trackTree_->Branch("trkDxyOverDxyError",&pev_.trkDxyOverDxyError,"trkDxyOverDxyError[nTrkTimesnVtx]/F");
    trackTree_->Branch("trkDzOverDzError",&pev_.trkDzOverDzError,"trkDzOverDzError[nTrkTimesnVtx]/F");
  }
  else{
    trackTree_->Branch("trkVtxIndex",&pev_.trkVtxIndex,"trkVtxIndex[nTrk]/I");
  }

  if (doDeDx_) {
    trackTree_->Branch("dedx",&pev_.dedx,"dedx[nTrk]/F");
  }

  //  trackTree_->Branch("trkQual",&pev_.trkQual,"trkQual[nTrk]/I");

  for(unsigned int i  = 0; i < qualityStrings_.size(); ++i){
    trackTree_->Branch(qualityStrings_[i].data(),&pev_.trkQual[i],(qualityStrings_[i]+"[nTrk]/O").data());
  }


  trackTree_->Branch("trkChi2",&pev_.trkChi2,"trkChi2[nTrk]/F");
  trackTree_->Branch("trkNdof",&pev_.trkNdof,"trkNdof[nTrk]/b");
  trackTree_->Branch("trkDxy1",&pev_.trkDxy1,"trkDxy1[nTrk]/F");
  trackTree_->Branch("trkDxyError1",&pev_.trkDxyError1,"trkDxyError1[nTrk]/F");
  trackTree_->Branch("trkDz1",&pev_.trkDz1,"trkDz1[nTrk]/F");
  trackTree_->Branch("trkDzError1",&pev_.trkDzError1,"trkDzError1[nTrk]/F");
  //trackTree_->Branch("trkDzError2",&pev_.trkDzError2,"trkDzError2[nTrk]/F");
  //trackTree_->Branch("trkDxy2",&pev_.trkDxy2,"trkDxy2[nTrk]/F");
  //trackTree_->Branch("trkDz2",&pev_.trkDz2,"trkDz2[nTrk]/F");
  //trackTree_->Branch("trkDxyError2",&pev_.trkDxyError2,"trkDxyError2[nTrk]/F");
  trackTree_->Branch("trkFake",&pev_.trkFake,"trkFake[nTrk]/O");
  trackTree_->Branch("trkAlgo",&pev_.trkAlgo,"trkAlgo[nTrk]/b");
  trackTree_->Branch("trkOriginalAlgo",&pev_.trkOriginalAlgo,"trkOriginalAlgo[nTrk]/b");
  if(doMVA_){
   trackTree_->Branch("trkMVA",&pev_.trkMVA,"trkMVA[nTrk]/F");
   if(mvaSrcLabel_.label() == "generalTracks"){
    trackTree_->Branch("trkMVALoose",&pev_.trkMVALoose,"trkMVALoose[nTrk]/O");
    trackTree_->Branch("trkMVATight",&pev_.trkMVATight,"trkMVATight[nTrk]/O");  
   }
   if(mvaSrcLabel_.label() == "hiGeneralTracks")    trackTree_->Branch("trkMVATight",&pev_.trkMVATight,"trkMVATight[nTrk]/O");  
  }
  if (doDebug_) {
    trackTree_->Branch("trkNlayer3D",&pev_.trkNlayer3D,"trkNlayer3D[nTrk]/I");
    trackTree_->Branch("trkDxyBS",&pev_.trkDxyBS,"trkDxyBS[nTrk]/F");
    trackTree_->Branch("trkDxyErrorBS",&pev_.trkDxyErrorBS,"trkDxyErrorBS[nTrk]/F");
    trackTree_->Branch("trkDxy",&pev_.trkDxy,"trkDxy[nTrk]/F");
    trackTree_->Branch("trkDz",&pev_.trkDz,"trkDz[nTrk]/F");
    trackTree_->Branch("trkDxyError",&pev_.trkDxyError,"trkDxyError[nTrk]/F");
    trackTree_->Branch("trkDzError",&pev_.trkDzError,"trkDzError[nTrk]/F");
    trackTree_->Branch("trkChi2hit1D",&pev_.trkChi2hit1D,"trkChi2hit1D[nTrk]/F");
    trackTree_->Branch("trkVx",&pev_.trkVx,"trkVx[nTrk]/F");
    trackTree_->Branch("trkVy",&pev_.trkVy,"trkVy[nTrk]/F");
    trackTree_->Branch("trkVz",&pev_.trkVz,"trkVz[nTrk]/F");
  }

  if (doPFMatching_) {
    trackTree_->Branch("pfType",&pev_.pfType,"pfType[nTrk]/I");
    trackTree_->Branch("pfCandPt",&pev_.pfCandPt,"pfCandPt[nTrk]/F");
    trackTree_->Branch("pfEcal",&pev_.pfEcal,"pfEcal[nTrk]/F");
    trackTree_->Branch("pfHcal",&pev_.pfHcal,"pfHcal[nTrk]/F");
  }

  // Track Extra
  if (doTrackExtra_) {
    trackTree_->Branch("trkExpHit1Eta",&pev_.trkExpHit1Eta,"trkExpHit1Eta[nTrk]/F");
    trackTree_->Branch("trkExpHit2Eta",&pev_.trkExpHit2Eta,"trkExpHit2Eta[nTrk]/F");
    trackTree_->Branch("trkExpHit3Eta",&pev_.trkExpHit3Eta,"trkExpHit3Eta[nTrk]/F");
  }

  // Sim Tracks
  if (doSimTrack_) {
    trackTree_->Branch("trkStatus",&pev_.trkStatus,"trkStatus[nTrk]/F");
    trackTree_->Branch("trkPId",&pev_.trkPId,"trkPId[nTrk]/F");
    trackTree_->Branch("trkMPId",&pev_.trkMPId,"trkMPId[nTrk]/F");
    trackTree_->Branch("trkGMPId",&pev_.trkGMPId,"trkGMPId[nTrk]/F");
    trackTree_->Branch("matchedGenID",&pev_.matchedGenID,"matchedGenID[nTrk][5]/I");

    if(fillSimTrack_){

      trackTree_->Branch("nParticle",&pev_.nParticle,"nParticle/I");
      trackTree_->Branch("pStatus",&pev_.pStatus,"pStatus[nParticle]/I");
      trackTree_->Branch("pPId",&pev_.pPId,"pPId[nParticle]/I");
      trackTree_->Branch("pEta",&pev_.pEta,"pEta[nParticle]/F");
      trackTree_->Branch("pPhi",&pev_.pPhi,"pPhi[nParticle]/F");
      trackTree_->Branch("pPt",&pev_.pPt,"pPt[nParticle]/F");
      trackTree_->Branch("pAcc",&pev_.pAcc,"pAcc[nParticle]/F");
      trackTree_->Branch("pAccPair",&pev_.pAccPair,"pAccPair[nParticle]/F");
      trackTree_->Branch("pNRec",&pev_.pNRec,"pNRec[nParticle]/I");
      trackTree_->Branch("pNHit",&pev_.pNHit,"pNHit[nParticle]/I");
      trackTree_->Branch("mtrkPt",&pev_.mtrkPt,"mtrkPt[nParticle]/F");
      trackTree_->Branch("mtrkPtError",&pev_.mtrkPtError,"mtrkPtError[nParticle]/F");
      trackTree_->Branch("mtrkNHit",&pev_.mtrkNHit,"mtrkNHit[nParticle]/I");
      trackTree_->Branch("mtrkNlayer",&pev_.mtrkNlayer,"mtrkNlayer[nParticle]/I");
      trackTree_->Branch("mtrkNlayer3D",&pev_.mtrkNlayer3D,"mtrkNlayer3D[nParticle]/I");
      // trackTree_->Branch("mtrkQual",&pev_.mtrkQual,"mtrkQual[nParticle]/I");
      for(unsigned int i  = 0; i < qualityStrings_.size(); ++i){
        trackTree_->Branch(("m"+qualityStrings_[i]).data(),&pev_.mtrkQual[i],("m"+qualityStrings_[i]+"[nParticle]/O").data());
      }
      trackTree_->Branch("mtrkChi2",&pev_.mtrkChi2,"mtrkChi2[nParticle]/F");
      trackTree_->Branch("mtrkNdof",&pev_.mtrkNdof,"mtrkNdof[nParticle]/I");
      trackTree_->Branch("mtrkDz1",&pev_.mtrkDz1,"mtrkDz1[nParticle]/F");
      trackTree_->Branch("mtrkDzError1",&pev_.mtrkDzError1,"mtrkDzError1[nParticle]/F");
      trackTree_->Branch("mtrkDxy1",&pev_.mtrkDxy1,"mtrkDxy1[nParticle]/F");
      trackTree_->Branch("mtrkDxyError1",&pev_.mtrkDxyError1,"mtrkDxyError1[nParticle]/F");
      //trackTree_->Branch("mtrkDz2",&pev_.mtrkDz2,"mtrkDz2[nParticle]/F");
      //trackTree_->Branch("mtrkDzError2",&pev_.mtrkDzError2,"mtrkDzError2[nParticle]/F");
      //trackTree_->Branch("mtrkDxy2",&pev_.mtrkDxy2,"mtrkDxy2[nParticle]/F");
      //trackTree_->Branch("mtrkDxyError2",&pev_.mtrkDxyError2,"mtrkDxyError2[nParticle]/F");
      trackTree_->Branch("mtrkAlgo",&pev_.mtrkAlgo,"mtrkAlgo[nParticle]/I");
      trackTree_->Branch("mtrkOriginalAlgo",&pev_.mtrkOriginalAlgo,"mtrkOriginalAlgo[nParticle]/I");
      if(doTrackVtxWImpPar_){
        trackTree_->Branch("nParticleTimesnVtx",&pev_.nParticleTimesnVtx,"nParticleTimesnVtx/I");
        trackTree_->Branch("mtrkDzOverDzError",&pev_.mtrkDzOverDzError,"mtrkDzOverDzError[nParticleTimesnVtx]/F");
        trackTree_->Branch("mtrkDxyOverDxyError",&pev_.mtrkDxyOverDxyError,"mtrkDxyOverDxyError[nParticleTimesnVtx]/F");
      }
      if(doMVA_){
 	trackTree_->Branch("mtrkMVA",&pev_.mtrkMVA,"mtrkMVA[nParticle]/F");
	   
        if(mvaSrcLabel_.label() == "generalTracks"){
        trackTree_->Branch("mtrkMVALoose",&pev_.mtrkMVALoose,"mtrkMVALoose[nTrk]/O");
        trackTree_->Branch("mtrkMVATight",&pev_.mtrkMVATight,"mtrkMVATight[nTrk]/O");  
       }
       if(mvaSrcLabel_.label() == "hiGeneralTracks")    trackTree_->Branch("mtrkMVATight",&pev_.mtrkMVATight,"mtrkMVATight[nTrk]/O");  
      }
      if (doPFMatching_) {
	trackTree_->Branch("mtrkPfType",&pev_.mtrkPfType,"mtrkPfType[nParticle]/I");
	trackTree_->Branch("mtrkPfCandPt",&pev_.mtrkPfCandPt,"mtrkPfCandPt[nParticle]/F");
	trackTree_->Branch("mtrkPfEcal",&pev_.mtrkPfEcal,"mtrkPfEcal[nParticle]/F");
	trackTree_->Branch("mtrkPfHcal",&pev_.mtrkPfHcal,"mtrkPfHcal[nParticle]/F");
      }
    }
  }


}

// ------------ method called once each job just after ending the event loop  ------------
void
TrackAnalyzer::endJob() {
}

//define this as a plug-in
DEFINE_FWK_MODULE(TrackAnalyzer);
