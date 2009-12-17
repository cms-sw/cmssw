#include <memory>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/Framework/interface/ESHandle.h"

#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/JetReco/interface/Jet.h"
#include "DataFormats/JetReco/interface/GenJet.h"
#include "DataFormats/JetReco/interface/GenJetCollection.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/JetReco/interface/BasicJet.h"
#include "DataFormats/JetReco/interface/BasicJetCollection.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"

#include <iostream>
#include <string>
#include <vector>
#include "TFile.h"
#include "TH1.h"
#include "TH2.h"
#include "TTree.h"
#include "TMath.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "PhysicsTools/UtilAlgos/interface/TFileService.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "SimDataFormats/GeneratorProducts/interface/GenEventInfoProduct.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"

// access trigger results
#include <FWCore/Framework/interface/TriggerNames.h>
#include <DataFormats/Common/interface/TriggerResults.h>
#include <DataFormats/HLTReco/interface/TriggerEvent.h> 
#include <DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h>

#include "DataFormats/Common/interface/TriggerResults.h"
#include "FWCore/Framework/interface/TriggerNames.h"
#include <DataFormats/HLTReco/interface/TriggerEvent.h>
#include <DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h>
#include "CondFormats/DataRecord/interface/L1GtTriggerMenuRcd.h"
#include "CondFormats/L1TObjects/interface/L1GtTriggerMenu.h"
#include "CondFormats/L1TObjects/interface/L1GtTriggerMenuFwd.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutSetupFwd.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutSetup.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutRecord.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerObjectMapRecord.h"

//handle recHits and clusters: maybe some package are redundant
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2D.h"
#include "DataFormats/TrackerRecHit2D/interface/ProjectedSiStripRecHit2D.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripMatchedRecHit2D.h"
#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripMatchedRecHit2DCollection.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2DCollection.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHitCollection.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "DataFormats/SiPixelDetId/interface/PXBDetId.h"
#include "DataFormats/SiPixelDetId/interface/PXFDetId.h"
#include "DataFormats/SiPixelCluster/interface/SiPixelCluster.h"


using namespace edm;
using namespace reco;
using namespace std;

bool DEBUG = false;

class Event_Ntuplizer : public edm::EDAnalyzer {
 
 public:
 
   Event_Ntuplizer(const edm::ParameterSet& iConfig);
   ~Event_Ntuplizer(){}

   virtual void beginJob(const edm::EventSetup&);
   virtual void analyze(const edm::Event&, const edm::EventSetup&);
   virtual void endJob();
 
 private:
  
  edm::InputTag particleCollection_;
  edm::InputTag vertexCollection_;
  edm::InputTag CaloJetsInputTag_;
  edm::InputTag genEventScaleTag_;
  edm::InputTag pixelClusterInput_;
  CaloJetCollection theCaloJets;
  bool OnlyRECO;
  
  Service<TFileService> fs;
  TTree* EventTree;
  
  int eventNum;
  int lumiBlock;
  int runNumber;
  int bunchCrossing;

  int numVertices;
  double vertex_x[30];
  double vertex_y[30];
  double vertex_z[30];
  double vertex_xError[30];
  double vertex_yError[30];
  double vertex_zError[30];
  int vertex_nOutgoingTracks[30]; 
  
  int numTracks;
  double track_eta[10000];
  double track_phi[10000];
  double track_p[10000];
  double track_pt[10000];
  double track_px[10000];
  double track_py[10000];
  double track_pz[10000];
  double track_d0[10000];
  double track_d0Error[10000];
  double track_dz[10000];
  double track_dzError[10000];
  double track_recHitsSize[10000];
  double track_chi2[10000];
  double track_ndof[10000];
  double track_normalizedChi2[10000];
  double track_vx[10000];
  double track_vy[10000];
  double track_vz[10000];
  int track_loose[10000];
  int track_tight[10000];
  int track_highPurity[10000];
  int track_charge[10000];
  
  double vtxAssocTrack_eta[10000];
  double vtxAssocTrack_phi[10000];
  double vtxAssocTrack_p[10000];
  double vtxAssocTrack_pt[10000];
  double vtxAssocTrack_px[10000];
  double vtxAssocTrack_py[10000];
  double vtxAssocTrack_pz[10000];
  double vtxAssocTrack_d0[10000];
  double vtxAssocTrack_d0Error[10000];
  double vtxAssocTrack_dz[10000];
  double vtxAssocTrack_dzError[10000];
  double vtxAssocTrack_recHitsSize[10000];
  double vtxAssocTrack_chi2[10000];
  double vtxAssocTrack_ndof[10000];
  double vtxAssocTrack_normalizedChi2[10000];
  double vtxAssocTrack_vx[10000];
  double vtxAssocTrack_vy[10000];
  double vtxAssocTrack_vz[10000];

  int genEventScale;
  int L1Trigger[200]; 
  int HLTTrigger[200];

  int cluster_size[50000];
  int clusterTotal;
    
  //***************************************trigger
  
  double maxOrbit_;
  double minOrbit_;
  bool HLTCut_;
  vector<string> HLTPaths_;
  bool L1TechCut_;
  vector<string> L1TechPaths_byName_;
  vector<int> L1TechPaths_byBit_;
  string L1TechComb_byBit_;
  
  //***************************************
  class PtSorter {
  public:
    template <class T> bool operator() ( const T& a, const T& b ) {
      return ( a.pt() > b.pt() );
    }
  };


};

Event_Ntuplizer::Event_Ntuplizer(const edm::ParameterSet& iConfig):
  particleCollection_( iConfig.getParameter<edm::InputTag>("particleCollection")), 
  vertexCollection_( iConfig.getParameter<edm::InputTag>("vertexCollection")), 
  //CaloJetsInputTag_( iConfig.getParameter<edm::InputTag>( "CaloJetCollectionName")),
  genEventScaleTag_(iConfig.getParameter<InputTag>( "genEventScale")),
  OnlyRECO(iConfig.getParameter<bool>("OnlyRECO")),
  pixelClusterInput_( iConfig.getParameter<edm::InputTag>("pixelClusterInput") )
  // HLTCut_(iConfig.getUntrackedParameter<bool>("HLTCut",false)),
  //HLTPaths_(iConfig.getParameter< vector<string> >("HLTPaths")),
  //L1TechCut_(iConfig.getUntrackedParameter<bool>("L1TechCut",false)),
  //L1TechPaths_byName_(iConfig.getParameter< vector<string> >("L1TechPaths_byName")),
  //L1TechPaths_byBit_(iConfig.getParameter< vector<int> >("L1TechPaths_byBit")),
  //L1TechComb_byBit_(iConfig.getParameter< string >("L1TechComb_byBit"))
 {
 
  EventTree = fs->make<TTree>("EventTree","EventTree");
    
  }
void Event_Ntuplizer::beginJob(const edm::EventSetup& setup)
{
  for(int i=0;i<200;i++){ L1Trigger[i]=0; HLTTrigger[i]=0;}

  EventTree->Branch("eventNum",&eventNum,"eventNum/I");
  EventTree->Branch("lumiBlock",&lumiBlock,"lumiBlock/I");
  EventTree->Branch("runNumber",&runNumber,"runNumber/I");
  EventTree->Branch("bunchCrossing",&bunchCrossing,"bunchCrossing/I");

  EventTree->Branch("numVertices",&numVertices,"numVertices/I");
  EventTree->Branch("vertex_x",vertex_x,"vertex_x[30]/D");
  EventTree->Branch("vertex_y",vertex_y,"vertex_y[30]/D");
  EventTree->Branch("vertex_z",vertex_z,"vertex_z[30]/D");
  EventTree->Branch("vertex_xError",vertex_xError,"vertex_xError[30]/D");
  EventTree->Branch("vertex_yError",vertex_yError,"vertex_yError[30]/D");
  EventTree->Branch("vertex_zError",vertex_zError,"vertex_zError[30]/D");
  EventTree->Branch("vertex_nOutgoingTracks",vertex_nOutgoingTracks,"vertex_nOutgoingTracks[30]/I");
 
  EventTree->Branch("numTracks",&numTracks,"numTracks/I");
  EventTree->Branch("track_eta",track_eta,"track_eta[10000]/D");
  EventTree->Branch("track_phi",track_phi,"track_phi[10000]/D");
  EventTree->Branch("track_p",track_p,"track_p[10000]/D");
  EventTree->Branch("track_pt",track_pt,"track_pt[10000]/D");
  EventTree->Branch("track_px",track_px,"track_px[10000]/D");
  EventTree->Branch("track_py",track_py,"track_py[10000]/D");
  EventTree->Branch("track_pz",track_pz,"track_pz[10000]/D");
  EventTree->Branch("track_d0",track_d0,"track_d0[10000]/D");
  EventTree->Branch("track_d0Error",track_d0Error,"track_d0Error[10000]/D");
  EventTree->Branch("track_dz",track_dz,"track_dz[10000]/D");
  EventTree->Branch("track_dzError",track_dzError,"track_dzError[10000]/D");
  EventTree->Branch("track_recHitsSize",track_recHitsSize,"track_recHitsSize[10000]/D");
  EventTree->Branch("track_chi2",track_chi2,"track_chi2[10000]/D");
  EventTree->Branch("track_ndof",track_ndof,"track_ndof[10000]/D");
  EventTree->Branch("track_normalizedChi2",track_normalizedChi2,"track_normalizedChi2[10000]/D");
  EventTree->Branch("track_vx",track_vx,"track_vx[10000]/D");
  EventTree->Branch("track_vy",track_vy,"track_vy[10000]/D");
  EventTree->Branch("track_vz",track_vz,"track_vz[10000]/D");
  EventTree->Branch("track_loose",track_loose,"track_loose[10000]/I");
  EventTree->Branch("track_tight",track_tight,"track_tight[10000]/I");
  EventTree->Branch("track_highPurity",track_highPurity,"track_highPurity[10000]/I");
  EventTree->Branch("track_charge",track_charge,"track_charge[10000]/I");

  EventTree->Branch("vtxAssocTrack_eta",vtxAssocTrack_eta,"vtxAssocTrack_eta[10000]/D");
  EventTree->Branch("vtxAssocTrack_phi",vtxAssocTrack_phi,"vtxAssocTrack_phi[10000]/D");
  EventTree->Branch("vtxAssocTrack_p",vtxAssocTrack_p,"vtxAssocTrack_p[10000]/D");
  EventTree->Branch("vtxAssocTrack_pt",vtxAssocTrack_pt,"vtxAssocTrack_pt[10000]/D");
  EventTree->Branch("vtxAssocTrack_px",vtxAssocTrack_px,"vtxAssocTrack_px[10000]/D");
  EventTree->Branch("vtxAssocTrack_py",vtxAssocTrack_py,"vtxAssocTrack_py[10000]/D");
  EventTree->Branch("vtxAssocTrack_pz",vtxAssocTrack_pz,"vtxAssocTrack_pz[10000]/D");
  EventTree->Branch("vtxAssocTrack_d0",vtxAssocTrack_d0,"vtxAssocTrack_d0[10000]/D");
  EventTree->Branch("vtxAssocTrack_d0Error",vtxAssocTrack_d0Error,"vtxAssocTrack_d0Error[10000]/D");
  EventTree->Branch("vtxAssocTrack_dz",vtxAssocTrack_dz,"vtxAssocTrack_dz[10000]/D");
  EventTree->Branch("vtxAssocTrack_dzError",vtxAssocTrack_dzError,"vtxAssocTrack_dzError[10000]/D");
  EventTree->Branch("vtxAssocTrack_recHitsSize",vtxAssocTrack_recHitsSize,"vtxAssocTrack_recHitsSize[10000]/D");
  EventTree->Branch("vtxAssocTrack_chi2",vtxAssocTrack_chi2,"vtxAssocTrack_chi2[10000]/D");
  EventTree->Branch("vtxAssocTrack_ndof",vtxAssocTrack_ndof,"vtxAssocTrack_ndof[10000]/D");
  EventTree->Branch("vtxAssocTrack_normalizedChi2",vtxAssocTrack_normalizedChi2,"vtxAssocTrack_normalizedChi2[10000]/D");
  EventTree->Branch("vtxAssocTrack_vx",vtxAssocTrack_vx,"vtxAssocTrack_vx[10000]/D");
  EventTree->Branch("vtxAssocTrack_vy",vtxAssocTrack_vy,"vtxAssocTrack_vy[10000]/D");
  EventTree->Branch("vtxAssocTrack_vz",vtxAssocTrack_vz,"vtxAssocTrack_vz[10000]/D");
  EventTree->Branch("genEventScale",&genEventScale,"genEventScale/I");
  //trigger L1 e HLT
  EventTree->Branch("L1Trigger",L1Trigger,"L1Trigger[200]/I");
  EventTree->Branch("HLTTrigger",HLTTrigger,"HLTTrigger[200]/I");

  EventTree->Branch("cluster_size",cluster_size,"cluster_size[50000]/I");
  EventTree->Branch("clusterTotal",&clusterTotal,"clusterTotal/I");

}

void Event_Ntuplizer::endJob()
{
 
}
void Event_Ntuplizer::analyze(const edm::Event& iEvent, const edm::EventSetup& setup){

  eventNum      = iEvent.id().event() ;
  runNumber     = iEvent.id().run() ;
  lumiBlock     = iEvent.luminosityBlock() ;
  bunchCrossing = iEvent.bunchCrossing();

  edm::Handle< reco::TrackCollection  > trackColl;
  edm::Handle< reco::VertexCollection > vertexColl;
  edm::Handle< edmNew::DetSetVector<SiPixelCluster> > theClusters;
  //edm::Handle< CaloJetCollection > CaloJetColl ;

  iEvent.getByLabel(particleCollection_, trackColl);
  iEvent.getByLabel(vertexCollection_, vertexColl);
  iEvent.getByLabel(pixelClusterInput_, theClusters);
  //iEvent.getByLabel( CaloJetsInputTag_, CaloJetColl );

  const edmNew::DetSetVector<SiPixelCluster>& input = *theClusters;

  if(!OnlyRECO){
    Handle< HepMCProduct > EvtHandle ;
    iEvent.getByLabel( genEventScaleTag_, EvtHandle );
    const HepMC::GenEvent* Evt = EvtHandle->GetEvent() ;
    genEventScale = Evt->signal_process_id();
    }
  
  
//Trigger L1 info ********
/*
  bool useEvent_L1Tech = false;
  
  edm::Handle<L1GlobalTriggerReadoutRecord> L1GTRR;
  try{iEvent.getByLabel("gtDigis",L1GTRR);}
  catch(...){cout<<"The L1 Trigger branch was not correctly taken"<<endl;}
  
  /*Handle<L1GlobalTriggerObjectMapRecord> hL1GTmap; 
  try{iEvent.getByLabel("hltL1GtObjectMap", hL1GTmap);}
  catch(...){cout<<"The L1 Trigger map was not correctly taken"<<endl;} //da commentare
  
  edm::ESHandle<L1GtTriggerMenu> hL1GtMenu;
  try{setup.get<L1GtTriggerMenuRcd>().get(hL1GtMenu);}
  catch(...){cout<<"The L1 Trigger menu was not correctly taken"<<endl;}
  const L1GtTriggerMenu* l1GtMenu = hL1GtMenu.product();

  bool hasOneFalse=false;
  if (L1GTRR.isValid()) {
    cout<<"valid trigger..."<<ebdl;
    const AlgorithmMap& algorithmMap = l1GtMenu->gtTechnicalTriggerMap();
    for(vector<string>::iterator requested_l1Tech_bit=L1TechPaths_byName_.begin();
	  requested_l1Tech_bit!=L1TechPaths_byName_.end();++requested_l1Tech_bit){

      AlgorithmMap::const_iterator it=algorithmMap.find(*requested_l1Tech_bit);
      if(it==algorithmMap.end())
        cout<<"Wrong name for technical trigger:"<<*requested_l1Tech_bit<<endl;
      else{
        cout<<"good technical trigger: "<<*requested_l1Tech_bit<<endl;
        cout<<"L1GTRR->technicalTriggerWord()[(it->second).algoBitNumber()]"<<L1GTRR->technicalTriggerWord()[(it->second).algoBitNumber()]<<endl;
        if(L1GTRR->technicalTriggerWord()[(it->second).algoBitNumber()])
	  useEvent_L1Tech = true;}
    }
    
    for(vector<int>::iterator requested_l1Tech_bit=L1TechPaths_byBit_.begin();
	  requested_l1Tech_bit!=L1TechPaths_byBit_.end();++requested_l1Tech_bit){
        cout<<"L1GTRR->technicalTriggerWord()[*requested_l1Tech_bit]"<<L1GTRR->technicalTriggerWord()[*requested_l1Tech_bit]<<endl;
	if(L1GTRR->technicalTriggerWord()[*requested_l1Tech_bit])
	  useEvent_L1Tech = true;
	else
	  if(L1TechComb_byBit_=="AND")
	    hasOneFalse=true;
    }
    
  }
    

  ////HLT Trigger *******
/*
  bool useEvent_HLT = false;
  
  vector<Handle<TriggerResults> > trhv;
  try{iEvent.getManyByType(trhv);}
  catch(...){cout<<"The HLT Trigger branch was not correctly taken"<<endl;}
  const int nt(trhv.size());
  
  TriggerNames triggerNames;
  
  for(int ntt=0;ntt<nt;++ntt){
    triggerNames.init(*trhv[ntt]);
    vector<string> hltNames = triggerNames.triggerNames();
    cout<<"hlt names size is "<<hltNames.size()<<endl;
    int n = 0;
    for(vector<string>::const_iterator i = hltNames.begin();i!= hltNames.end(); ++i){
      for(vector<string>::iterator requested_hlt_bit=HLTPaths_.begin();//hlt_bits request by user
	  requested_hlt_bit!=HLTPaths_.end();++requested_hlt_bit){
	if(*i == *requested_hlt_bit){
	  if((*trhv[ntt]).accept(n) && HLTCut_){
	    useEvent_HLT = true;
	    break;
	  }
	}
      }
      n++;
    }//for i 
  }//for ntt
 
  if(HLTCut_ && !useEvent_HLT)
        cout<<"false"<<endl;

*/

//********  vertex info  ********

  int itV=0;
  int itAssocT=0;
  const reco::VertexCollection theVertices = *(vertexColl.product());
  numVertices = theVertices.size();  //surely at least one: the beam spot
  if (theVertices.size() > 0){
    for (reco::VertexCollection::const_iterator vertexIt = theVertices.begin(),
                                                   last = theVertices.end();
	              	                      vertexIt != last; ++vertexIt) {
      vertex_x[itV] = vertexIt->x();
      vertex_y[itV] = vertexIt->y();
      vertex_z[itV] = vertexIt->z();
      vertex_xError[itV] = vertexIt->xError();
      vertex_yError[itV] = vertexIt->yError();
      vertex_zError[itV] = vertexIt->zError();
      
      //getting number of tracks associated to the vertex
      vertex_nOutgoingTracks[itV] = 0;
    
        

      for(reco::Vertex::trackRef_iterator vTrackIter = vertexIt->tracks_begin(); 
                                         vTrackIter != vertexIt->tracks_end();
	                                 vTrackIter++){
         vertex_nOutgoingTracks[itV]++;
	const reco::Track & linkedTrack = *vTrackIter->get();
          vtxAssocTrack_eta[itAssocT]     = linkedTrack.eta();
          vtxAssocTrack_phi[itAssocT]     = linkedTrack.phi();
          vtxAssocTrack_p[itAssocT]       = linkedTrack.p();
          vtxAssocTrack_pt[itAssocT]      = linkedTrack.pt();
          vtxAssocTrack_d0[itAssocT]      = linkedTrack.d0();
          vtxAssocTrack_d0Error[itAssocT] = linkedTrack.d0Error();
          vtxAssocTrack_dz[itAssocT]      = linkedTrack.dz();
          vtxAssocTrack_dzError[itAssocT] = linkedTrack.dzError();
          vtxAssocTrack_recHitsSize[itAssocT]    = linkedTrack.recHitsSize();
          vtxAssocTrack_chi2[itAssocT]           = linkedTrack.chi2();
          vtxAssocTrack_ndof[itAssocT]           = linkedTrack.ndof();
          vtxAssocTrack_normalizedChi2[itAssocT] = linkedTrack.normalizedChi2();
          vtxAssocTrack_vx[itAssocT] = linkedTrack.vx();
          vtxAssocTrack_vy[itAssocT] = linkedTrack.vy();
          vtxAssocTrack_vz[itAssocT] = linkedTrack.vz();
	  itAssocT++;
	  }

      itV++;
      }
    }
 

//******* track info ********

  numTracks=0;
  for (reco::TrackCollection::const_iterator trackIt = trackColl->begin();
                                            trackIt != trackColl->end();
                                            trackIt++){
    numTracks++;
    }
  
  if (numTracks>0){
    int itT=0;  
    for (reco::TrackCollection::const_iterator trackIt=trackColl->begin();
                                               trackIt!=trackColl->end();
					       trackIt++){
string undefQuality="undefQuality";
string loose="loose";
string tight="tight";
string highPurity="highPurity";
string confirmed="confirmed";
string goodIterative="goodIterative";
string qualitySize="qualitySize";

if (DEBUG) cout<<"traccia "<<itT<<endl;
if (DEBUG) cout<<"    undefQuality: "<<trackIt->quality( reco::TrackBase::qualityByName(undefQuality) )<<endl;
if (DEBUG) cout<<"  quality mask"<<trackIt->qualityMask()<<endl;
if (DEBUG) cout<<"    loose: "<<trackIt->quality( reco::TrackBase::qualityByName(loose) )<<endl;
if (DEBUG) cout<<"    tight: "<<trackIt->quality( reco::TrackBase::qualityByName(tight) )<<endl;
if (DEBUG) cout<<"    highPurity: "<<trackIt->quality( reco::TrackBase::qualityByName(highPurity) )<<endl;
if (DEBUG) cout<<"    confirmed: "<<trackIt->quality( reco::TrackBase::qualityByName(confirmed) )<<endl;
if (DEBUG) cout<<"    goodIterative: "<<trackIt->quality( reco::TrackBase::qualityByName(goodIterative) )<<endl;
if (DEBUG) cout<<"    qualitySize: "<<trackIt->quality( reco::TrackBase::qualityByName(qualitySize) )<<endl;

      track_eta[itT]     = trackIt->eta();
      track_phi[itT]     = trackIt->phi();
      track_p[itT]       = trackIt->p();
      track_pt[itT]      = trackIt->pt();
      track_px[itT]      = trackIt->px();
      track_py[itT]      = trackIt->py();
      track_pz[itT]      = trackIt->pz();
      track_d0[itT]      = trackIt->d0();
      track_d0Error[itT] = trackIt->d0Error();
      track_dz[itT]      = trackIt->dz();
      track_dzError[itT] = trackIt->dzError();
      track_recHitsSize[itT]    = trackIt->recHitsSize();
      track_chi2[itT]           = trackIt->chi2();
      track_ndof[itT]           = trackIt->ndof();
      track_normalizedChi2[itT] = trackIt->normalizedChi2();
      track_vx[itT] = trackIt->vx();
      track_vy[itT] = trackIt->vy();
      track_vz[itT] = trackIt->vz();

      track_loose[itT]     = 0;
      track_tight[itT]     = 0;
      track_highPurity[itT]= 0;
      if ( trackIt->quality(reco::TrackBase::qualityByName(loose)) ) 
        track_loose[itT]    = 1;
      if ( trackIt->quality(reco::TrackBase::qualityByName(tight)) )
        track_tight[itT]     = 1;
      if ( trackIt->quality(reco::TrackBase::qualityByName(highPurity)) )
        track_highPurity[itT]= 1;;

      track_charge[itT] = trackIt->charge();
      
      itT++;
      }
    } 
 
  /*Jet Analysis  
  if( CaloJetColl->size()){
    theCaloJets = *CaloJetColl;
    std::stable_sort( theCaloJets.begin(), theCaloJets.end(), PtSorter() );

    CaloJetCollection::const_iterator it   ( theCaloJets.begin() );
    CaloJetCollection::const_iterator itEnd( theCaloJets.end()   );
    for ( ; it != itEnd; ++it ){
    
      }
    
    }
  */
//******** cluster INFO ********

  int itCluster=0;
  for (edmNew::DetSetVector<SiPixelCluster>::const_iterator DSViter = input.begin(); DSViter != input.end(); DSViter++)
    {
    edmNew::DetSet<SiPixelCluster>::const_iterator begin=DSViter->begin();
    edmNew::DetSet<SiPixelCluster>::const_iterator end  =DSViter->end();
    for(edmNew::DetSet<SiPixelCluster>::const_iterator clustIt=begin; clustIt!=end;++clustIt)
      {
      itCluster++;
      }
      if (itCluster>49000) break;
    }
  clusterTotal=itCluster;
  
  itCluster=0;
  for (edmNew::DetSetVector<SiPixelCluster>::const_iterator DSViter = input.begin(); DSViter != input.end(); DSViter++)
    {
    edmNew::DetSet<SiPixelCluster>::const_iterator begin=DSViter->begin();
    edmNew::DetSet<SiPixelCluster>::const_iterator end  =DSViter->end();
    for(edmNew::DetSet<SiPixelCluster>::const_iterator clustIt=begin; clustIt!=end;++clustIt)
      {
      cluster_size[itCluster]=(*clustIt).size();
      itCluster++;
      }
      if (itCluster>49000) break;
    }
	    



  EventTree->Fill();
  
}



DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(Event_Ntuplizer);

