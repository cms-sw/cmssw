#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "RecoEgamma/EgammaTools/interface/ggPFPhotonAnalyzer.h"


ggPFPhotonAnalyzer::ggPFPhotonAnalyzer(const edm::ParameterSet& iConfig){
  PFPhotonTag_=iConfig.getParameter<InputTag>("PFPhotons");
  recoPhotonTag_=iConfig.getParameter<InputTag>("Photons");
  ebReducedRecHitCollection_=iConfig.getParameter<InputTag>("ebReducedRecHitCollection");
  eeReducedRecHitCollection_=iConfig.getParameter<InputTag>("eeReducedRecHitCollection");
  esRecHitCollection_=iConfig.getParameter<InputTag>("esRecHitCollection");
  beamSpotCollection_ =iConfig.getParameter<InputTag>("BeamSpotCollection");
  
  tf1=new TFile("PF_test.root", "RECREATE");
  pf=new TTree("pf", "PFPhotons");
  pfclus=new TTree("pflcus", "PFClusters");
  pf->Branch("isConv", &isConv_, "isConv/I");
  pf->Branch("hasSLConv", &hasSLConv_, "hasSLConv/I");
  pf->Branch("PFPS1", &PFPS1_, "PFPS1/F");
  pf->Branch("PFPS2", &PFPS2_, "PFPS2/F");
  pf->Branch("MustE", &MustE_, "MustE/F");
  pf->Branch("MustEOut", &MustEOut_, "MustEOut/F");
  pf->Branch("PFLowCE", &PFLowCE_, "PFLowCE/F"); 
  pf->Branch("PFdEta", &PFdEta_, "PFdEta/F");
  pf->Branch("PFdPhi", &PFdPhi_, "PFdPhi/F"); 
  pf->Branch("PFClusRMS", &PFClusRMS_, "PFClusRMS/F"); 
  pf->Branch("PFClusRMSMust", &PFClusRMSMust_, "PFClusRMSMust/F"); 
  pf->Branch("VtxZ", &VtxZ_, "VtxZ/F"); 
  pf->Branch("VtxZErr", &VtxZErr_, "VtxZErr/F"); 
  
}

ggPFPhotonAnalyzer::~ggPFPhotonAnalyzer(){}

void ggPFPhotonAnalyzer::beginRun(const edm::Run & r, const edm::EventSetup & c){
  
  
}

void ggPFPhotonAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& es){
  Handle<reco::PhotonCollection> PFPhotons;
  Handle<reco::PhotonCollection> recoPhotons;
  PhotonCollection::const_iterator iPfPho;
  PhotonCollection::const_iterator iPho;
  iEvent.getByLabel(PFPhotonTag_, PFPhotons);
  iEvent.getByLabel(recoPhotonTag_, recoPhotons);
  //for PFPhoton Constructor:
  edm::ESHandle<CaloGeometry> pG;
  es.get<CaloGeometryRecord>().get(pG);
  geomBar_=pG->getSubdetectorGeometry(DetId::Ecal,1);
  geomEnd_=pG->getSubdetectorGeometry(DetId::Ecal,2);
  edm::Handle<BeamSpot> beamSpotHandle;
  edm::Handle<EcalRecHitCollection> EBReducedRecHits;
  edm::Handle<EcalRecHitCollection> EEReducedRecHits;
  edm::Handle<EcalRecHitCollection> ESRecHits; 
  iEvent.getByLabel(beamSpotCollection_    , beamSpotHandle);
  iEvent.getByLabel(ebReducedRecHitCollection_, EBReducedRecHits);
  iEvent.getByLabel(eeReducedRecHitCollection_, EEReducedRecHits);
  iEvent.getByLabel(esRecHitCollection_       , ESRecHits);
  iEvent.getByLabel(beamSpotCollection_,beamSpotHandle);
  for(reco::PhotonCollection::const_iterator iPho = recoPhotons->begin(); iPho!=recoPhotons->end(); ++iPho) {
    ggPFPhotons ggPFPhoton(*iPho, PFPhotons,
			   EBReducedRecHits,
			   EEReducedRecHits,
			   ESRecHits,
			   geomBar_,
			   geomEnd_,
			   beamSpotHandle
			   );
    if(ggPFPhoton.MatchPFReco()){
      std::pair<float, float>VertexZ=ggPFPhoton.SLPoint();
      VtxZ_=VertexZ.first;
      VtxZErr_=VertexZ.second;
      
      if(ggPFPhoton.isConv()){
	isConv_=1;
      }
      else isConv_=0;
      if(ggPFPhoton.hasSLConv()){
	hasSLConv_=1;
      }
      else hasSLConv_=0;
      
      ggPFPhoton.fillPFClusters();
      PFPS1_=ggPFPhoton.PFPS1();
      PFPS2_=ggPFPhoton.PFPS2();
      MustE_=ggPFPhoton.MustE();
      MustEOut_=ggPFPhoton.MustEOut();
      PFLowCE_=ggPFPhoton.PFLowE();
      PFdEta_=ggPFPhoton.PFdEta();
      PFdPhi_=ggPFPhoton.PFdPhi();
      PFClusRMS_=ggPFPhoton.PFClusRMSTot();
      PFClusRMSMust_=ggPFPhoton.PFClusRMSMust();
      ggPFPhoton.PFClusters();
      pf->Fill();
    }
  }
  
}


void ggPFPhotonAnalyzer::endJob(){
  tf1->cd();
  pf->Write();
  tf1->Write();
  tf1->Close();

}


DEFINE_FWK_MODULE(ggPFPhotonAnalyzer);
