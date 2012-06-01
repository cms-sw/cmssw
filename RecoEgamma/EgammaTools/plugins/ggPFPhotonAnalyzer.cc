#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "RecoEgamma/EgammaTools/interface/ggPFPhotonAnalyzer.h"
#include "RecoEcal/EgammaCoreTools/interface/EcalClusterLazyTools.h"
ggPFPhotonAnalyzer::ggPFPhotonAnalyzer(const edm::ParameterSet& iConfig){
  PFPhotonTag_=iConfig.getParameter<InputTag>("PFPhotons");
  PFElectronTag_=iConfig.getParameter<InputTag>("PFElectrons");
  
  recoPhotonTag_=iConfig.getParameter<InputTag>("Photons");
  ebReducedRecHitCollection_=iConfig.getParameter<InputTag>("ebReducedRecHitCollection");
  eeReducedRecHitCollection_=iConfig.getParameter<InputTag>("eeReducedRecHitCollection");
  esRecHitCollection_=iConfig.getParameter<InputTag>("esRecHitCollection");
  beamSpotCollection_ =iConfig.getParameter<InputTag>("BeamSpotCollection");
  pfPartTag_=iConfig.getParameter<InputTag>("PFParticles");
  TFile *fgbr1 = new TFile("/afs/cern.ch/work/r/rpatel/public/TMVARegressionBarrelLC.root","READ");
  TFile *fgbr2 = new TFile("/afs/cern.ch/work/r/rpatel/public/TMVARegressionEndCapLC.root","READ");
  PFLCBarrel_=(const GBRForest*)fgbr1->Get("PFLCorrEB");
  PFLCEndcap_=(const GBRForest*)fgbr2->Get("PFLCorrEE");
  tf1=new TFile("PF_test.root", "RECREATE");
  pf=new TTree("pf", "PFPhotons");
  pfclus=new TTree("pflcus", "PFClusters");
  pf->Branch("isConv", &isConv_, "isConv/I");
  pf->Branch("hasSLConv", &hasSLConv_, "hasSLConv/I");
  pf->Branch("isMatch", &isMatch_, "isMatch/I");
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
  pf->Branch("PFPhoECorr", &PFPhoECorr_, "PFPhoECorr/F"); 
  pf->Branch("recoPFEnergy", &recoPFEnergy_, "recoPFEnergy/F"); 
  pf->Branch("SCRawE", &SCRawE_, "SCRawE");
}

ggPFPhotonAnalyzer::~ggPFPhotonAnalyzer(){}

void ggPFPhotonAnalyzer::beginRun(const edm::Run & r, const edm::EventSetup & c){
  
  
}

void ggPFPhotonAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& es){
  Handle<reco::PhotonCollection> PFPhotons;
  Handle<reco::PhotonCollection> recoPhotons;
  Handle<reco::GsfElectronCollection> PFElectrons;
  Handle<reco::PFCandidateCollection>PFParticles;
  PhotonCollection::const_iterator iPfPho;
  PhotonCollection::const_iterator iPho;
  iEvent.getByLabel(PFPhotonTag_, PFPhotons);
  iEvent.getByLabel(recoPhotonTag_, recoPhotons);
  iEvent.getByLabel(PFElectronTag_,PFElectrons);
  iEvent.getByLabel(pfPartTag_,PFParticles);
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
  EcalClusterLazyTools lazyToolEcal(iEvent, es, ebReducedRecHitCollection_, eeReducedRecHitCollection_);
  for(reco::PhotonCollection::const_iterator iPho = recoPhotons->begin(); iPho!=recoPhotons->end(); ++iPho) {
    recoPFEnergy_=0;
	
    ggPFPhotons ggPFPhoton(*iPho, PFPhotons,
			   PFElectrons,
			  PFParticles,
			   EBReducedRecHits,
			   EEReducedRecHits,
			   ESRecHits,
			   geomBar_,
			   geomEnd_,
			   beamSpotHandle
			   );
    if(ggPFPhoton.MatchPFReco()){
      isMatch_=1;
      
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
      std::vector<reco::CaloCluster>PFC=ggPFPhoton.PFClusters();
      PFPhoECorr_=ggPFPhoton.getPFPhoECorr(PFC, PFLCBarrel_, PFLCEndcap_);
    }
    else{
      isMatch_=0;
      std::vector<reco::CaloCluster>PFC;
      std::vector<reco::PFCandidatePtr>insideBox;
      ggPFPhoton.PhotonPFCandMatch(*(iPho->superCluster()), insideBox,PFParticles,PFC);  
      recoPFEnergy_=0;
      //cout<<"Inside Box "<<insideBox.size()<<endl;
      for(unsigned int i=0; i<PFC.size(); ++i)recoPFEnergy_=recoPFEnergy_+PFC[i].energy();
      SCRawE_=iPho->superCluster()->rawEnergy();
      //cout<<"PF reconstructed E "<<recoPFEnergy_<<"SC Raw E "<<(*iPho).superCluster()->rawEnergy()<<endl;
      PFPS1_=ggPFPhoton.PFPS1();
      PFPS2_=ggPFPhoton.PFPS2();
      MustE_=ggPFPhoton.MustE();
      MustEOut_=ggPFPhoton.MustEOut();
      PFLowCE_=ggPFPhoton.PFLowE();
      PFdEta_=ggPFPhoton.PFdEta();
      PFdPhi_=ggPFPhoton.PFdPhi();
      PFClusRMS_=ggPFPhoton.PFClusRMSTot();
      PFClusRMSMust_=ggPFPhoton.PFClusRMSMust();
    }
    pf->Fill();
  }
  
}


void ggPFPhotonAnalyzer::endJob(){
  tf1->cd();
  pf->Write();
  tf1->Write();
  tf1->Close();

}


DEFINE_FWK_MODULE(ggPFPhotonAnalyzer);

