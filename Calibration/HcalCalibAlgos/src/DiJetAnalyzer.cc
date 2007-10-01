#include "Calibration/HcalCalibAlgos/interface/DiJetAnalyzer.h"
#include "DataFormats/JetReco/interface/CaloJet.h"

#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"

#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"

namespace cms{

DiJetAnalyzer::DiJetAnalyzer(const edm::ParameterSet& iConfig)
{
   fOutputFileName = iConfig.getUntrackedParameter<string>("HistOutFile");
   jetsLabel_ = iConfig.getParameter<edm::InputTag>("jetsInput");
   ecalLabels_=iConfig.getParameter<std::vector<edm::InputTag> >("ecalInputs");
   hbheLabel_ = iConfig.getParameter<edm::InputTag>("hbheInput");
   hoLabel_ = iConfig.getParameter<edm::InputTag>("hoInput");
   hfLabel_=iConfig.getParameter<edm::InputTag>("hfInput");

}

DiJetAnalyzer::~DiJetAnalyzer()
{
                                                                                                                             
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)
                                                                                                                             
}

void DiJetAnalyzer::beginJob( const edm::EventSetup& iSetup)
{
   hOutputFile   = new TFile( fOutputFileName.c_str(), "RECREATE" ) ;
   myTree = new TTree("RecJet","RecJet Tree");
   myTree->Branch("eta_jet", &eta_jet, "eta_jet/D");
   myTree->Branch("phi_jet", &phi_jet, "phi_jet/D");
   myTree->Branch("px_jet", &px_jet, "px_jet/D");
   myTree->Branch("py_jet", &py_jet, "py_jet/D");
   myTree->Branch("pz_jet", &pz_jet, "pz_jet/D");

   myTree->Branch("nEcalHits",&nEcalHits,"nEcalHits/I");
   myTree->Branch("ecalHit_energy", &ecalHit_energy, "ecalHit_energy[nEcalHits]/D");
   myTree->Branch("ecalHit_eta", &ecalHit_eta, "ecalHit_eta[nEcalHits]/D");
   myTree->Branch("ecalHit_phi", &ecalHit_phi, "ecalHit_phi[nEcalHits]/D");

 
   myTree->Branch("nHBHEHits",&nHBHEHits,"nHBHEHits/I");
   myTree->Branch("hbheHit_energy", &hbheHit_energy, "hbheHit_energy[nHBHEHits]/D");
   myTree->Branch("hbheHit_eta", &hbheHit_eta, "hbheHit_eta[nHBHEHits]/D");
   myTree->Branch("hbheHit_phi", &hbheHit_phi, "hbheHit_phi[nHBHEHits]/D");

   myTree->Branch("nHOHits",&nHOHits,"nHOHits/I");
   myTree->Branch("hoHit_energy", &hoHit_energy, "hoHit_energy[nHOHits]/D");
   myTree->Branch("hoHit_eta", &hoHit_eta, "hoHit_eta[nHOHits]/D");
   myTree->Branch("hoHit_phi", &hoHit_phi, "hoHit_phi[nHOHits]/D");

   myTree->Branch("nHFHits",&nHFHits,"nHFHits/I");
   myTree->Branch("hfHit_energy", &hfHit_energy, "hfHit_energy[nHFHits]/D");
   myTree->Branch("hfHit_eta", &hfHit_eta, "hfHit_eta[nHFHits]/D");
   myTree->Branch("hfHit_phi", &hfHit_phi, "hfHit_phi[nHFHits]/D");

   edm::ESHandle<CaloGeometry> pG;
   iSetup.get<IdealGeometryRecord>().get(pG);
   geo = pG.product();

}

void DiJetAnalyzer::endJob()
{
   myTree -> Write();
   hOutputFile->Close() ;
}

void
DiJetAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm; 

   double pi = 4.*atan(1.);

   typedef reco::CaloJetCollection CaloCollection;
   Handle<CaloCollection> jets_calo;
   iEvent.getByLabel(jetsLabel_,jets_calo);
   CaloCollection::const_iterator it_1; 
  for(it_1 = jets_calo->begin(); it_1 != jets_calo->end(); it_1++)
  {
   eta_jet = it_1->eta(); 
   phi_jet = it_1->phi(); 
   px_jet  = it_1->px(); 
   py_jet  = it_1->py(); 
   pz_jet  = it_1->pz(); 

   //   std::cout << "eta_jet=" << eta_jet << " phi_jet=" << phi_jet 
   //     <<" pt_jet=" << it_1->pt() << std::endl;

   nEcalHits = 0; 
   std::vector<edm::InputTag>::const_iterator it_2;
   for (it_2=ecalLabels_.begin(); it_2!=ecalLabels_.end(); it_2++) {
   Handle<EcalRecHitCollection> ec;
   iEvent.getByLabel(*it_2,ec);
   for(EcalRecHitCollection::const_iterator recHit = (*ec).begin();
                                       recHit != (*ec).end(); ++recHit)
   {
      GlobalPoint pos = geo->getPosition(recHit->detid());
      double dphi = fabs(pos.phi()-phi_jet);
      if(dphi>pi){dphi = 2*pi-dphi;}
      double deta = fabs(pos.eta()-eta_jet);
      double dr = sqrt(dphi*dphi+deta*deta); 
      if(dr<1.4){
        ecalHit_eta[nEcalHits] = pos.eta(); 
        ecalHit_phi[nEcalHits] = pos.phi(); 
        ecalHit_energy[nEcalHits] = recHit->energy(); 
        nEcalHits++; 
      }
      
   }
   }
   //   std::cout << "nEcalHits=" << nEcalHits << std::endl;


   nHBHEHits = 0; 
   Handle<HBHERecHitCollection> hbhe;
   iEvent.getByLabel(hbheLabel_,hbhe);
   const HBHERecHitCollection Hithbhe = *(hbhe.product());
   for(HBHERecHitCollection::const_iterator hbheItr=Hithbhe.begin(); hbheItr!=Hithbhe.end(); hbheItr++)       
   {
      GlobalPoint pos = geo->getPosition(hbheItr->detid());
      double dphi = fabs(pos.phi()-phi_jet);
      if(dphi>pi){dphi = 2*pi-dphi;}
      double deta = fabs(pos.eta()-eta_jet);
      double dr = sqrt(dphi*dphi+deta*deta); 
      if(dr<1.4){
        hbheHit_eta[nHBHEHits] = pos.eta(); 
        ecalHit_phi[nHBHEHits] = pos.phi(); 
        ecalHit_energy[nHBHEHits] = hbheItr->energy(); 
        nHBHEHits++; 
      }
   }

   nHOHits = 0; 
   Handle<HORecHitCollection> ho;
   iEvent.getByLabel(hoLabel_,ho);
   const HORecHitCollection Hitho = *(ho.product());
   //cout<<" Size of HO collection "<<Hitho.size()<<endl;
   for(HORecHitCollection::const_iterator hoItr=Hitho.begin(); hoItr!=Hitho.end(); hoItr++)
   {
      GlobalPoint pos = geo->getPosition(hoItr->detid());
      double dphi = fabs(pos.phi()-phi_jet);
      if(dphi>pi){dphi = 2*pi-dphi;}
      double deta = fabs(pos.eta()-eta_jet);
      double dr = sqrt(dphi*dphi+deta*deta); 
      if(dr<1.4){
        hoHit_eta[nHOHits] = pos.eta(); 
        hoHit_phi[nHOHits] = pos.phi(); 
        hoHit_energy[nHOHits] = hoItr->energy(); 
        nHOHits++; 
      }
   }

   nHFHits = 0; 
   Handle<HFRecHitCollection> hf;
   iEvent.getByLabel(hfLabel_,hf);
   const HFRecHitCollection Hithf = *(hf.product());
//  cout<<" Size of HF collection "<<Hithf.size()<<endl;
   for(HFRecHitCollection::const_iterator hfItr=Hithf.begin(); hfItr!=Hithf.end(); hfItr++)
   {
      GlobalPoint pos = geo->getPosition(hfItr->detid());
      double dphi = fabs(pos.phi()-phi_jet);
      if(dphi>pi){dphi = 2*pi-dphi;}
      double deta = fabs(pos.eta()-eta_jet);
      double dr = sqrt(dphi*dphi+deta*deta); 
      if(dr<1.4){
        hfHit_eta[nHFHits] = pos.eta(); 
        hfHit_phi[nHFHits] = pos.phi(); 
        hfHit_energy[nHFHits] = hfItr->energy(); 
        nHFHits++; 
      }
   }
   myTree -> Fill();

   }
}

} // namespace cms
