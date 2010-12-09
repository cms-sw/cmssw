#include "Calibration/HcalCalibAlgos/interface/DiJetAnalyzer.h"

#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"

#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/CaloTowers/interface/CaloTowerCollection.h"
#include "DataFormats/CaloTowers/interface/CaloTowerDetId.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/DetId/interface/DetId.h"

#include "Calibration/Tools/interface/GenericMinL3Algorithm.h"
#include "CondFormats/DataRecord/interface/HcalRespCorrsRcd.h"
#include "CalibCalorimetry/HcalAlgos/interface/HcalDbASCIIIO.h"

#include "FWCore/Utilities/interface/Exception.h"
#include <fstream>



#include "TString.h"
#include "TFile.h"
#include "TTree.h"
#include "TObject.h"
#include "TObjArray.h"
#include "TClonesArray.h"
#include "TRefArray.h"
#include "TLorentzVector.h"



namespace cms
{
DiJetAnalyzer::DiJetAnalyzer(const edm::ParameterSet& iConfig)
{
  jets_=iConfig.getParameter<edm::InputTag>("jetsInput");
  ec_=iConfig.getParameter<edm::InputTag>("ecInput");
  hbhe_=iConfig.getParameter<edm::InputTag>("hbheInput");
  ho_=iConfig.getParameter<edm::InputTag>("hoInput");
  hf_=iConfig.getParameter<edm::InputTag>("hfInput");

  // get name of output file with histogramms
  fOutputFileName = iConfig.getUntrackedParameter<std::string>("HistOutFile");

   allowMissingInputs_ = true;
}


DiJetAnalyzer::~DiJetAnalyzer()
{

}


//
// member functions
//

// ------------ method called to for each event  ------------
void
DiJetAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{

   using namespace std; 
   using namespace edm;
   using namespace reco; 

   const double pi = 4.*atan(1.);

  // event number & run number 
   eventNumber = iEvent.id().event(); 
   runNumber = iEvent.id().run();
 

  // read jets
   CaloJet jet1, jet2, jet3; 
   try {
   edm::Handle<CaloJetCollection> jets;
   iEvent.getByLabel(jets_,jets);
   if(jets->size()>1){ 
    jet1 = (*jets)[0]; 
    jet2 = (*jets)[1];
     if(fabs(jet1.eta())>fabs(jet2.eta())){
       CaloJet jet = jet1; 
       jet1 = jet2; 
       jet2 = jet; 
     } 
     //     if(fabs(jet1.eta())>eta_1 || (fabs(jet2.eta())-jet_R) < eta_2){ return;}
   } else {return;}
   tagJetP4->SetPxPyPzE(jet1.px(), jet1.py(), jet1.pz(), jet1.energy());
   probeJetP4->SetPxPyPzE(jet2.px(), jet2.py(), jet2.pz(), jet2.energy());
   if(jets->size()>2){
     jet3 = (*jets)[2];
     etVetoJet = jet3.et();
   } else { etVetoJet = 0.;}
   }catch (cms::Exception& e) { // can't find it!
     if (!allowMissingInputs_) { throw e; }  
   }


   double dR = 1000.; 
   edm::ESHandle<CaloGeometry> pG;
   iSetup.get<CaloGeometryRecord>().get(pG);
   const CaloGeometry* geo = pG.product();
   vector<DetId> vid = geo->getValidDetIds();
   for(vector<DetId>::const_iterator idItr = vid.begin(); idItr != vid.end(); idItr++)
     {
      if( (*idItr).det() == DetId::Hcal ) {
        GlobalPoint pos = geo->getPosition(*idItr); 
        double deta = fabs(jet2.eta() - pos.eta());
        double dphi = fabs(jet2.phi() - pos.phi()); 
        if(dphi>pi){dphi=2*pi-dphi;}
        double dR_candidate = sqrt(deta*deta + dphi*dphi); 
        int ieta = HcalDetId(*idItr).ieta();
        int iphi = HcalDetId(*idItr).iphi();
        int idepth = HcalDetId(*idItr).depth();
        if(dR_candidate < dR && idepth == 1){
	  dR = dR_candidate; 
          iEtaHit = ieta; 
          iPhiHit = iphi; 
	}
      }
     }

   targetE = jet1.et()/sin(jet2.theta()); 

   std::map<DetId,int> mapId; 
   vector<CaloTowerPtr> towers_fw = jet2.getCaloConstituents();
   vector<CaloTowerPtr>::const_iterator towersItr;
   for(towersItr = towers_fw.begin(); towersItr != towers_fw.end(); towersItr++){
      size_t tower_size = (*towersItr)->constituentsSize();
      for(size_t i=0; i<tower_size; i++){
        DetId id = (*towersItr)->constituent(i);
        mapId[id] = 1;
      }
   }


   //   probeJetEmFrac = 0.;
   double emEnergy = 0.; 
   try {
      Handle<EcalRecHitCollection> ec;
      iEvent.getByLabel(ec_,ec);
      for(EcalRecHitCollection::const_iterator ecItr = (*ec).begin();
                                                ecItr != (*ec).end(); ++ecItr)
      {
        DetId id = ecItr->detid();
        if(mapId[id]==1){
	  emEnergy += ecItr->energy(); 
	}
      }
   } catch (cms::Exception& e) { // can't find it!
      if (!allowMissingInputs_) throw e;
   }

   targetE += -emEnergy;
   probeJetEmFrac = emEnergy/jet2.energy(); 

   int nHits = 0; 
   try {
      Handle<HBHERecHitCollection> hbhe;
      iEvent.getByLabel(hbhe_, hbhe);
      for(HBHERecHitCollection::const_iterator hbheItr=hbhe->begin(); 
                                                 hbheItr!=hbhe->end(); hbheItr++)
      {
         DetId id = hbheItr->detid();     
         if(mapId[id]==1){
          TCell* cell = new TCell(id.rawId(),hbheItr->energy()); 
          (*cells)[nHits] = cell;
          nHits++;  
         }       
      }
   } catch (cms::Exception& e) { // can't find it!
      if (!allowMissingInputs_) throw e;
   }
 

   try {
      Handle<HORecHitCollection> ho;
      iEvent.getByLabel(ho_, ho);
      for(HORecHitCollection::const_iterator hoItr=ho->begin(); 
                                               hoItr!=ho->end(); hoItr++)
      {
         DetId id = hoItr->detid();
         if(mapId[id]==1){
          TCell* cell = new TCell(id.rawId(),hoItr->energy()); 
          (*cells)[nHits] = cell;
          nHits++;  
         }
      }
   } catch (cms::Exception& e) { // can't find it!
      if (!allowMissingInputs_) throw e;
   }

   try {
      Handle<HFRecHitCollection> hf;
      iEvent.getByLabel(hf_, hf);
      for(HFRecHitCollection::const_iterator hfItr=hf->begin(); 
                                               hfItr!=hf->end(); hfItr++)
      {
         DetId id = hfItr->detid();
         if(mapId[id]==1){
          TCell* cell = new TCell(id.rawId(),hfItr->energy()); 
          (*cells)[nHits] = cell; 
          nHits++; 
         }
      }
   } catch (cms::Exception& e) { // can't find it!
      if (!allowMissingInputs_) throw e;
   }
 
   probeJetEmFrac = emEnergy/jet2.energy();
   
   tree->Fill(); 
 
}


// ------------ method called once each job just before starting event loop  ------------
void 
DiJetAnalyzer::beginJob()
{

  hOutputFile   = new TFile( fOutputFileName.c_str(), "RECREATE" ) ;

    tree = new TTree("hcalCalibTree", "Tree for IsoTrack Calibration");

    cells = new TClonesArray("TCell");
    tagJetP4 = new TLorentzVector();
    probeJetP4 = new TLorentzVector();
 

    tree->Branch("eventNumber", &eventNumber, "eventNumber/i");
    tree->Branch("runNumber", &runNumber, "runNumber/i");  
    tree->Branch("iEtaHit", &iEtaHit, "iEtaHit/I");
    tree->Branch("iPhiHit", &iPhiHit, "iPhiHit/i");    

    tree->Branch("xTrkEcal", &xTrkEcal, "xTrkEcal/F");
    tree->Branch("yTrkEcal", &yTrkEcal, "yTrkEcal/F");
    tree->Branch("zTrkEcal", &zTrkEcal, "zTrkEcal/F");
    tree->Branch("xTrkHcal", &xTrkHcal, "xTrkHcal/F");
    tree->Branch("yTrkHcal", &yTrkHcal, "yTrkHcal/F");
    tree->Branch("zTrkHcal", &zTrkHcal, "zTrkHcal/F");

    tree->Branch("cells", &cells, 64000); 
    tree->Branch("emEnergy", &emEnergy, "emEnergy/F"); 
    tree->Branch("targetE", &targetE, "targetE/F");
    tree->Branch("etVetoJet", &etVetoJet, "etVetoJet/F");
    tree->Branch("tagJetP4", "TLorentzVector", &tagJetP4);
    tree->Branch("probeJetP4", "TLorentzVector", &probeJetP4);
    tree->Branch("tagJetEmFrac", &tagJetEmFrac,"tagJetEmFrac/F"); 
    tree->Branch("probeJetEmFrac", &probeJetEmFrac,"probeJetEmFrac/F");    
   
}

// ------------ method called once each job just after ending the event loop  ------------
void 
DiJetAnalyzer::endJob() {

   hOutputFile->SetCompressionLevel(2);
   hOutputFile->cd();
   tree->Write(); 
   hOutputFile->Close() ;

}
}
