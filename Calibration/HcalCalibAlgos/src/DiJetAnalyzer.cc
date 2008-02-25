#include "Calibration/HcalCalibAlgos/interface/DiJetAnalyzer.h"

#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"

#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/CaloTowers/interface/CaloTowerCollection.h"
#include "DataFormats/CaloTowers/interface/CaloTowerDetId.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/DetId/interface/DetId.h"

#include "Calibration/Tools/interface/GenericMinL3Algorithm.h"

#include "FWCore/Utilities/interface/Exception.h"
//#include "DataFormats/JetReco/interface/CaloJet.h"
namespace cms
{
DiJetAnalyzer::DiJetAnalyzer(const edm::ParameterSet& iConfig)
{
  jet_product = iConfig.getUntrackedParameter<std::string>("jetsProduct","DiJProd");
  jets_ = iConfig.getUntrackedParameter<std::string>("jetsInput","DiJetsBackToBackCollection");
  ec_=iConfig.getUntrackedParameter<std::string>("ecInput","DiJetsEcalRecHitCollection");
  hbhe_=  iConfig.getUntrackedParameter<std::string>("hbheInput","DiJetsHBHERecHitCollection");
  hf_=    iConfig.getUntrackedParameter<std::string>("hfInput","DiJetsHFRecHitCollection");
  ho_ = iConfig.getUntrackedParameter<std::string>("hoInput","DiJetsHORecHitCollection");
  eta_1 = iConfig.getParameter<double>("eta_1");
  eta_2 = iConfig.getParameter<double>("eta_2");
  jet_R = iConfig.getParameter<double>("jet_R");
  et_threshold = iConfig.getParameter<double>("et_threshold");
  et_veto = iConfig.getParameter<double>("et_veto"); 
  m_histoFlag = iConfig.getUntrackedParameter<int>("histoFlag",0);


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
   nHits = 0; 
   using namespace edm;
   using namespace reco; 

   const double pi = 4.*atan(1.);

   CaloJet jet1, jet2, jet3; 
   try {
   Handle<CaloJetCollection> jets;
   iEvent.getByLabel(jet_product, jets_, jets);
   if(jets->size()>1){ 
     jet1 = (*jets)[0]; 
     jet2 = (*jets)[1];
     if(fabs(jet1.eta())>fabs(jet2.eta())){
       CaloJet jet = jet1; 
       jet1 = jet2; 
       jet2 = jet; 
     } 
     if(fabs(jet1.eta())>eta_1||fabs(jet2.eta())<eta_2){ return;}
     if(jets->size()==3){jet3 = (*jets)[2];}
   } else {return;}

   }catch (cms::Exception& e) { // can't find it!
     if (!allowMissingInputs_) { throw e; }  
   }

   if(jet3.et()>et_veto || jet1.et()<et_threshold){return;}

     et_jet_centr = jet1.et(); 
     eta_jet_centr = jet1.eta(); 
     phi_jet_centr = jet1.phi();

     et_jet_forw = jet2.et(); 
     eta_jet_forw = jet2.eta(); 
     phi_jet_forw = jet2.phi();

   std::map<DetId,float> DetEMap;

   float e_em_in_forw_jet = 0; 
   try {
      Handle<EcalRecHitCollection> ec;
      iEvent.getByLabel(jet_product, ec_,ec);
      for(EcalRecHitCollection::const_iterator ecItr = (*ec).begin();
                                                ecItr != (*ec).end(); ++ecItr)
      {
       DetId id = ecItr->detid();
       GlobalPoint pos = geo->getPosition(id);
       double etahit = pos.eta();
       double phihit = pos.phi();
       double deta = fabs(jet2.eta()-etahit);
       double dphi = fabs(jet2.phi()-phihit);
       if(dphi>pi){dphi = 2*pi-dphi;}
       double dR = sqrt(pow(deta,2)+pow(dphi,2));
       if(dR<jet_R){DetEMap[id] = ecItr->energy(); e_em_in_forw_jet += ecItr->energy();
       hitEta[nHits]=etahit; hitPhi[nHits]=phihit; nHits++;}
      }

   } catch (cms::Exception& e) { // can't find it!
      if (!allowMissingInputs_) throw e;
   }
  

    float EJet = jet1.et()/sin(jet2.theta()) - e_em_in_forw_jet; 
    energyVector.push_back(EJet);        

   try {
      Handle<HBHERecHitCollection> hbhe;
      iEvent.getByLabel(jet_product, hbhe_, hbhe);
      for(HBHERecHitCollection::const_iterator hbheItr=hbhe->begin(); 
                                                 hbheItr!=hbhe->end(); hbheItr++)
      {
       DetId id = hbheItr->detid(); 
       GlobalPoint pos = geo->getPosition(id);
       double etahit = pos.eta();
       double phihit = pos.phi();
       double deta = fabs(jet2.eta()-etahit); 
       double dphi = fabs(jet2.phi()-phihit); 
       if(dphi>pi){dphi = 2*pi-dphi;}
       double dR = sqrt(pow(deta,2)+pow(dphi,2)); 
       if(dR<jet_R){DetEMap[id] = hbheItr->energy(); NDetEntries[id]++;
       hitEta[nHits]=etahit; hitPhi[nHits]=phihit; nHits++;}
      }

   } catch (cms::Exception& e) { // can't find it!
      if (!allowMissingInputs_) throw e;
   }

   try {
      Handle<HORecHitCollection> ho;
      iEvent.getByLabel(jet_product, ho_, ho);
      for(HORecHitCollection::const_iterator hoItr=ho->begin(); 
                                               hoItr!=ho->end(); hoItr++)
      {
       DetId id = hoItr->detid();
       GlobalPoint pos = geo->getPosition(id);
       double etahit = pos.eta();
       double phihit = pos.phi();
       double deta = fabs(jet2.eta()-etahit);
       double dphi = fabs(jet2.phi()-phihit);
       if(dphi>pi){dphi = 2*pi-dphi;}
       double dR = sqrt(pow(deta,2)+pow(dphi,2));
       if(dR<jet_R){DetEMap[id] = hoItr->energy(); NDetEntries[id]++;
       hitEta[nHits]=etahit; hitPhi[nHits]=phihit; nHits;}
      }

   } catch (cms::Exception& e) { // can't find it!
      if (!allowMissingInputs_) {std::cout<<" No HO collection "<<std::endl; throw e;}
   }



   try {
      Handle<HFRecHitCollection> hf;
      iEvent.getByLabel(jet_product, hf_, hf);
      for(HFRecHitCollection::const_iterator hfItr=hf->begin(); 
                                               hfItr!=hf->end(); hfItr++)
      {
       DetId id = hfItr->detid();
       GlobalPoint pos = geo->getPosition(id);
       double etahit = pos.eta();
       double phihit = pos.phi();
       double deta = fabs(jet2.eta()-etahit);
       double dphi = fabs(jet2.phi()-phihit);
       if(dphi>pi){dphi = 2*pi-dphi;}
       double dR = sqrt(pow(deta,2)+pow(dphi,2));
       if(dR<jet_R){DetEMap[id] = hfItr->energy(); NDetEntries[id]++;
       hitEta[nHits]=etahit; hitPhi[nHits]=phihit; nHits++;}
      }

   } catch (cms::Exception& e) { // can't find it!
      if (!allowMissingInputs_) {std::cout<<" No HF collection "<<std::endl; throw e;}
   }



   std::vector<float> en; 
   for(std::vector<DetId>::const_iterator id=did_selected.begin(); id != did_selected.end(); id++)
   {
        en.push_back(DetEMap[(*id)]); 
   }
   eventMatrix.push_back(en); 
   
  if(m_histoFlag==1){
   myTree->Fill();
  }

}


// ------------ method called once each job just before starting event loop  ------------
void 
DiJetAnalyzer::beginJob(const edm::EventSetup& iSetup)
{

  hOutputFile   = new TFile( fOutputFileName.c_str(), "RECREATE" ) ;

   myTree = new TTree("RecJet","RecJet Tree");


   myTree->Branch("et_jet_centr",  &et_jet_centr, "et_jet_centr/F");
   myTree->Branch("eta_jet_centr",  &eta_jet_centr, "eta_jet_centr/F");
   myTree->Branch("phi_jet_centr",  &phi_jet_centr, "phi_jet_centr/F");
   myTree->Branch("et_jet_forw",  &et_jet_forw, "et_jet_forw/F");
   myTree->Branch("eta_jet_forw",  &eta_jet_forw, "eta_jet_forw/F");
   myTree->Branch("phi_jet_forw",  &phi_jet_forw, "phi_jet_forw/F");

   myTree->Branch("nHits", &nHits, "nHits/I");
   myTree->Branch("hitEta",  hitEta, "hitEta[nHits]/F");
   myTree->Branch("hitPhi",  hitPhi, "hitPhi[nHits]/F");


   edm::ESHandle<CaloGeometry> pG;
   iSetup.get<IdealGeometryRecord>().get(pG);
   geo = pG.product();


   std::vector<DetId> did = geo->getValidDetIds();
   for(std::vector<DetId>::const_iterator id=did.begin(); id != did.end(); id++)
   {
      if( (*id).det() == DetId::Hcal ) {
        GlobalPoint pos = geo->getPosition(*id); 
        double eta = pos.eta();
        if(fabs(eta)>eta_2){did_selected.push_back(*id);}
      }
   }

}

// ------------ method called once each job just after ending the event loop  ------------
void 
DiJetAnalyzer::endJob() {

   nIter = 10; 
   GenericMinL3Algorithm* minL3 = new GenericMinL3Algorithm(); 
      for(int j=0; j<eventMatrix.size(); j++){
        vector<float> h = eventMatrix[j]; 
      } 
   vector<float> calib = minL3->iterate(eventMatrix,energyVector,nIter);         
   for(int i=0; i<calib.size(); i++){
       DetId id = did_selected[i];  
       int i_eta = HcalDetId(id).ieta();
       int i_phi = HcalDetId(id).iphi();
       int i_depth = HcalDetId(id).depth();
       int mydet = (id.rawId()>>28)&0xF;
       int mysubd = (id.rawId()>>25)&0x7;

       std::cout << "ieta=" << i_eta << " iphi=" << i_phi << " idepth=" << i_depth << 
       " mydet=" << mydet << " mysubd=" << mysubd << " calib=" << calib[i] << "NEntries=" 
       << NDetEntries[id] << std::endl;
   }

   hOutputFile->SetCompressionLevel(2);
   hOutputFile->cd();
   myTree->Write();
   hOutputFile->Close() ;

}
}
