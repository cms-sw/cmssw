// -*- C++ -*-
//
// Package:  HcalCalibAlgos  
// Class:    HcalIsoTrkAnalyzer 

// 
/**\class HcalIsoTrkAnalyzer HcalIsoTrkAnalyzer.cc Calibration/HcalCalibAlgos/src/HcalIsoTrkAnalyzer.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Andrey Pozdnyakov
//         Created:  Thu Jul 12 18:12:19 CEST 2007
// $Id$
//
//

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"

//#include "DataFormats/Common/interface/Ref.h"
//#include "DataFormats/CaloTowers/interface/CaloTowerCollection.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
//#include "DataFormats/CaloTowers/interface/CaloTowerDetId.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
//#include "RecoTracker/TrackProducer/interface/TrackProducerBase.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloEventSetup/interface/CaloTopologyRecord.h"
#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "TROOT.h"
#include "TH1.h"
#include "TFile.h"
#include "TTree.h"

using namespace edm;
using namespace std;
using namespace reco;

//
// class decleration
//

class HcalIsoTrkAnalyzer : public edm::EDAnalyzer {
public:
  explicit HcalIsoTrkAnalyzer(const edm::ParameterSet&);
  ~HcalIsoTrkAnalyzer();
  
private:
  virtual void beginJob(const edm::EventSetup&) ;
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void endJob() ;
  
  // ----------member data ---------------------------
  
  const CaloGeometry* geo;
  InputTag hbheLabel_;
  InputTag hoLabel_;
  InputTag eLabel_;
  InputTag trackLabel_;
  double associationConeSize_;
  bool allowMissingInputs_;
  string outputFileName_;

  double trackPt, trackEta, trackPhi; 
  int nHCRecHits;
  int nECRecHits;
  int nHORecHits;
  int iEventNumber;
  int iRunNumber;
  
  
  double ecRHen[500];
  double ecRHeta[500];
  double ecRHphi[500];
  
  double hcRHen[500];
  double hcRHeta[500];
  double hcRHphi[500];
  int hcRHieta[500];
  int hcRHiphi[500];
  int hcRHdepth[500];
  
  double hoRHen[500];
  double hoRHeta[500];
  double hoRHphi[500];
  int hoRHieta[500];
  int hoRHiphi[500];
  int hoRHdepth[500];


  TFile* tf2;
  
  TH1F* thPtIso; 
  TH1F* thEtaIso; 
  TH1F* thPhiIso; 
  TH1F* thDrTrEHits;
  TH1F* thDrTrHBHEHits;
  TH1F* thDrTrHOHits;
  TH1F* thERecHitEn; 
  TH1F* thHBHERecHitEn; 
  TH1F* thHORecHitEn;
  
  TTree* calibTr;


};

//
// constants, enums and typedefs
//


//
// static data member definitions
//

//
// constructors and destructor
//
HcalIsoTrkAnalyzer::HcalIsoTrkAnalyzer(const edm::ParameterSet& iConfig)
{
   //now do what ever initialization is needed

  hbheLabel_= iConfig.getParameter<edm::InputTag>("hbheInput");
  hoLabel_=iConfig.getParameter<edm::InputTag>("hoInput");
  eLabel_=iConfig.getParameter<edm::InputTag>("eInput");
  trackLabel_ = iConfig.getParameter<edm::InputTag>("trackInput");
  associationConeSize_=iConfig.getParameter<double>("associationConeSize");
  allowMissingInputs_=iConfig.getParameter<bool>("allowMissingInputs");
  outputFileName_=iConfig.getParameter<std::string>("outputFileName");
}

HcalIsoTrkAnalyzer::~HcalIsoTrkAnalyzer()
{
  tf2 -> Close();


   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}

//
// member functions
//

// ------------ method called to for each event  ------------
void
HcalIsoTrkAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;

   Handle<TrackCollection> isoTrCol;
   iEvent.getByLabel(trackLabel_,isoTrCol);

   ESHandle<CaloGeometry> pG;
   iSetup.get<IdealGeometryRecord>().get(pG);
   geo = pG.product();

   double IsoPt;
   double IsoEta = 0.;
   double IsoPhi = 0.;

for (TrackCollection::const_iterator it = isoTrCol->begin(); it!=isoTrCol->end(); it++)
  {
    
    IsoPt  = it->pt();
    IsoEta = it->eta();
    IsoPhi = it->phi();
   
    thPtIso  -> Fill(IsoPt);
    thEtaIso -> Fill(IsoEta);
    thPhiIso -> Fill(IsoPhi);
    
    trackPt = IsoPt;
    trackEta = IsoEta;
    trackPhi = IsoPhi;
    
    nHCRecHits=0;
    nECRecHits=0;
    nHORecHits=0;

   try {
     
     Handle<EcalRecHitCollection> ecal;
     iEvent.getByLabel(eLabel_,ecal);
     for(EcalRecHitCollection::const_iterator eItr=ecal->begin(); eItr!=ecal->end(); eItr++)
       {
	 GlobalPoint pos = geo->getPosition(eItr->detid());
	 double phihit = pos.phi();
	 double etahit = pos.eta();
	 
	 double dphi = fabs(IsoPhi - phihit); 
	 if(dphi > 4.*atan(1.)) dphi = 8.*atan(1.) - dphi;
	 double deta = fabs(IsoEta - etahit); 
	 double dr = sqrt(dphi*dphi + deta*deta);
	 
	 thDrTrEHits -> Fill(dr);
	 thERecHitEn -> Fill(eItr->energy());

	 if (dr<associationConeSize_){	 
	   ecRHen[nECRecHits] = eItr->energy();
	   ecRHeta[nECRecHits] = etahit;
	   ecRHphi[nECRecHits] = phihit;
	   nECRecHits++;
	 }
       }
     
     
   } 
   catch (cms::Exception& e) 
     { // can't find it!
       if (!allowMissingInputs_) throw e;
     }
   
   //   cout<<" Ecal is done "<<endl;
   
   try {
     
     edm::Handle<HBHERecHitCollection> hbhe;
     iEvent.getByLabel(hbheLabel_,hbhe);
     const HBHERecHitCollection Hithbhe = *(hbhe.product());
     for(HBHERecHitCollection::const_iterator hbheItr=Hithbhe.begin(); hbheItr!=Hithbhe.end(); hbheItr++)
       {
	 GlobalPoint pos = geo->getPosition(hbheItr->detid());
	 double phihit = pos.phi();
	 double etahit = pos.eta();
	 
	 int iphihit = (hbheItr->id()).iphi();
	 int ietahit = (hbheItr->id()).ieta();
	 int depthhit = (hbheItr->id()).depth();

	 // LogInfo("iCoord: ")<<ietahit<<"   "<<iphihit<<"       "<< depthhit;
		 
	 double dphi = fabs(IsoPhi - phihit); 
	 if(dphi > 4.*atan(1.)) dphi = 8.*atan(1.) - dphi;
	 double deta = fabs(IsoEta - etahit); 
	 double dr = sqrt(dphi*dphi + deta*deta);
	 
	 thDrTrHBHEHits -> Fill(dr);
	 thHBHERecHitEn -> Fill(hbheItr->energy());
	 if(dr<associationConeSize_){
	   hcRHen[nHCRecHits] = hbheItr->energy();
	   hcRHeta[nHCRecHits] = etahit;
	   hcRHphi[nHCRecHits] = phihit;
	   hcRHieta[nHCRecHits] = ietahit;
	   hcRHiphi[nHCRecHits] = iphihit;
	   hcRHdepth[nHCRecHits] = depthhit;
	   nHCRecHits++;
	 }
	 
       }
   } 
   catch (cms::Exception& e) 
     { // can't find it!
       if (!allowMissingInputs_) throw e;
     }
   
   //   cout<<" HBHE is done "<<endl;
   
   
   try {
     Handle<HORecHitCollection> ho;
     iEvent.getByLabel(hoLabel_,ho);
     const HORecHitCollection Hitho = *(ho.product());
     for(HORecHitCollection::const_iterator hoItr=Hitho.begin(); hoItr!=Hitho.end(); hoItr++)
       {
	 GlobalPoint pos = geo->getPosition(hoItr->detid());
	 double phihit = pos.phi();
	 double etahit = pos.eta();

	 int iphihit = (hoItr->id()).iphi();
	 int ietahit = (hoItr->id()).ieta();
	 int depthhit = (hoItr->id()).depth();

	 double dphi = fabs(IsoPhi - phihit); 
	 if(dphi > 4.*atan(1.)) dphi = 8.*atan(1.) - dphi;
	 double deta = fabs(IsoEta - etahit); 
	 double dr = sqrt(dphi*dphi + deta*deta);
	 
	 thDrTrHOHits -> Fill(dr);
	 thHORecHitEn -> Fill(hoItr->energy()); 
	 if(dr<associationConeSize_) {
	   hoRHen[nHORecHits] = hoItr->energy();
	   hoRHeta[nHORecHits] = etahit;
	   hoRHphi[nHORecHits] = phihit;
	   hoRHieta[nHORecHits] = ietahit;
	   hoRHiphi[nHORecHits] = iphihit;
	   hoRHdepth[nHORecHits] = depthhit;
	   nHORecHits++;
	 }
  
       }
   } catch (cms::Exception& e) { // can't find it!
     if (!allowMissingInputs_) throw e;
   }
   //  cout<<" HO is done "<<endl; 
   
   calibTr -> Fill();
  }
}

// ------------ method called once each job just before starting event loop  ------------
void 
HcalIsoTrkAnalyzer::beginJob(const edm::EventSetup&)
{
  
  tf2 = new TFile(outputFileName_.c_str(),"RECREATE");
  
  thPtIso  = new TH1F("thPtIso","thPtIso", 100, 0, 100);
  thEtaIso = new TH1F("thEtaIso","thEtaIso", 50, -4., 4.);  
  thPhiIso = new TH1F("thPhiIso","thPhiIso", 50, -4., 4.);

  thDrTrEHits = new TH1F("thDrTrEHits","thDrETrHits", 50, 0., 5.);
  thDrTrHBHEHits = new TH1F("thDrHBHETrHits","thDrHBHETrHits", 50, 0., 5.);
  thDrTrHOHits = new TH1F("thDrHOTrHits","thDrHOTrHits", 50, 0., 5.);

  thERecHitEn = new TH1F("thERecHitEn","thERecHitEn", 50, 0., 10.);
  thHBHERecHitEn = new TH1F("thHBHERecHitEn","thHBHERecHitEn", 50, 0., 10.);
  thHORecHitEn = new TH1F("thHORecHitEn","thHORecHitEn", 50, 0., 10.);


  calibTr = new TTree("calibTr","calibTr");
  
  calibTr->Branch("trackEta",&trackEta,"trackEta/D");
  calibTr->Branch("trackPt",&trackPt,"trackPt/D");
  calibTr->Branch("iEventNamber",&iEventNumber,"iEventNumber/I");
  calibTr->Branch("iRunNumber",&iRunNumber,"iRunNumber/I");

  calibTr->Branch("nECRecHits",&nECRecHits,"nECRecHits/I");
  calibTr->Branch("ecRHen",ecRHen,"ecRHen[nECRecHits]/D");
  calibTr->Branch("ecRHeta",ecRHeta,"ecRHeta[nECRecHits]/D");
  calibTr->Branch("ecRHphi",ecRHphi,"ecRHphi[nECRecHits]/D");
 
  calibTr->Branch("nHCRecHits",&nHCRecHits,"nHCRecHits/I");
  calibTr->Branch("hcRHen",hcRHen,"hcRHen[nHCRecHits]/D");
  calibTr->Branch("hcRHeta",hcRHeta,"hcRHeta[nHCRecHits]/D");
  calibTr->Branch("hcRHphi",hcRHphi,"hcRHphi[nHCRecHits]/D");
  calibTr->Branch("hcRHieta",hcRHieta,"hcRHieta[nHCRecHits]/I");
  calibTr->Branch("hcRHiphi",hcRHiphi,"hcRHiphi[nHCRecHits]/I");
  calibTr->Branch("hcRHdepth",hcRHdepth,"hcRHdepth[nHCRecHits]/I");
  
  calibTr->Branch("nHORecHits",&nHORecHits,"nHORecHits/I");
  calibTr->Branch("hoRHen",hoRHen,"hoRHen[nHORecHits]/D");
  calibTr->Branch("hoRHeta",hoRHeta,"hoRHeta[nHORecHits]/D");
  calibTr->Branch("hoRHphi",hoRHphi,"hoRHphi[nHORecHits]/D");
  calibTr->Branch("hoRHieta",hoRHieta,"hoRHieta[nHORecHits]/I");
  calibTr->Branch("hoRHiphi",hoRHiphi,"hoRHiphi[nHORecHits]/I");
  calibTr->Branch("hoRHdepth",hoRHdepth,"hoRHdepth[nHORecHits]/I");
  
}

// ------------ method called once each job just after ending the event loop  ------------
void 
HcalIsoTrkAnalyzer::endJob() {
  
  tf2 -> cd();

  thPtIso  -> Write();
  thEtaIso -> Write();
  thPhiIso -> Write();

  thDrTrEHits -> Write();  
  thDrTrHBHEHits -> Write();
  thDrTrHOHits -> Write(); 

  thERecHitEn -> Write();  
  thHBHERecHitEn -> Write();
  thHORecHitEn -> Write();

  calibTr -> Write();
  
}

//define this as a plug-in
DEFINE_FWK_MODULE(HcalIsoTrkAnalyzer);
