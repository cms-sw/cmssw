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
// Original Authors: Andrey Pozdnyakov, Sergey Petrushanko,
//                   Grigory Safronov, Olga Kodolova
//         Created:  Thu Jul 12 18:12:19 CEST 2007
// $Id: HcalIsoTrkAnalyzer.cc,v 1.13 2009/06/29 22:39:01 anastass Exp $
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
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "DataFormats/DetId/interface/DetId.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "TrackingTools/TrackAssociator/interface/TrackDetMatchInfo.h"
#include "TrackingTools/TrackAssociator/interface/TrackAssociatorParameters.h"
#include "TrackingTools/TrackAssociator/interface/TrackDetectorAssociator.h"

#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"
#include "DataFormats/HcalDetId/interface/HcalSubdetector.h" 

#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h" 
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloEventSetup/interface/CaloTopologyRecord.h"
#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"

#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "TrackPropagation/SteppingHelixPropagator/interface/SteppingHelixPropagator.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"

#include "DataFormats/HcalIsolatedTrack/interface/IsolatedPixelTrackCandidate.h"
#include "DataFormats/HcalIsolatedTrack/interface/IsolatedPixelTrackCandidateFwd.h"

#include "Calibration/Tools/interface/MinL3AlgoUniv.h"

#include "TROOT.h"
#include "TH1.h"
#include "TH2.h"
#include "TFile.h"
#include "TTree.h"

#include "TString.h"
#include "TObject.h"
#include "TObjArray.h"
#include "TClonesArray.h"
#include "TRefArray.h"
#include "TLorentzVector.h"

#include "Calibration/HcalCalibAlgos/src/TCell.h"

#include "CondFormats/HcalObjects/interface/HcalRespCorrs.h"
#include "CondFormats/DataRecord/interface/HcalRespCorrsRcd.h"

#include <iostream>
#include <fstream>

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
  
 
  TrackDetectorAssociator trackAssociator_;
  TrackAssociatorParameters parameters_;

  const CaloGeometry* geo;
  InputTag hbheLabel_;
  InputTag hoLabel_;
  InputTag eLabel_;
  InputTag trackLabel_;
  InputTag trackLabel1_;

  std::string m_inputTrackLabel;
  std::string m_ecalLabel;
  std::string m_ebInstance;
  std::string m_eeInstance;
  std::string m_hcalLabel;
  int m_histoFlag;

  double associationConeSize_;
  string AxB_;
  bool allowMissingInputs_;
  string outputFileName_;

  double trackEta, trackPhi; 
  double rvert;
//  double eecal, ehcal;
  
/*AP
  int nHCRecHits,  nECRecHits, nHORecHits;
  double ecRHen[1500], ecRHeta[1500], ecRHphi[1500];
  double hcRHen[1500], hcRHeta[1500], hcRHphi[1500];
  int hcRHieta[1500], hcRHiphi[1500], hcRHdepth[1500];
  double hoRHen[1500], hoRHeta[1500], hoRHphi[1500];
  int hoRHieta[1500], hoRHiphi[1500], hoRHdepth[1500];
  double HcalAxBxDepthEnergy, MaxHhitEnergy;
  double HCAL3x3[9], HCAL5x5[25];
*/
  
/*
  int numbers[60][72];
  int numbers2[60][72];
  vector<float> EnergyVector;
  vector<vector<float> > EventMatrix;
  vector<vector<HcalDetId> > EventIds;
  MinL3AlgoUniv<HcalDetId>* MyL3Algo;
  map<HcalDetId,float> solution;
*/
  
  int nIterations, MinNTrackHitsBarrel,MinNTECHitsEndcap;
  float eventWeight;
  double energyMinIso, energyMaxIso;
  double energyECALmip, maxPNear;

  char dirname[50];
  char hname[20];
  char htitle[80];
  
  TFile* rootFile;
  TTree* tree;
  Float_t targetE;
  UInt_t numberOfCells;
  UInt_t cell;
  Float_t cellEnergy; 
  
  TClonesArray* cells;
  TRefArray* cells3x3;
  TRefArray* cellsPF;
  UInt_t  eventNumber;
  UInt_t  runNumber;
  Int_t   iEtaHit;
  UInt_t  iPhiHit;
  Float_t emEnergy;
  //    TLorentzVector* exampleP4; 
  TLorentzVector* tagJetP4; // dijet
  TLorentzVector* probeJetP4; // dijet
  Float_t etVetoJet; // dijet
  Float_t tagJetEmFrac; // dijet
  Float_t probeJetEmFrac; // dijet
  
  ofstream input_to_L3;
  
};

HcalIsoTrkAnalyzer::HcalIsoTrkAnalyzer(const edm::ParameterSet& iConfig)
{
  
  m_ecalLabel = iConfig.getUntrackedParameter<std::string> ("ecalRecHitsLabel","ecalRecHit");
  m_ebInstance = iConfig.getUntrackedParameter<std::string> ("ebRecHitsInstance","EcalRecHitsEB");
  m_eeInstance = iConfig.getUntrackedParameter<std::string> ("eeRecHitsInstance","EcalRecHitsEE");
  m_hcalLabel = iConfig.getUntrackedParameter<std::string> ("hcalRecHitsLabel","hbhereco");
/*AP  m_histoFlag = iConfig.getUntrackedParameter<int>("histoFlag",0);*/
 
  hbheLabel_= iConfig.getParameter<edm::InputTag>("hbheInput");
  hoLabel_=iConfig.getParameter<edm::InputTag>("hoInput");
  eLabel_=iConfig.getParameter<edm::InputTag>("eInput");
  trackLabel_ = iConfig.getParameter<edm::InputTag>("HcalIsolTrackInput");
  trackLabel1_ = iConfig.getParameter<edm::InputTag>("trackInput");
  associationConeSize_=iConfig.getParameter<double>("associationConeSize");
  allowMissingInputs_=iConfig.getUntrackedParameter<bool>("allowMissingInputs",true);
  outputFileName_=iConfig.getParameter<std::string>("outputFileName");

  AxB_=iConfig.getParameter<std::string>("AxB");

  nIterations = iConfig.getParameter<int>("noOfIterations");
  eventWeight = iConfig.getParameter<double>("eventWeight");
  energyMinIso = iConfig.getParameter<double>("energyMinIso");
  energyMaxIso = iConfig.getParameter<double>("energyMaxIso");
  energyECALmip = iConfig.getParameter<double>("energyECALmip");
  maxPNear = iConfig.getParameter<double>("maxPNear");
  MinNTrackHitsBarrel =  iConfig.getParameter<int>("MinNTrackHitsBarrel");
  MinNTECHitsEndcap =  iConfig.getParameter<int>("MinNTECHitsEndcap");

  edm::ParameterSet parameters = iConfig.getParameter<edm::ParameterSet>("TrackAssociatorParameters");
  parameters_.loadParameters( parameters );
  trackAssociator_.useDefaultPropagator();

}

HcalIsoTrkAnalyzer::~HcalIsoTrkAnalyzer()
{
}

// ------------ method called to for each event  ------------
void
HcalIsoTrkAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  using namespace edm;
  using namespace std;

  vector<float> rawEnergyVec;
  vector<int> detiphi;
  vector<int> detieta;
  vector<int> i3i5;
  vector<HcalDetId> detidvec;
  float calEnergy;

  edm::Handle<reco::TrackCollection> generalTracks;
  iEvent.getByLabel(trackLabel1_,generalTracks);  

  edm::Handle<reco::IsolatedPixelTrackCandidateCollection> trackCollection;
  iEvent.getByLabel(trackLabel_,trackCollection);
    
  edm::Handle<EcalRecHitCollection> ecal;
  iEvent.getByLabel(eLabel_,ecal);
  const EcalRecHitCollection Hitecal = *(ecal.product());
    
  edm::Handle<HBHERecHitCollection> hbhe;
  iEvent.getByLabel(hbheLabel_,hbhe);
  const HBHERecHitCollection Hithbhe = *(hbhe.product());

  edm::ESHandle<CaloGeometry> pG;
  iSetup.get<CaloGeometryRecord>().get(pG);
  geo = pG.product();

// rof 16.05.2008 start: include the possibility for recalibration (use "recalibrate" label for safety)
/*
  edm::ESHandle <HcalRespCorrs> recalibCorrs;
  iSetup.get<HcalRespCorrsRcd>().get("recalibrate",recalibCorrs);
  const HcalRespCorrs* myRecalib = recalibCorrs.product();
*/
// rof end

  parameters_.useEcal = true;
  parameters_.useHcal = true;
  parameters_.useCalo = false;
  parameters_.useMuon = false;
  parameters_.dREcal = 0.5;
  parameters_.dRHcal = 0.6;  

  if (trackCollection->size()==0) return;

  for (reco::TrackCollection::const_iterator trit=generalTracks->begin(); trit!=generalTracks->end(); trit++)
    {
      reco::IsolatedPixelTrackCandidateCollection::const_iterator isoMatched=trackCollection->begin();
      bool matched=false;
      for (reco::IsolatedPixelTrackCandidateCollection::const_iterator it = trackCollection->begin(); it!=trackCollection->end(); it++)
	{ 
	  //AP           if (floor(100000*trit->pt())==floor(100000*it->pt())) 
	  if (abs((trit->pt() - it->pt())/it->pt()) < 0.005 && abs(trit->eta() - it->eta()) < 0.01) 
	    {
	      isoMatched=it;
	      matched=true;
	      break;
	    }
	}
      if (!matched) continue;
      
      if (trit->hitPattern().numberOfValidHits()<MinNTrackHitsBarrel) continue;
      if (fabs(trit->eta())>1.47&&trit->hitPattern().numberOfValidStripTECHits()<MinNTECHitsEndcap) continue;
      
      //container for used recHits
      std::vector<int> usedHits; 
      //      

      calEnergy = sqrt(trit->px()*trit->px()+trit->py()*trit->py()+trit->pz()*trit->pz()+0.14*0.14);

      trackEta = trit->eta();
      trackPhi = trit->phi();
      
      double corrHCAL = 1.;
      /*      
      if (fabs(trackEta)<1.47) {

	if (calEnergy < 5.) corrHCAL = 1.55;
	if (calEnergy > 5. && calEnergy < 9.) corrHCAL = 1.55 - 0.18*(calEnergy-5.)/4.;
	if (calEnergy > 9. && calEnergy < 20.) corrHCAL = 1.37 - 0.18*(calEnergy-9.)/11.;
	if (calEnergy > 20. && calEnergy < 30.) corrHCAL = 1.19 - 0.06*(calEnergy-20.)/10.;
	if (calEnergy > 30. && calEnergy < 50.) corrHCAL = 1.13 - 0.02*(calEnergy-30.)/20.;
	if (calEnergy > 50. && calEnergy < 100.) corrHCAL = 1.11 - 0.02*(calEnergy-50.)/50.;
	if (calEnergy > 100. && calEnergy < 1000.) corrHCAL = 1.09 - 0.09*(calEnergy-100.)/900.;
      
      }
      
      if (fabs(trackEta)>1.47) {

	if (calEnergy < 5.) corrHCAL = 1.49;
	if (calEnergy > 5. && calEnergy < 9.) corrHCAL = 1.49 - 0.08*(calEnergy-5.)/4.;
	if (calEnergy > 9. && calEnergy < 20.) corrHCAL = 1.41- 0.15*(calEnergy-9.)/11.;
	if (calEnergy > 20. && calEnergy < 30.) corrHCAL = 1.26 - 0.07*(calEnergy-20.)/10.;
	if (calEnergy > 30. && calEnergy < 50.) corrHCAL = 1.19 - 0.04*(calEnergy-30.)/20.;
	if (calEnergy > 50. && calEnergy < 100.) corrHCAL = 1.15 - 0.03*(calEnergy-50.)/50.;
	if (calEnergy > 100. && calEnergy < 1000.) corrHCAL = 1.12 - 0.12*(calEnergy-100.)/900.;
      
      }      
      */

      //      cout << endl << " ISO TRACK E = "<< calEnergy << " ETA = " << trackEta<< " PHI = " << trackPhi <<  " Correction " <<  corrHCAL<< endl;
      
      rvert = sqrt(trit->vx()*trit->vx()+trit->vy()*trit->vy()+trit->vz()*trit->vz());      
      
      //Associate track with a calorimeter
      TrackDetMatchInfo info = trackAssociator_.associate(iEvent, iSetup,trackAssociator_.getFreeTrajectoryState(iSetup, *trit),parameters_);
     //*(it->track().get())
      double etaecal=info.trkGlobPosAtEcal.eta();
      double phiecal=info.trkGlobPosAtEcal.phi();

//      eecal=info.coneEnergy(parameters_.dREcal, TrackDetMatchInfo::EcalRecHits);
//      ehcal=info.coneEnergy(parameters_.dRHcal, TrackDetMatchInfo::HcalRecHits);

      double rmin = 0.07;
      if( fabs(etaecal) > 1.47 ) rmin = 0.07*(fabs(etaecal)-0.47)*1.2;
      if( fabs(etaecal) > 2.2 ) rmin = 0.07*(fabs(etaecal)-0.47)*1.4;

      struct
      {
	int iphihitm;
	int ietahitm;
	int depthhit;
	double hitenergy;
      } MaxHit;

      // Find Ecal RecHit with maximum energy and collect other information
      MaxHit.hitenergy=-100;
/*AP      nECRecHits=0; */
      
      double econus = 0.;
      float ecal_cluster = 0.;      

      //clear usedHits
      usedHits.clear();
      //

      for (std::vector<EcalRecHit>::const_iterator ehit=Hitecal.begin(); ehit!=Hitecal.end(); ehit++)	
	{
	  //check that this hit was not considered before and push it into usedHits
	  bool hitIsUsed=false;
	  int hitHashedIndex=-10000; 
	  if (ehit->id().subdetId()==EcalBarrel) 
	    {
	      EBDetId did(ehit->id());
	      hitHashedIndex=did.hashedIndex();
	    }
	  
	  if (ehit->id().subdetId()==EcalEndcap) 
	    {
	      EEDetId did(ehit->id());
	      hitHashedIndex=did.hashedIndex();
	    }
	  for (uint32_t i=0; i<usedHits.size(); i++)
	    {
	      if (usedHits[i]==hitHashedIndex) hitIsUsed=true;
	    }
	  if (hitIsUsed) continue;
	  usedHits.push_back(hitHashedIndex);
	  //

	  if((*ehit).energy() > MaxHit.hitenergy) 
	    {
	      MaxHit.hitenergy = (*ehit).energy();
	    }

	  GlobalPoint pos = geo->getPosition((*ehit).detid());
	  double phihit = pos.phi();
	  double etahit = pos.eta();
	 
	  double dphi = fabs(phiecal - phihit); 
	  if(dphi > 4.*atan(1.)) dphi = 8.*atan(1.) - dphi;
	  double deta = fabs(etaecal - etahit); 
	  double dr = sqrt(dphi*dphi + deta*deta);

	  if (dr < rmin) {
	    econus = econus + (*ehit).energy();
	  }
	
	    if (dr < 0.13) ecal_cluster += (*ehit).energy();
	  /*AP    
	    ecRHen [nECRecHits] = (*ehit).energy();
	    ecRHeta[nECRecHits] = etahit;
	    ecRHphi[nECRecHits] = phihit;
	    nECRecHits++;
	    
	  */
	}
      MaxHit.hitenergy=-100;
	  /*AP     nHCRecHits=0; */
	  
	  //clear usedHits
	  usedHits.clear();
	  //
	  
      float dddeta = 1000.;
      float dddphi = 1000.;
      int iphitrue = 1234;
      int ietatrue = 1234;

      for (HBHERecHitCollection::const_iterator hhit=Hithbhe.begin(); hhit!=Hithbhe.end(); hhit++) 
	{
	  
	  //check that this hit was not considered before and push it into usedHits
	  bool hitIsUsed=false;
	  int hitHashedIndex=hhit->id().hashed_index();
	  for (uint32_t i=0; i<usedHits.size(); i++)
	    {
	      if (usedHits[i]==hitHashedIndex) hitIsUsed=true;
	    }
	  if (hitIsUsed) continue;
	  usedHits.push_back(hitHashedIndex);
	  //

	  // rof 16.05.2008 start: include the possibility for recalibration
	  float recal = 1;
	  // rof end

	  GlobalPoint pos = geo->getPosition(hhit->detid());
	  double phihit = pos.phi();
	  double etahit = pos.eta();
	  
	  int iphihitm  = (hhit->id()).iphi();
	  int ietahitm  = (hhit->id()).ieta();
	  int depthhit = (hhit->id()).depth();
	  double enehit = hhit->energy() * recal;
	  
	  if (depthhit!=1) continue;
	   
	  double dphi = fabs(info.trkGlobPosAtHcal.phi() - phihit); 
	  if(dphi > 4.*atan(1.)) dphi = 8.*atan(1.) - dphi;
	  double deta = fabs(info.trkGlobPosAtHcal.eta() - etahit); 
	  double dr = sqrt(dphi*dphi + deta*deta);
	  
	  if (deta<dddeta) {
	   ietatrue = ietahitm;
	   dddeta=deta;
	  }
	  
	  if (dphi<dddphi) {
	  iphitrue = iphihitm;
	  dddphi=dphi;
	  }
	  
	  if(dr<associationConeSize_) 
	    {
	      /*AP
		hcRHen[nHCRecHits]    = hhit->energy() * recal;
		hcRHeta[nHCRecHits]   = etahit;
		hcRHphi[nHCRecHits]   = phihit;
		hcRHieta[nHCRecHits]  = ietahitm;
		hcRHiphi[nHCRecHits]  = iphihitm;
		hcRHdepth[nHCRecHits] = depthhit;
		nHCRecHits++;
	      */
	      
	      for (HBHERecHitCollection::const_iterator hhit2=Hithbhe.begin(); hhit2!=Hithbhe.end(); hhit2++) 
		{
		  int iphihitm2  = (hhit2->id()).iphi();
		  int ietahitm2  = (hhit2->id()).ieta();
		  int depthhit2 = (hhit2->id()).depth();
		  double enehit2 = hhit2->energy() * recal;
		  
		  if ( iphihitm==iphihitm2 && ietahitm==ietahitm2  && depthhit!=depthhit2){
		    
		    enehit = enehit+enehit2;
		    
		  }
		  
		}
	      
	      if(enehit > MaxHit.hitenergy) 
		{
		  MaxHit.hitenergy =  enehit;
		  MaxHit.ietahitm   = (hhit->id()).ieta();
		  MaxHit.iphihitm   = (hhit->id()).iphi();
		  MaxHit.depthhit  = (hhit->id()).depth();
		}
	    }
	}
      
      /*AP
	MaxHhitEnergy = MaxHit.hitenergy;
	
	if(m_histoFlag==1) 
	{
	int MinIETA= 999;
	int MaxIETA= -999;
	int MinIPHI= 999;   
	int MaxIPHI= -999;
	for (int k=0; k<nHCRecHits; k++)
	{
	
	MinIETA = MinIETA > hcRHieta[k] ? hcRHieta[k] : MinIETA;
	MaxIETA = MaxIETA < hcRHieta[k] ? hcRHieta[k] : MaxIETA;
	MinIPHI = MinIPHI > hcRHiphi[k] ? hcRHiphi[k] : MinIPHI;
	MaxIPHI = MaxIPHI < hcRHiphi[k] ? hcRHiphi[k] : MaxIPHI;
	}
	}
      */
      
      // ---AP---
      Bool_t passCuts = kFALSE;
      if(calEnergy > energyMinIso && calEnergy < energyMaxIso && isoMatched->energyIn() < energyECALmip && 
	 isoMatched->maxPtPxl() < maxPNear && abs(MaxHit.ietahitm)<30 && MaxHit.hitenergy > 0.){ passCuts = kTRUE; }
      
      // --- AP --- 
      
      if(AxB_=="5x5" || AxB_=="3x3")
	{
	  /*AP
	    HcalAxBxDepthEnergy=0.;
	    for(int ih=0; ih<9; ih++){HCAL3x3[ih]=0.;}
	    for(int iht=0; iht<25; iht++){HCAL5x5[iht]=0.;}
	  */
	  //clear usedHits
	  usedHits.clear();
	  //

	  for (HBHERecHitCollection::const_iterator hhit=Hithbhe.begin(); hhit!=Hithbhe.end(); hhit++) 
	    {

	      //check that this hit was not considered before and push it into usedHits
	      bool hitIsUsed=false;
	      int hitHashedIndex=hhit->id().hashed_index();
	      for (uint32_t i=0; i<usedHits.size(); i++)
		{
		  if (usedHits[i]==hitHashedIndex) hitIsUsed=true;
		}
	      if (hitIsUsed) continue;
	      usedHits.push_back(hitHashedIndex);
	      //

	      int DIETA = 100;
	      if(MaxHit.ietahitm*(hhit->id()).ieta()>0)
		{
		  DIETA = MaxHit.ietahitm - (hhit->id()).ieta();
		}
	      if(MaxHit.ietahitm*(hhit->id()).ieta()<0)
		{
		  DIETA = MaxHit.ietahitm - (hhit->id()).ieta();
		  DIETA = DIETA>0 ? DIETA-1 : DIETA+1; 
		}
	      
	      int DIPHI = abs(MaxHit.iphihitm - (hhit->id()).iphi());
	      DIPHI = DIPHI>36 ? 72-DIPHI : DIPHI;
	      /*AP DIPHI = DIPHI<-36 ? 72+DIPHI : DIPHI; */
	      
	      int numbercell=0;
	      if(AxB_=="3x3") numbercell = 1;
	      if(AxB_=="5x5") numbercell = 2;
	      
	      if( abs(DIETA)<=numbercell && (abs(DIPHI)<=numbercell || ( abs(MaxHit.ietahitm)>20 && abs(DIPHI)<=numbercell+1)) )  {
		
		// rof 16.05.2008 start: include the possibility for recalibration
		float recal = 1;
		
  	        int iii3i5 = 0;
		/*AP
		  if(DIPHI==-1  && DIETA== 1) {HCAL3x3[0] += hhit->energy() * recal; iii3i5 = 1;}
		  if(DIPHI==-1  && DIETA== 0) {HCAL3x3[1] += hhit->energy() * recal; iii3i5 = 1;}
		  if(DIPHI==-1  && DIETA==-1) {HCAL3x3[2] += hhit->energy() * recal; iii3i5 = 1;}
		  if(DIPHI== 0  && DIETA== 1) {HCAL3x3[3] += hhit->energy() * recal; iii3i5 = 1;}
		  if(DIPHI== 0  && DIETA== 0) {HCAL3x3[4] += hhit->energy() * recal; iii3i5 = 1;}
		  if(DIPHI== 0  && DIETA==-1) {HCAL3x3[5] += hhit->energy() * recal; iii3i5 = 1;}
		  if(DIPHI== 1  && DIETA== 1) {HCAL3x3[6] += hhit->energy() * recal; iii3i5 = 1;}
		  if(DIPHI== 1  && DIETA== 0) {HCAL3x3[7] += hhit->energy() * recal; iii3i5 = 1;}
		  if(DIPHI== 1  && DIETA==-1) {HCAL3x3[8] += hhit->energy() * recal; iii3i5 = 1;}

		  if(DIPHI==-2  && DIETA== 2) {HCAL5x5[0] += hhit->energy() * recal;}
		  if(DIPHI==-2  && DIETA== 1) {HCAL5x5[1] += hhit->energy() * recal;}
		  if(DIPHI==-2  && DIETA== 0) {HCAL5x5[2] += hhit->energy() * recal;}
		  if(DIPHI==-2  && DIETA==-1) {HCAL5x5[3] += hhit->energy() * recal;}
		  if(DIPHI==-2  && DIETA==-2) {HCAL5x5[4] += hhit->energy() * recal;}

		  if(DIPHI==-1  && DIETA== 2) {HCAL5x5[5] += hhit->energy() * recal;}
		  if(DIPHI==-1  && DIETA== 1) {HCAL5x5[6] += hhit->energy() * recal;}
		  if(DIPHI==-1  && DIETA== 0) {HCAL5x5[7] += hhit->energy() * recal;}
		  if(DIPHI==-1  && DIETA==-1) {HCAL5x5[8] += hhit->energy() * recal;}
		  if(DIPHI==-1  && DIETA==-2) {HCAL5x5[9] += hhit->energy() * recal;}

		  if(DIPHI== 0  && DIETA== 2) {HCAL5x5[10] += hhit->energy() * recal;}
		  if(DIPHI== 0  && DIETA== 1) {HCAL5x5[11] += hhit->energy() * recal;}
		  if(DIPHI== 0  && DIETA== 0) {HCAL5x5[12] += hhit->energy() * recal;}
		  if(DIPHI== 0  && DIETA==-1) {HCAL5x5[13] += hhit->energy() * recal;}
		  if(DIPHI== 0  && DIETA==-2) {HCAL5x5[14] += hhit->energy() * recal;}
		  
		  if(DIPHI== 1  && DIETA== 2) {HCAL5x5[15] += hhit->energy() * recal;}
		  if(DIPHI== 1  && DIETA== 1) {HCAL5x5[16] += hhit->energy() * recal;}
		  if(DIPHI== 1  && DIETA== 0) {HCAL5x5[17] += hhit->energy() * recal;}
		  if(DIPHI== 1  && DIETA==-1) {HCAL5x5[18] += hhit->energy() * recal;}
		  if(DIPHI== 1  && DIETA==-2) {HCAL5x5[19] += hhit->energy() * recal;}
		  
		  if(DIPHI== 2  && DIETA== 2) {HCAL5x5[20] += hhit->energy() * recal;}
		  if(DIPHI== 2  && DIETA== 1) {HCAL5x5[21] += hhit->energy() * recal;}
		  if(DIPHI== 2  && DIETA== 0) {HCAL5x5[22] += hhit->energy() * recal;}
		  if(DIPHI== 2  && DIETA==-1) {HCAL5x5[23] += hhit->energy() * recal;}
		  if(DIPHI== 2  && DIETA==-2) {HCAL5x5[24] += hhit->energy() * recal;}
		*/
		if(passCuts){
		  /*AP
		  //if(calEnergy > energyMinIso && calEnergy < energyMaxIso 
		  //    && isoMatched->energyIn() < energyECALmip && isoMatched->maxPtPxl() < maxPNear 
		  //    && (MaxHit.ietahitm+29) > -1 && (MaxHit.ietahitm+29) < 60 && MaxHit.hitenergy > 0.){
		  */
		  
		  rawEnergyVec.push_back(hhit->energy() * recal * corrHCAL);
		  detidvec.push_back(hhit->id());
		  detiphi.push_back((hhit->id()).iphi());
		  detieta.push_back((hhit->id()).ieta());
		  i3i5.push_back(iii3i5);
		  //		      numbers2[(hhit->id()).ieta()+29][(hhit->id()).iphi()] = numbers2[(hhit->id()).ieta()+29][(hhit->id()).iphi()] + 1;
		}
	      }
	    }
	}

      if(AxB_!="3x3" && AxB_!="5x5") LogWarning(" AxB ")<<"   Not supported: "<< AxB_;
      
      /*
	if(calEnergy > energyMinIso && calEnergy < energyMaxIso &&
	isoMatched->energyIn() < energyECALmip && isoMatched->maxPtPxl() < maxPNear &&
	(MaxHit.ietahitm+29) > -1 && (MaxHit.ietahitm+29) < 60 && MaxHit.hitenergy > 0.)
	{
	EventMatrix.push_back(rawEnergyVec);
	EventIds.push_back(detidvec);
	EnergyVector.push_back(calEnergy);
	numbers[MaxHit.ietahitm+29][MaxHit.iphihitm] = numbers[MaxHit.ietahitm+29][MaxHit.iphihitm] + 1;
	}
      */
      
      if(passCuts){
	/*AP
	//if(calEnergy > energyMinIso && calEnergy < energyMaxIso 
	// && isoMatched->energyIn() < energyECALmip && isoMatched->maxPtPxl() < maxPNear 
	//&& (MaxHit.ietahitm+29) > -1 && (MaxHit.ietahitm+29) < 60 && MaxHit.hitenergy > 0.){
	*/
	
	input_to_L3 << rawEnergyVec.size() << "   " << calEnergy;
	//	  input_to_L3 << rawEnergyVec.size() << "   " << calEnergy <<  "   " << ietatrue << "   " << iphitrue;
	
	for (unsigned int i=0; i<rawEnergyVec.size(); i++)
	  {
	    input_to_L3 << "   " << rawEnergyVec.at(i) << "   " << detidvec.at(i).rawId() ;	    
	    //	      input_to_L3 << "   " << rawEnergyVec.at(i) << "   " << detidvec.at(i).rawId() << "   " << detiphi.at(i) << "   " << detieta.at(i);
	  }
	input_to_L3 <<endl;
	
        eventNumber = iEvent.id().event();
        runNumber = iEvent.id().run();
        iEtaHit = ietatrue;
        iPhiHit = iphitrue;
        emEnergy = isoMatched->energyIn();
//        exampleP4->SetPxPyPzE(2, 1, 1, 10);
	
	numberOfCells=rawEnergyVec.size();
	targetE = calEnergy;
        
        for (unsigned int ia=0; ia<numberOfCells; ++ia) {
	  cellEnergy = rawEnergyVec.at(ia);
	  cell = detidvec.at(ia).rawId();
          
	  new((*cells)[ia])  TCell(cell, cellEnergy);
	  
	  /*            
			if (i3i5.at(ia)==1) {
			cells3x3->Add((*cells)[ia]);
			//	        input_to_L3 << "  3x3 " << detiphi.at(ia) << "   " <<detieta.at(ia) << endl;
			}
			if (i3i5.at(ia)==1) {
			cellsPF->Add((*cells)[ia]);
			}
	  */
        } 
	
        tree->Fill();
        
	//        cells3x3->Clear();
	//        cellsPF->Clear();
        cells->Clear();	  
	
      }
      
      rawEnergyVec.clear();
      detidvec.clear();
      detiphi.clear();
      detieta.clear();
      i3i5.clear();

      /*AP      nHORecHits=0; */
      try {
	Handle<HORecHitCollection> ho;
	iEvent.getByLabel(hoLabel_,ho);
	const HORecHitCollection Hitho = *(ho.product());
	
	//clear usedHits
	usedHits.clear();
	//
	
	for(HORecHitCollection::const_iterator hoItr=Hitho.begin(); hoItr!=Hitho.end(); hoItr++)
	  {

	    //check that this hit was not considered before and push it into usedHits
	    bool hitIsUsed=false;
	    int hitHashedIndex=hoItr->id().hashed_index();
	    for (uint32_t i=0; i<usedHits.size(); i++)
	      {
		if (usedHits[i]==hitHashedIndex) hitIsUsed=true;
	      }
	    if (hitIsUsed) continue;
	    usedHits.push_back(hitHashedIndex);
	    //
	    /*AP do not delete this yet. might be useful later
	    // rof 16.05.2008 start: include the possibility for recalibration
	    float recal = 1; 
	    // rof end

	    GlobalPoint pos = geo->getPosition(hoItr->detid());
	    double phihit = pos.phi();
	    double etahit = pos.eta();
	    
	    int iphihitm = (hoItr->id()).iphi();
	    int ietahitm = (hoItr->id()).ieta();
	    int depthhit = (hoItr->id()).depth();
	    
	    double dphi = fabs(trackPhi - phihit); 
	    if(dphi > 4.*atan(1.)) dphi = 8.*atan(1.) - dphi;
	    double deta = fabs(trackEta - etahit); 
	    double dr = sqrt(dphi*dphi + deta*deta);
	    */
	    
	    /*AP	    
	      if(dr<associationConeSize_) {
	      hoRHen[nHORecHits] = hoItr->energy() * recal;
	      hoRHeta[nHORecHits] = etahit;
	      hoRHphi[nHORecHits] = phihit;
	      hoRHieta[nHORecHits] = ietahitm;
	      hoRHiphi[nHORecHits] = iphihitm;
	      hoRHdepth[nHORecHits] = depthhit;
	      nHORecHits++;
	      }
	    */	      
	    
	  }
      } catch (cms::Exception& e) { // can't find it!
	if (!allowMissingInputs_) throw e;
      }
    }
}
    
      

// ------------ method called once each job just before starting event loop  ------------
void 
HcalIsoTrkAnalyzer::beginJob(const edm::EventSetup&)
{

  //  MyL3Algo = new MinL3AlgoUniv<HcalDetId>(eventWeight);

    input_to_L3.open("input_to_l3.txt");

    rootFile = new TFile("rootFile.root", "RECREATE");
    tree = new TTree("hcalCalibTree", "Tree for IsoTrack Calibration");  
    cells = new TClonesArray("TCell", 10000);
//    cells3x3 = new TRefArray;
//    cellsPF = new TRefArray;
//    exampleP4 = new TLorentzVector();
    tagJetP4 = new TLorentzVector(); // dijet
    probeJetP4 = new TLorentzVector();  // dijet   
    
    tree->Branch("cells", &cells, 64000, 0);
    tree->Branch("targetE", &targetE, "targetE/F");
    tree->Branch("emEnergy", &emEnergy, "emEnergy/F");
//    tree->Branch("exampleP4", "TLorentzVector", &exampleP4);
//    tree->Branch("cells3x3", &cells3x3, 64000, 0);
//    tree->Branch("cellsPF", &cellsPF, 64000, 0);
    tree->Branch("iEtaHit", &iEtaHit, "iEtaHit/I");
    tree->Branch("iPhiHit", &iPhiHit, "iPhiHit/i");
    tree->Branch("eventNumber", &eventNumber, "eventNumber/i");
    tree->Branch("runNumber", &runNumber, "runNumber/i");

    tree->Branch("tagJetP4", "TLorentzVector", &tagJetP4); // dijet
    tree->Branch("probeJetP4", "TLorentzVector", &probeJetP4); // dijet
    tree->Branch("etVetoJet", &etVetoJet, "etVetoJet/F"); // dijet
    tree->Branch("tagJetEmFrac", &tagJetEmFrac,"tagJetEmFrac/F"); // dijet
    tree->Branch("probeJetEmFrac", &probeJetEmFrac,"probeJetEmFrac/F"); // dijet

}

// ------------ method called once each job just after ending the event loop  ------------
void 
HcalIsoTrkAnalyzer::endJob() {

  /*
  // perform the calibration with given number of iterations  
  cout<<" Begin of solution "<< EnergyVector.size() << "  "<< EventMatrix.size()<< "  "<< EventIds.size()<<endl; 
  solution = MyL3Algo->iterate(EventMatrix, EventIds, EnergyVector, nIterations);
  cout<<" End of solution "<<endl; 

  // print the solution and make a few plots
  map<HcalDetId,float>::iterator ii;
  for (ii = solution.begin(); ii != solution.end(); ii++)
    {
      int curr_eta = ii->first.ieta();
      int curr_phi = ii->first.iphi();
      int curr_depth = ii->first.depth();
      cout << "solution[eta=" << curr_eta << ",phi=" << curr_phi << ",subdet=" << ii->first.subdet()
	   << ",depth=" << curr_depth << "] = " << ii->second 
	   << " Stat " << numbers[curr_eta+29][curr_phi] << "  " << numbers2[curr_eta+29][curr_phi]
	   << endl; 
    }

  for (ii = solution.begin(); ii != solution.end(); ii++)
    {
      int curr_eta = ii->first.ieta();
      int curr_phi = ii->first.iphi();
      int curr_depth = ii->first.depth();
      cout << "  "<< ii->first.subdet() << "  " << curr_depth  << "  "<< curr_eta << "  " << curr_phi <<  "  "<< ii->second
	   << endl; 
    }
  */
  
    input_to_L3.close();

    rootFile->Write();
    rootFile->Close();

    if (cells) delete cells;
//    if (cells3x3) delete cells3x3;
//    if (cellsPF) delete cellsPF;
//    if (exampleP4) delete exampleP4;
    if (tagJetP4) delete tagJetP4; // dijet
    if (probeJetP4) delete probeJetP4; // dijet

}

//define this as a plug-in
DEFINE_FWK_MODULE(HcalIsoTrkAnalyzer);
