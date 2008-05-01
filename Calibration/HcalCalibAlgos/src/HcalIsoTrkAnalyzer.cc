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
//                   ... and Sergey Petrushanko (all lines between M+ and M-)
//         Created:  Thu Jul 12 18:12:19 CEST 2007
// $Id: HcalIsoTrkAnalyzer.cc,v 1.4 2007/10/10 06:18:25 dlange Exp $
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
#include "Geometry/Records/interface/IdealGeometryRecord.h"
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

// M+
#include "Calibration/Tools/interface/MinL3AlgoUniv.h"
// M-

#include "TROOT.h"
#include "TH1.h"
#include "TH2.h"
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
  
 
  TrackDetectorAssociator trackAssociator_;
  TrackAssociatorParameters parameters_;

  const CaloGeometry* geo;
  InputTag hbheLabel_;
  InputTag hoLabel_;
  InputTag eLabel_;
  InputTag trackLabel_;

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

  int NisoTrk;
  double trackPt, trackEta, trackPhi; 

  double ptrack, rvert;
  double eecal, ehcal;
  
  int nHCRecHits,  nECRecHits, nHORecHits;
  double ecRHen[1500], ecRHeta[1500], ecRHphi[1500];

  double hcRHen[1500], hcRHeta[1500], hcRHphi[1500];
  int hcRHieta[1500], hcRHiphi[1500], hcRHdepth[1500];
  double hoRHen[1500], hoRHeta[1500], hoRHphi[1500];
  int hoRHieta[1500], hoRHiphi[1500], hoRHdepth[1500];

  double HcalAxBxDepthEnergy, MaxHhitEnergy;
  double HCAL3x3[9], HCAL5x5[25];

  int numbers[60][72];
  int numbers2[60][72];

// M+
  vector<float> EnergyVector;
  vector<vector<float> > EventMatrix;
  vector<vector<HcalDetId> > EventIds;
  MinL3AlgoUniv<HcalDetId>* MyL3Algo;
  map<HcalDetId,float> solution;
  int nIterations;
  float eventWeight;
  double energyMinIso, energyMaxIso;
// M-  

  TFile* tf2;

  TH1F* thDrTrEHits;
  TH1F* thDrTrHBHEHits;
  TH1F* thDrTrHOHits;

  TH1F* isoTrE;
  TH1F* isoTrEta;
  TH1F* isoTrPhi;
  TH1F* isoEcalCone;
  
  TTree* CalibTree;
   char dirname[50];
   char hname[20];
   char htitle[80];

};



HcalIsoTrkAnalyzer::HcalIsoTrkAnalyzer(const edm::ParameterSet& iConfig)
{
   //now do what ever initialization is needed

  m_ecalLabel = iConfig.getUntrackedParameter<std::string> ("ecalRecHitsLabel","ecalRecHit");
  m_ebInstance = iConfig.getUntrackedParameter<std::string> ("ebRecHitsInstance","EcalRecHitsEB");
  m_eeInstance = iConfig.getUntrackedParameter<std::string> ("eeRecHitsInstance","EcalRecHitsEE");
  m_hcalLabel = iConfig.getUntrackedParameter<std::string> ("hcalRecHitsLabel","hbhereco");
  m_histoFlag = iConfig.getUntrackedParameter<int>("histoFlag",0);
 
  hbheLabel_= iConfig.getParameter<edm::InputTag>("hbheInput");
  hoLabel_=iConfig.getParameter<edm::InputTag>("hoInput");
  eLabel_=iConfig.getParameter<edm::InputTag>("eInput");
  trackLabel_ = iConfig.getParameter<edm::InputTag>("trackInput");
  associationConeSize_=iConfig.getParameter<double>("associationConeSize");
  allowMissingInputs_=iConfig.getParameter<bool>("allowMissingInputs");
  outputFileName_=iConfig.getParameter<std::string>("outputFileName");

  AxB_=iConfig.getParameter<std::string>("AxB");

// M+
  nIterations = iConfig.getUntrackedParameter<int>("noOfIterations",10);
  eventWeight = iConfig.getUntrackedParameter<double>("eventWeight",0.);
  energyMinIso = iConfig.getUntrackedParameter<double>("energyMinIso",2.);
  energyMaxIso = iConfig.getUntrackedParameter<double>("energyMaxIso",1000.);
// M-

  edm::ParameterSet parameters = iConfig.getParameter<edm::ParameterSet>("TrackAssociatorParameters");
  parameters_.loadParameters( parameters );
  trackAssociator_.useDefaultPropagator();

 }

HcalIsoTrkAnalyzer::~HcalIsoTrkAnalyzer()
{

//  if(m_histoFlag==1)  {tf2 -> Close();}
  
  tf2 -> Close();
  
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
}


// ------------ method called to for each event  ------------
void
HcalIsoTrkAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  using namespace edm;
  using namespace std;

// M+
   vector<float> rawEnergyVec;
   vector<HcalDetId> detidvec;
   float calEnergy;
// M-

  
  edm::Handle<reco::TrackCollection> trackCollection;
  iEvent.getByLabel(trackLabel_,trackCollection);
  const reco::TrackCollection isoTrack = *(trackCollection.product());
//  LogInfo("IsoTracks: ")<<" IsoTracks size "<<(isoTrack).size();
//    cout << " IsoTracks size "<<(isoTrack).size() << endl;
    
  edm::Handle<EcalRecHitCollection> ecal;
  iEvent.getByLabel(eLabel_,ecal);
  const EcalRecHitCollection Hitecal = *(ecal.product());
//  LogInfo("ECAL: ")<<" Size of ECAl "<<(Hitecal).size();
//    cout << " Size of ECAl "<<(Hitecal).size() << endl;
    
  edm::Handle<HBHERecHitCollection> hbhe;
  iEvent.getByLabel(hbheLabel_,hbhe);
  const HBHERecHitCollection Hithbhe = *(hbhe.product());
//  LogInfo("HBHE: ")<<" Size of HBHE "<<(Hithbhe).size();

  ESHandle<CaloGeometry> pG;
  iSetup.get<IdealGeometryRecord>().get(pG);
  geo = pG.product();
  

  parameters_.useEcal = true;
  parameters_.useHcal = true;
  parameters_.useCalo = false;
  parameters_.dREcal = 0.5;
  parameters_.dRHcal = 0.6;  

//	    cout<<"  BEGINNING "<< calEnergy << "  " << rawEnergyVec.size() <<  "  " << detidvec.size()<< endl; 

 for (reco::TrackCollection::const_iterator it = isoTrack.begin(); it!=isoTrack.end(); it++)
    { 

//	    cout<<"  BEGINNING 1 "<< calEnergy << "  " << rawEnergyVec.size() <<  "  " << detidvec.size()<< endl; 

      NisoTrk++;

      ptrack = sqrt(it->px()*it->px()+it->py()*it->py()+it->pz()*it->pz());

// M+      
      calEnergy = sqrt(it->px()*it->px()+it->py()*it->py()+it->pz()*it->pz()+0.14*0.14);
// M-

      
      //if (ptrack < 20) continue;
      
      trackPt  = it->pt();
      trackEta = it->eta();
      trackPhi = it->phi();
      
      double corrHCAL = 1.;
      
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

	  if (abs(trackEta)<2.1)  {
	  
	  isoTrE -> Fill(calEnergy);
	  isoTrEta -> Fill(trackEta);
	  isoTrPhi -> Fill(trackPhi);
	  
//	  isoTrE -> Fill(calEnergy);
	  
	  }


//      cout << endl << " ISO TRACK E = "<< calEnergy << " ETA = " << trackEta<< " PHI = " << trackPhi <<  " Correction " <<  corrHCAL<< endl;

      rvert = sqrt(it->vx()*it->vx()+it->vy()*it->vy()+it->vz()*it->vz());      
                 
      //Associate track with a calorimeter
      TrackDetMatchInfo info = trackAssociator_.associate(iEvent, iSetup, trackAssociator_.getFreeTrajectoryState(iSetup, *it), parameters_);
      
      double etaecal=info.trkGlobPosAtEcal.eta();
      double phiecal=info.trkGlobPosAtEcal.phi();
      
      //double thetaecal=2.*atan(1.)-asin(2.*exp(etaecal)/(1.+exp(2.*etaecal)));
      //if(etaecal<0) thetaecal=-thetaecal;
      //double etahcal=info.trkGlobPosAtHcal.eta();
      //double phihcal=info.trkGlobPosAtHcal.phi();

      eecal=info.coneEnergy(parameters_.dREcal, TrackDetMatchInfo::EcalRecHits);
      ehcal=info.coneEnergy(parameters_.dRHcal, TrackDetMatchInfo::HcalRecHits);

	   double rmin = 0.07;
	   if( fabs(etaecal) > 1.47 ) rmin = 0.07*(fabs(etaecal)-0.47)*1.2;
	   if( fabs(etaecal) > 2.2 ) rmin = 0.07*(fabs(etaecal)-0.47)*1.4;
      
//      cout << " ISO TRACK ecal "<<  eecal<< " ETA = "<< etaecal  << " PHI = " << phiecal << "  " << rmin<< endl;
  
      //LogInfo("CaloConeEnergy:")<<" eecalCone:"<<eecal<<" ehcalCone:"<<ehcal<<" etaecal:"<<etaecal<<" phiecal:"<<phiecal<<" etahcal:"<<etahcal<<" phihcal:"<<phihcal;
                  
// Select ECAL & HCAL RecHits 

      struct
      {
	int iphihit;
	int ietahit;
	int depthhit;
	double hitenergy;
      } MaxHit;

      // Find Ecal RecHit with maximum energy and collect other information
      MaxHit.hitenergy=-100;
      nECRecHits=0;
      
      double econus = 0.;
      
      for (std::vector<EcalRecHit>::const_iterator ehit=Hitecal.begin(); ehit!=Hitecal.end(); ehit++)	
{
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

//         cout << " eta "<<  etahit << " phi "<< phihit << " en " << (*ehit).energy() << "  dr "<< dr << endl;

	 if (dr < rmin) {
	   econus = econus + (*ehit).energy();
	 }
	 
	 if(m_histoFlag==1)  {thDrTrEHits -> Fill(dr);}
	 ecRHen [nECRecHits] = (*ehit).energy();
	 ecRHeta[nECRecHits] = etahit;
	 ecRHphi[nECRecHits] = phihit;
	 nECRecHits++;
	}

//      cout << " ISO TRACK ecal "<<  eecal<< " ETA = "<< etaecal  << " PHI = " << phiecal << "  " << econus << endl;

	  if (abs(trackEta)<2.1 && calEnergy > 5.)  {
                isoEcalCone-> Fill(econus);
            }

      //LogInfo("iCoord of MaxHhit: ")<<" ehitenergy:"<<MaxHit.hitenergy;
      
      // Find Hcal RecHit with maximum energy and collect other information
      
      MaxHit.hitenergy=-100;
      nHCRecHits=0;
      
      for (HBHERecHitCollection::const_iterator hhit=Hithbhe.begin(); hhit!=Hithbhe.end(); hhit++) 
	{
	  // LogInfo("ALL HBHERecHit: ")<<" ieta->"<<(hhit->id()).ieta()<<" iphi->"<<(hhit->id()).iphi()<<" depth->"<<(hhit->id()).depth();


	  GlobalPoint pos = geo->getPosition(hhit->detid());
	  double phihit = pos.phi();
	  double etahit = pos.eta();
	  
	  int iphihit  = (hhit->id()).iphi();
	  int ietahit  = (hhit->id()).ieta();
	  int depthhit = (hhit->id()).depth();
	  
	  // LogInfo("iCoord: ")<<ietahit<<"   "<<iphihit<<"       "<< depthhit;
	  
	  double dphi = fabs(info.trkGlobPosAtHcal.phi() - phihit); 
	  if(dphi > 4.*atan(1.)) dphi = 8.*atan(1.) - dphi;
	  double deta = fabs(info.trkGlobPosAtHcal.eta() - etahit); 
	  double dr = sqrt(dphi*dphi + deta*deta);
	  
	  if(m_histoFlag==1)  { thDrTrHBHEHits -> Fill(dr);}
	  
	  if(dr<associationConeSize_) 
	    {
	      hcRHen[nHCRecHits]    = hhit->energy();
	      hcRHeta[nHCRecHits]   = etahit;
	      hcRHphi[nHCRecHits]   = phihit;
	      hcRHieta[nHCRecHits]  = ietahit;
	      hcRHiphi[nHCRecHits]  = iphihit;
	      hcRHdepth[nHCRecHits] = depthhit;
	      nHCRecHits++;
	 
	      if(hhit->energy() > MaxHit.hitenergy) 
		{
		  MaxHit.hitenergy =  hhit->energy();
		  MaxHit.ietahit   = (hhit->id()).ieta();
		  MaxHit.iphihit   = (hhit->id()).iphi();
		  MaxHit.depthhit  = (hhit->id()).depth();
		}
	    }
	}
      MaxHhitEnergy = MaxHit.hitenergy;
      //LogInfo("iCoord of MaxHhit: ")<<" hhitenergy:"<<MaxHit.hitenergy<<"  ietahhit:"<<MaxHit.ietahit <<" iphihhit:"<<MaxHit.iphihit<<" depth:"<<MaxHit.depthhit;

//      cout <<""<<  " hhitenergy:"<<MaxHit.hitenergy<<"  ietahhit:"<<MaxHit.ietahit <<" iphihhit:"<<MaxHit.iphihit<<" depth:"<<MaxHit.depthhit <<endl;

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
	  // cout <<"MINIMAX "<<"MinIETA->"<<MinIETA<<"MaxIETA->"<<MaxIETA<<"MinIPHI->"<<MinIPHI<<"MaxIPHI->"<<MaxIPHI<<endl;
	  
	  sprintf(hname,"Track_%d",NisoTrk);
	  TH2F* h2 = new TH2F(hname,"IETA :: IPHI ",10,MinIETA-1,MaxIETA+1, 10,MinIPHI-1,MaxIPHI+1);
	  for (int k=0; k<nHCRecHits; k++) { h2->Fill( hcRHieta[k],hcRHiphi[k],hcRHen[k]);}
	  h2->Write();
	}

      if(AxB_=="3x3")
	{
	  HcalAxBxDepthEnergy=0.;
	  for(int ih=0; ih<9; ih++){HCAL3x3[ih]=0.;}

	  for (HBHERecHitCollection::const_iterator hhit=Hithbhe.begin(); hhit!=Hithbhe.end(); hhit++) 
	    {
	      
	      int DIETA = 100;
	      if(MaxHit.ietahit*(hhit->id()).ieta()>0)
		{
		  DIETA = MaxHit.ietahit - (hhit->id()).ieta();
		}
	      if(MaxHit.ietahit*(hhit->id()).ieta()<0)
		{
		  DIETA = MaxHit.ietahit - (hhit->id()).ieta();
		  DIETA = DIETA>0 ? DIETA-1 : DIETA+1; 
		}
	      
	      int DIPHI = MaxHit.iphihit - (hhit->id()).iphi();
	      DIPHI = DIPHI>36 ? 72-DIPHI : DIPHI;

	      if( abs(DIETA)<=1 && abs(DIPHI)<=1) 
		{
		  //HCAl Energy in 3x3 Clastor (sum of depth)
		  HcalAxBxDepthEnergy += hhit->energy();
		  
		  // LogInfo("3x3: ")<<" --- OK----"<<"  ieta->"<<(hhit->id()).ieta()<<"  iphi->"<<(hhit->id()).iphi()<<"  depth->"<<(hhit->id()).depth();
		  
		  // HCAL Energy in each RecHit of 3x3 (sum of depth) 
		  // HCAL3x3[9] -> 0  1  2
		  //               3  4  5
		  //               6  7  8
		  //
		  //  i       ^72
		  //  p       |
		  //  h       |
		  //  i       |
		  //  - - - - 0 - - - ->
		  //                  ieta
		  
		  if(DIPHI==-1  && DIETA== 1) {HCAL3x3[0] += hhit->energy();} 
		  if(DIPHI==-1  && DIETA== 0) {HCAL3x3[1] += hhit->energy();}
		  if(DIPHI==-1  && DIETA==-1) {HCAL3x3[2] += hhit->energy();}
		  if(DIPHI== 0  && DIETA== 1) {HCAL3x3[3] += hhit->energy();}
		  if(DIPHI== 0  && DIETA== 0) {HCAL3x3[4] += hhit->energy();}
		  if(DIPHI== 0  && DIETA==-1) {HCAL3x3[5] += hhit->energy();}
		  if(DIPHI== 1  && DIETA== 1) {HCAL3x3[6] += hhit->energy();}
		  if(DIPHI== 1  && DIETA== 0) {HCAL3x3[7] += hhit->energy();}
		  if(DIPHI== 1  && DIETA==-1) {HCAL3x3[8] += hhit->energy();}

// M+
// collect into rawEnergyVector:

//                   cout<<" rawEnergyVec "<< hhit->energy() << "  "<< hhit->id() <<  " ieta "<< (hhit->id()).ieta() <<  " iphi " << (hhit->id()).iphi()<< endl;

	   if(calEnergy > energyMinIso && calEnergy < energyMaxIso && econus < 1. && MaxHit.hitenergy > 0.)
	   {
		   rawEnergyVec.push_back(hhit->energy() * corrHCAL);
		   detidvec.push_back(hhit->id());

	    numbers2[(hhit->id()).ieta()+29][(hhit->id()).iphi()] = numbers2[(hhit->id()).ieta()+29][(hhit->id()).iphi()] + 1;
		   
		   }
// M-

		}
	    }
	}

      if(AxB_=="5x5")
	{
	  HcalAxBxDepthEnergy=0.;
	  for(int ih=0; ih<9; ih++){HCAL5x5[ih]=0.;}
	  
	  for (HBHERecHitCollection::const_iterator hhit=Hithbhe.begin(); hhit!=Hithbhe.end(); hhit++) 
	    {
	      int DIETA = 100;
	      if(MaxHit.ietahit*(hhit->id()).ieta()>0)	{DIETA = MaxHit.ietahit - (hhit->id()).ieta();}
	      if(MaxHit.ietahit*(hhit->id()).ieta()<0){ DIETA = MaxHit.ietahit - (hhit->id()).ieta();DIETA = DIETA>0 ? DIETA-1 : DIETA+1;}
	      int DIPHI = MaxHit.iphihit - (hhit->id()).iphi();
	      DIPHI = DIPHI>36 ? 72-DIPHI : DIPHI;
	      
	      if( abs(DIETA)<=2 && abs(DIPHI)<=2) 
		{
		
		  HcalAxBxDepthEnergy += hhit->energy();
		  
		  
		  // HCAL Energy in each RecHit of 5x5 (sum of depth) 
		  // HCAL3x3[25] ->  0  1  2  3  4
		  //                 5  6  7  9 10
		  //                11 12 13 14 15
		  //                16 17 18 19 20
		  //                21 22 23 24 25
		  if(DIPHI==-2  && DIETA== 2) {HCAL3x3[0] += hhit->energy();}
		  if(DIPHI==-2  && DIETA== 1) {HCAL3x3[1] += hhit->energy();}
		  if(DIPHI==-2  && DIETA== 0) {HCAL3x3[3] += hhit->energy();}
		  if(DIPHI==-2  && DIETA==-1) {HCAL3x3[4] += hhit->energy();}
		  if(DIPHI==-2  && DIETA==-2) {HCAL3x3[5] += hhit->energy();}

		  if(DIPHI==-1  && DIETA== 2) {HCAL3x3[6] += hhit->energy();}
		  if(DIPHI==-1  && DIETA== 1) {HCAL3x3[7] += hhit->energy();}
		  if(DIPHI==-1  && DIETA== 0) {HCAL3x3[8] += hhit->energy();}
		  if(DIPHI==-1  && DIETA==-1) {HCAL3x3[9] += hhit->energy();}
		  if(DIPHI==-1  && DIETA==-2) {HCAL3x3[10] += hhit->energy();}

		  if(DIPHI== 0  && DIETA== 2) {HCAL3x3[11] += hhit->energy();}
		  if(DIPHI== 0  && DIETA== 1) {HCAL3x3[12] += hhit->energy();}
		  if(DIPHI== 0  && DIETA== 0) {HCAL3x3[13] += hhit->energy();}
		  if(DIPHI== 0  && DIETA==-1) {HCAL3x3[14] += hhit->energy();}
		  if(DIPHI== 0  && DIETA==-2) {HCAL3x3[15] += hhit->energy();}
		  
		  if(DIPHI== 1  && DIETA== 2) {HCAL3x3[16] += hhit->energy();}
		  if(DIPHI== 1  && DIETA== 1) {HCAL3x3[17] += hhit->energy();}
		  if(DIPHI== 1  && DIETA== 0) {HCAL3x3[18] += hhit->energy();}
		  if(DIPHI== 1  && DIETA==-1) {HCAL3x3[19] += hhit->energy();}
		  if(DIPHI== 1  && DIETA==-2) {HCAL3x3[20] += hhit->energy();}
		  
		  if(DIPHI== 2  && DIETA== 2) {HCAL3x3[21] += hhit->energy();}
		  if(DIPHI== 2  && DIETA== 1) {HCAL3x3[22] += hhit->energy();}
		  if(DIPHI== 2  && DIETA== 0) {HCAL3x3[23] += hhit->energy();}
		  if(DIPHI== 2  && DIETA==-1) {HCAL3x3[24] += hhit->energy();}
		  if(DIPHI== 2  && DIETA==-2) {HCAL3x3[25] += hhit->energy();}

// M+
// collect into rawEnergyVector:
		   rawEnergyVec.push_back(hhit->energy());
		   detidvec.push_back(hhit->id());
// M-

		}	  
	    }
	}
      
      if(AxB_!="3x3" && AxB_!="5x5") LogWarning(" AxB ")<<"   Not supported: "<< AxB_;

// M+
	   if(calEnergy > energyMinIso && calEnergy < energyMaxIso && econus < 1. && 
	   (MaxHit.ietahit+29) > -1 && (MaxHit.ietahit+29) < 60 && MaxHit.hitenergy > 0.)
	   {
// 	    cout<<" Begin of pushing "<<endl; 
	    EventMatrix.push_back(rawEnergyVec);
	    EventIds.push_back(detidvec);
	    EnergyVector.push_back(calEnergy);

//	    cout<<"        "<< MaxHit.ietahit << "  " <<MaxHit.iphihit << endl;
//            cout<<"   "<< calEnergy << "  " << trackEta << " " << trackPhi<< endl; 
	    
	    numbers[MaxHit.ietahit+29][MaxHit.iphihit] = numbers[MaxHit.ietahit+29][MaxHit.iphihit] + 1;
	    
//	    cout<<" End of pushing "<<endl; 
//	    cout<<"   "<< calEnergy << "  " << rawEnergyVec.size() <<  "  " << detidvec.size()<< endl; 
	   }
// M-
     
//	    cout<<"  BEGINNING 2 "<< calEnergy << "  " << rawEnergyVec.size() <<  "  " << detidvec.size()<< endl; 
      
      nHORecHits=0;
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
	    
	    double dphi = fabs(trackPhi - phihit); 
	    if(dphi > 4.*atan(1.)) dphi = 8.*atan(1.) - dphi;
	    double deta = fabs(trackEta - etahit); 
	    double dr = sqrt(dphi*dphi + deta*deta);
	    
	    if(m_histoFlag==1)  {thDrTrHOHits -> Fill(dr);}
	    
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
      
      if(m_histoFlag==1)  {CalibTree -> Fill();}
    }
}

// ------------ method called once each job just before starting event loop  ------------
void 
HcalIsoTrkAnalyzer::beginJob(const edm::EventSetup&)
{
      tf2 = new TFile(outputFileName_.c_str(),"RECREATE");
      isoTrE = new TH1F("isoTrE","isoTrE", 200, 0., 100.);
      isoTrEta = new TH1F("isoTrEta","isoTrEta", 100, -2.1, 2.1);
      isoTrPhi = new TH1F("isoTrPhi","isoTrPhi", 72, -3.14, 3.14);

      isoEcalCone = new TH1F("isoEcalCone","isoEcalCone", 550, -1., 10.);

  if(m_histoFlag==1)  
    {
      
      thDrTrEHits = new TH1F("thDrTrEHits","thDrETrHits", 50, 0., 5.);
      thDrTrHBHEHits = new TH1F("thDrHBHETrHits","thDrHBHETrHits", 50, 0., 5.);
      thDrTrHOHits = new TH1F("thDrHOTrHits","thDrHOTrHits", 50, 0., 5.);

     
      CalibTree = new TTree("CalibTree","CalibTree");
      CalibTree->Branch("trackEta",&trackEta,"trackEta/D");
      CalibTree->Branch("trackPt",&trackPt,"trackPt/D");
      CalibTree->Branch("ptrack",&ptrack,"ptrack/D");
      CalibTree->Branch("rvert",&rvert,"rvert/D");
      
      CalibTree->Branch("MaxHhitEnergy",&MaxHhitEnergy,"MaxHhitEnergy/D");
      CalibTree->Branch("HcalAxBxDepthEnergy",&HcalAxBxDepthEnergy,"HcalAxBxDepthEnergy/D");
      CalibTree->Branch("HCAL3x3",HCAL3x3,"HCAL3x3[9]/D");
      CalibTree->Branch("HCAL5x5",HCAL5x5,"HCAL5x5[25]/D");

      CalibTree->Branch("nECRecHits",&nECRecHits,"nECRecHits/I");//, "ECAL Number of RecHits"
      CalibTree->Branch("ecRHen",ecRHen,"ecRHen[nECRecHits]/D");//, "ECAL RecHits Energy"
      CalibTree->Branch("ecRHeta",ecRHeta,"ecRHeta[nECRecHits]/D");//, "ECAL RecHits eta"
      CalibTree->Branch("ecRHphi",ecRHphi,"ecRHphi[nECRecHits]/D");//, "ECAL RecHits phi"
      CalibTree->Branch("eecal",&eecal,"eecal/D");//, "ECAL Cone Energy"
      
      CalibTree->Branch("nHCRecHits",&nHCRecHits,"nHCRecHits/I");//,"HCAL Number of RecHits"
      CalibTree->Branch("hcRHen",hcRHen,"hcRHen[nHCRecHits]/D");//,"HCAL RecHits Energy"
      CalibTree->Branch("hcRHeta",hcRHeta,"hcRHeta[nHCRecHits]/D");//,"HCAL RecHits eta"
      CalibTree->Branch("hcRHphi",hcRHphi,"hcRHphi[nHCRecHits]/D");//,"HCAL RecHits phi"
      CalibTree->Branch("hcRHieta",hcRHieta,"hcRHieta[nHCRecHits]/I");//, "HCAL RecHits ieta"
      CalibTree->Branch("hcRHiphi",hcRHiphi,"hcRHiphi[nHCRecHits]/I");//, "HCAL RecHits iphi"
      CalibTree->Branch("hcRHdepth",hcRHdepth,"hcRHdepth[nHCRecHits]/I");//,"HCAL RecHits depth"
      CalibTree->Branch("ehcal",&ehcal,"ehcal/D");//, "HCAL Cone Energy"
      
      CalibTree->Branch("nHORecHits",&nHORecHits,"nHORecHits/I");
      CalibTree->Branch("hoRHen",hoRHen,"hoRHen[nHORecHits]/D");
      CalibTree->Branch("hoRHeta",hoRHeta,"hoRHeta[nHORecHits]/D");
      CalibTree->Branch("hoRHphi",hoRHphi,"hoRHphi[nHORecHits]/D");
      CalibTree->Branch("hoRHieta",hoRHieta,"hoRHieta[nHORecHits]/I");
      CalibTree->Branch("hoRHiphi",hoRHiphi,"hoRHiphi[nHORecHits]/I");
      CalibTree->Branch("hoRHdepth",hoRHdepth,"hoRHdepth[nHORecHits]/I"); 
    }
  NisoTrk=0;

// M+

// initialize the algorithm
  MyL3Algo = new MinL3AlgoUniv<HcalDetId>(eventWeight);

// M-

}

// ------------ method called once each job just after ending the event loop  ------------
void 
HcalIsoTrkAnalyzer::endJob() {

// M+
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

///      layers->Fill(curr_depth);

///      calibs->Fill(ii->second);
      // translate into eta, phi and fill into map

///      if (curr_depth == 1) calibMapL1->Fill(curr_eta,curr_phi,ii->second);
///      if (curr_depth == 2) calibMapL2->Fill(curr_eta,curr_phi,ii->second);
///      if (curr_depth == 3) calibMapL3->Fill(curr_eta,curr_phi,ii->second);
///      if (ii->first.subdet() == HcalOuter) calibMapHO->Fill(curr_eta,curr_phi,ii->second);

      //depth definition:
      //enum HcalSubdetector { HcalEmpty=0, HcalBarrel=1, HcalEndcap=2, HcalOuter=3, HcalForward=4, HcalTriggerTower=5, HcalOther=7 };
    }

  for (ii = solution.begin(); ii != solution.end(); ii++)
    {
      int curr_eta = ii->first.ieta();
      int curr_phi = ii->first.iphi();
      int curr_depth = ii->first.depth();
      cout << "  "<< ii->first.subdet() << "  " << curr_depth  << "  "<< curr_eta << "  " << curr_phi <<  "  "<< ii->second
      << endl; 
    }



// M-

      tf2 -> cd();

      isoTrE -> Write();      
      isoTrEta -> Write();      
      isoTrPhi -> Write();      
      isoEcalCone-> Write(); 
  
  if(m_histoFlag==1) 
    {
      
      thDrTrEHits -> Write();  
      thDrTrHBHEHits -> Write();
      thDrTrHOHits -> Write(); 

      CalibTree -> Write();
    }
}

//define this as a plug-in
DEFINE_FWK_MODULE(HcalIsoTrkAnalyzer);

