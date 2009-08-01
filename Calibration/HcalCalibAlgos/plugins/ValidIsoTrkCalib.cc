//
// Package:    ValidIsoTrkCalib
// Class:      ValidIsoTrkCalib
// 

/*
 Description: Validation for Isolated tracks Calibration
 
 Implementation:
See my page on twiki:
https://twiki.cern.ch/twiki/bin/view/Sandbox/ValidIsoTrkCalib

*/

//
// Original Author:  Andrey Pozdnyakov
//         Created:  Tue Nov  4 01:16:05 CET 2008
// $Id: ValidIsoTrkCalib.cc,v 1.31 2009/07/14 07:44:58 andrey Exp $
//

// system include files

#include <memory>


// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "TrackingTools/TrackAssociator/interface/TrackDetectorAssociator.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "DataFormats/HcalIsolatedTrack/interface/IsolatedPixelTrackCandidate.h"
#include "DataFormats/HcalIsolatedTrack/interface/IsolatedPixelTrackCandidateFwd.h"

#include "DataFormats/HcalDetId/interface/HcalDetId.h"

#include "TROOT.h"
#include "TFile.h"
#include "TTree.h"

#include <fstream>
#include <map>

using namespace edm;
using namespace std;
using namespace reco;

class ValidIsoTrkCalib : public edm::EDAnalyzer {
public:
  explicit ValidIsoTrkCalib(const edm::ParameterSet&);
  ~ValidIsoTrkCalib();
 
  
private:
  virtual void beginJob(const edm::EventSetup&) ;
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void endJob() ;

 
  
  // ----------member data ---------------------------
  
  //Variables from HcalIsotrackAnalyzer

  TrackDetectorAssociator trackAssociator_;
  TrackAssociatorParameters parameters_;
  double taECALCone_;
  double taHCALCone_;

  const CaloGeometry* geo;
  InputTag genTracksLabel_;
  InputTag genhbheLabel_;
  InputTag genhoLabel_;
  std::vector<edm::InputTag> genecalLabel_;
  InputTag hbheLabel_;
  InputTag hoLabel_;
  InputTag trackLabel_;
  InputTag trackLabel1_;

  //std::string m_inputTrackLabel;
  //std::string m_hcalLabel;

  double associationConeSize_;
  string AxB_;
  bool allowMissingInputs_;
  string outputFileName_;
  string calibFactorsFileName_;
  //  string corrfile;
  
  bool takeAllRecHits_, takeGenTracks_;
  int gen, iso, pix;
  float genPt[500], genPhi[500], genEta[500];
  float isoPt[500], isoPhi[500], isoEta[500];
  float pixPt[500], pixPhi[500], pixEta[500];
  int  hbheiEta[5000],hbheiPhi[5000],hbheDepth[5000];	 
  float hbheEnergy[5000];  


  int NisoTrk;
  float trackPt, trackE, trackEta, trackPhi;
  float ptNear;
  float ptrack, rvert;
  //float eecal, ehcal;

  int MinNTrackHitsBarrel, MinNTECHitsEndcap;
  double energyECALmip, maxPNear;
  double energyMinIso, energyMaxIso;


  Float_t emEnergy;
  Float_t targetE;

  TFile* rootFile;
  //  TTree* tree;
  TTree *tTree, *fTree;

  //Variables from mooney validator
  int Nhits;
  float eClustBefore;  //Calo energy before calibration
  float eClustAfter;   //After calibration
  float eTrack;      //Track energy
  float etaTrack;     
  float phiTrack;      
  float eECAL;      // Energy deposited in ECAL
  int numHits;       //number of rechits
  //float e3x3;

  float eBeforeDepth1;
  float eAfterDepth1;
  float eBeforeDepth2;
  float eAfterDepth2;
  float eCentHitBefore;
  float eCentHitAfter;
  float CentHitFactor;
  int iEta;
  int iPhi;
  int iEtaTr;
  int iPhiTr;
  float iDr;
  int dietatr;
  int diphitr;


  float iTime;
  float HTime[100];
  float e3x3Before;
  float e3x3After;
  float e5x5Before;
  float e5x5After;
  int eventNumber;
  int runNumber;
  float PtNearBy;
  float numVH, numVS, numValidTrkHits, numValidTrkStrips;

  map<UInt_t, Float_t> CalibFactors; 
  Bool_t ReadCalibFactors(string);

  
};



ValidIsoTrkCalib::ValidIsoTrkCalib(const edm::ParameterSet& iConfig)
  
{

  takeAllRecHits_=iConfig.getUntrackedParameter<bool>("takeAllRecHits");
  takeGenTracks_=iConfig.getUntrackedParameter<bool>("takeGenTracks");

  genTracksLabel_ = iConfig.getParameter<edm::InputTag>("genTracksLabel");
  genhbheLabel_= iConfig.getParameter<edm::InputTag>("genHBHE");
  //genhoLabel_=iConfig.getParameter<edm::InputTag>("genHO");
  //genecalLabel_=iConfig.getParameter<std::vector<edm::InputTag> >("genECAL");

  // m_hcalLabel = iConfig.getUntrackedParameter<std::string> ("hcalRecHitsLabel","hbhereco");

  hbheLabel_= iConfig.getParameter<edm::InputTag>("hbheInput");
  hoLabel_=iConfig.getParameter<edm::InputTag>("hoInput");
  //eLabel_=iConfig.getParameter<edm::InputTag>("eInput");
  trackLabel_ = iConfig.getParameter<edm::InputTag>("HcalIsolTrackInput");
  trackLabel1_ = iConfig.getParameter<edm::InputTag>("trackInput");

  associationConeSize_=iConfig.getParameter<double>("associationConeSize");
  allowMissingInputs_=iConfig.getUntrackedParameter<bool>("allowMissingInputs", true);
  outputFileName_=iConfig.getParameter<std::string>("outputFileName");
  calibFactorsFileName_=iConfig.getParameter<std::string>("calibFactorsFileName");


  AxB_=iConfig.getParameter<std::string>("AxB");

  MinNTrackHitsBarrel =  iConfig.getParameter<int>("MinNTrackHitsBarrel");
  MinNTECHitsEndcap =  iConfig.getParameter<int>("MinNTECHitsEndcap");
  energyECALmip = iConfig.getParameter<double>("energyECALmip");
  energyMinIso = iConfig.getParameter<double>("energyMinIso");
  energyMaxIso = iConfig.getParameter<double>("energyMaxIso");
  maxPNear = iConfig.getParameter<double>("maxPNear");

  edm::ParameterSet parameters = iConfig.getParameter<edm::ParameterSet>("TrackAssociatorParameters");
  parameters_.loadParameters( parameters );
  trackAssociator_.useDefaultPropagator();

  taECALCone_=iConfig.getUntrackedParameter<double>("TrackAssociatorECALCone",0.5);
  taHCALCone_=iConfig.getUntrackedParameter<double>("TrackAssociatorHCALCone",0.6);

}


ValidIsoTrkCalib::~ValidIsoTrkCalib()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


// ------------ method called to for each event  ------------
void
ValidIsoTrkCalib::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;
   

   edm::Handle<reco::TrackCollection> generalTracks;
   iEvent.getByLabel(genTracksLabel_, generalTracks);

  edm::Handle<reco::TrackCollection> isoProdTracks;
  iEvent.getByLabel(trackLabel1_,isoProdTracks);


  edm::Handle<reco::IsolatedPixelTrackCandidateCollection> pixelTracks;
  //edm::Handle<reco::TrackCollection> pixelTracks;
  iEvent.getByLabel(trackLabel_,pixelTracks);
  
  /*
  edm::Handle<EcalRecHitCollection> ecal;
  iEvent.getByLabel(eLabel_,ecal);
  const EcalRecHitCollection Hitecal = *(ecal.product());
  */

  edm::Handle<HBHERecHitCollection> hbhe;
  iEvent.getByLabel(hbheLabel_,hbhe);
  const HBHERecHitCollection Hithbhe = *(hbhe.product());


  edm::Handle<HBHERecHitCollection> genhbhe;
  iEvent.getByLabel(genhbheLabel_,genhbhe);
  const HBHERecHitCollection genHithbhe = *(genhbhe.product());

  edm::ESHandle<CaloGeometry> pG;
  iSetup.get<CaloGeometryRecord>().get(pG);
  geo = pG.product();

  parameters_.useEcal = true;
  parameters_.useHcal = true;
  parameters_.useCalo = false;
  parameters_.useMuon = false;
  parameters_.dREcal = taECALCone_;
  parameters_.dRHcal = taHCALCone_;


  //cout<<"Hello World. TrackCollectionSize: "<< pixelTracks->size()<<endl;

  //cout<<" generalTracks Size: "<< generalTracks->size()<<endl;
 
  if(takeGenTracks_ && iEvent.id().event()%10==1)
    {
      gen = generalTracks->size();
      iso = isoProdTracks->size();
      pix = pixelTracks->size();
      
      genPt[0] = -33;     	  
      genPhi[0] = -33;     	  
      genEta[0] = -33;     	  
      
      isoPt[0] = -33;     	  
      isoPhi[0] = -33;     	  
      isoEta[0] = -33;     	  
      
      pixPt[0] = -33;     	  
      pixPhi[0] = -33;     	  
      pixEta[0] = -33;     	  
      
      Int_t gencount=0, isocount=0, pixcount=0;
      for (reco::TrackCollection::const_iterator gentr=generalTracks->begin(); gentr!=generalTracks->end(); gentr++)
	{
	  genPt[gencount] = gentr->pt();     	  
	  genPhi[gencount] = gentr->phi();     	  
	  genEta[gencount] = gentr->eta();     	  
	  gencount++;
	}
      
      for (reco::TrackCollection::const_iterator isotr=isoProdTracks->begin(); isotr!=isoProdTracks->end(); isotr++)
	{
	  isoPt[isocount] = isotr->pt();     	  
	  isoPhi[isocount] = isotr->phi();     	  
	  isoEta[isocount] = isotr->eta();     	  
	  isocount++;
	}
      
      for (reco::IsolatedPixelTrackCandidateCollection::const_iterator pixtr=pixelTracks->begin(); pixtr!=pixelTracks->end(); pixtr++)
	{
	  pixPt[pixcount] = pixtr->pt();     	  
	  pixPhi[pixcount] = pixtr->phi();     	  
	  pixEta[pixcount] = pixtr->eta();     	  
	  pixcount++;
	}
    }

  if(takeAllRecHits_ && iEvent.id().event()%500==1)
    {
      Nhits=0;
      for (HBHERecHitCollection::const_iterator hhit=genHithbhe.begin(); hhit!=genHithbhe.end(); hhit++) 
	{
	  hbheiEta[Nhits] = hhit->id().ieta();	  
	  hbheiPhi[Nhits] = hhit->id().iphi();	 
	  hbheDepth[Nhits] = hhit->id().depth();	 
	  hbheEnergy[Nhits] = hhit->energy();
	  Nhits++;
	  
	}
    }    
  
  tTree -> Fill();

if (pixelTracks->size()==0) return;
  

for (reco::TrackCollection::const_iterator trit=isoProdTracks->begin(); trit!=isoProdTracks->end(); trit++)
    {
      
      reco::IsolatedPixelTrackCandidateCollection::const_iterator isoMatched=pixelTracks->begin();
      //reco::TrackCollection::const_iterator isoMatched=pixelTracks->begin();
      bool matched=false;
 
   //for (reco::IsolatedPixelTrackCandidateCollection::const_iterator trit = pixelTracks->begin(); trit!=pixelTracks->end(); trit++)
   for (reco::IsolatedPixelTrackCandidateCollection::const_iterator it = pixelTracks->begin(); it!=pixelTracks->end(); it++)
   //for (reco::TrackCollection::const_iterator it = pixelTracks->begin(); it!=pixelTracks->end(); it++)
   { 
	  //if (floor(100000*trit->pt())==floor(100000*it->pt())) 
	  if (abs((trit->pt() - it->pt())/it->pt()) < 0.005 && abs(trit->eta() - it->eta()) < 0.01) 
	   {
	     isoMatched=it;
	     matched=true;
	     break;
	   }
	}
      // CUT

   if (!matched) continue;
   if(isoMatched->maxPtPxl() > maxPNear) continue;
    
   ptNear = isoMatched->maxPtPxl();      
   //cout<<"Point 0.1  isoMatch. ptnear: "<<ptNear<<endl;

      
      // CUT
      if (trit->hitPattern().numberOfValidHits()<MinNTrackHitsBarrel) continue;
      if (fabs(trit->eta())>1.47&&trit->hitPattern().numberOfValidStripTECHits()<MinNTECHitsEndcap) continue;

      //cout<<"Point 0.2.1 after numofvalidhits HB: "<<trit->hitPattern().numberOfValidHits()<<endl;
      //cout<<"Point 0.2.2 after numofvalidstrips HE: "<<trit->hitPattern().numberOfValidStripTECHits()<<endl;

  numVH = trit->hitPattern().numberOfValidHits();
  numVS = trit->hitPattern().numberOfValidStripTECHits();
      
    
      
      trackE = sqrt(trit->px()*trit->px()+trit->py()*trit->py()+trit->pz()*trit->pz()+0.14*0.14);
      trackPt = trit->pt();
      trackEta = trit->eta();
      trackPhi = trit->phi();

      emEnergy = isoMatched->energyIn();

      TrackDetMatchInfo info = trackAssociator_.associate(iEvent, iSetup,trackAssociator_.getFreeTrajectoryState(iSetup, *trit), parameters_);
      
      //float etaecal=info.trkGlobPosAtEcal.eta();
      //float phiecal=info.trkGlobPosAtEcal.phi();
      

      float etahcal=info.trkGlobPosAtHcal.eta();
      float phihcal=info.trkGlobPosAtHcal.phi();

      //cout<<"Point 0.3.  Matched :: pt: "<<trit->pt()<<" wholeEnergy: "<<trackE<<"  emEnergy: "<<emEnergy<<"  eta: "<<etahcal<<" phi: "<<phihcal<<endl;
      //cout<<"Point 0.4.  EM energy in cone: "<<emEnergy<<"  EtaHcal: "<<etahcal<<"  PhiHcal: "<<phihcal<<endl;
      /*
	float dphi = fabs(info.trkGlobPosAtHcal.phi() - phihit); 
	if(dphi > 4.*atan(1.)) dphi = 8.*atan(1.) - dphi;
	float deta = fabs(info.trkGlobPosAtHcal.eta() - etahit); 
	float dr = sqrt(dphi*dphi + deta*deta);
      */


      struct
      {
	int iphihitm;
	int ietahitm;
	int depthhit;
	float hitenergy;
	float dr;
      } MaxHit;

      MaxHit.hitenergy=-100.;

      //container for used recHits
      std::vector<int> usedHits; 
      //  
      usedHits.clear();
      //cout <<"Point 1. Entrance to HBHECollection"<<endl;

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
	  float phihit = pos.phi();
	  float etahit = pos.eta();
	  
	  int iphihitm  = (hhit->id()).iphi();
	  int ietahitm  = (hhit->id()).ieta();
	  int depthhit = (hhit->id()).depth();
	  float enehit = hhit->energy() * recal;
	  
	  if (depthhit!=1) continue;
	   
	  float dphi = fabs(phihcal - phihit); 
	  if(dphi > 4.*atan(1.)) dphi = 8.*atan(1.) - dphi;
	  float deta = fabs(etahcal - etahit); 
	  float dr = sqrt(dphi*dphi + deta*deta);

	  if (deta<dddeta) {
	   ietatrue = ietahitm;
	   dddeta=deta;
	  }
	  
	  if (dphi<dddphi) {
	  iphitrue = iphihitm;
	  dddphi=dphi;
	  }

	  //  cout<<"ieta: "<<ietahitm<<"  iphi: "<<iphihitm<<"  depthhit: "<<depthhit<<"  dr: "<<dr<<endl;    
	  
	  if(dr<associationConeSize_) 
	    {
	      
	      for (HBHERecHitCollection::const_iterator hhit2=Hithbhe.begin(); hhit2!=Hithbhe.end(); hhit2++) 
		{
		  int iphihitm2  = (hhit2->id()).iphi();
		  int ietahitm2  = (hhit2->id()).ieta();
		  int depthhit2 = (hhit2->id()).depth();
		  float enehit2 = hhit2->energy() * recal;
		  
		  if (iphihitm==iphihitm2 && ietahitm==ietahitm2  && depthhit!=depthhit2)  enehit = enehit+enehit2;
		
		}	  	  
	      

	      //cout<<"IN CONE ieta: "<<ietahitm<<"  iphi: "<<iphihitm<<" depthhit: "<<depthhit<<"  dr: "<<dr<<" energy: "<<enehit<<endl;        

	      //Find a Hit with Maximum Energy
	      
	      if(enehit > MaxHit.hitenergy) 
		{
 		  MaxHit.hitenergy =  enehit;
		  MaxHit.ietahitm   = (hhit->id()).ieta();
		  MaxHit.iphihitm   = (hhit->id()).iphi();
		  MaxHit.dr   = dr;
		  //MaxHit.depthhit  = (hhit->id()).depth();
		  MaxHit.depthhit  = 1;
		}

	    }
	} //end of all HBHE hits cycle
      
      usedHits.clear();
      
      //cout<<"Point 3.  MaxHit :::En "<<MaxHit.hitenergy<<"  ieta: "<<MaxHit.ietahitm<<"  iphi: "<<MaxHit.iphihitm<<endl;
      
      Bool_t passCuts = kFALSE;
      if(trackE > energyMinIso && trackE < energyMaxIso && emEnergy < energyECALmip && MaxHit.hitenergy > 0. && abs(MaxHit.ietahitm)<29) passCuts = kTRUE;
      
      //cout<<"Pont 0.1.1.  trackE:"<<trackE<<"  emEn: "<<emEnergy<<endl;
      

      numHits=0;
      
      eClustBefore = 0.0;
      eClustAfter = 0.0;
      eBeforeDepth1 = 0.0;
      eAfterDepth1 = 0.0;
      eBeforeDepth2 = 0.0;    
      eAfterDepth2 = 0.0;
      CentHitFactor = 0.0; 
      e3x3After = 0.0;
      e3x3Before = 0.0;
      e5x5After = 0.0;
      e5x5Before = 0.0;

		  
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
	  


	  int numbercell=2; //always collect 5x5 clastor!

	  //if(AxB_=="3x3") numbercell = 1;
	  //if(AxB_=="5x5") numbercell = 2;
	  
	  if( abs(DIETA)<=numbercell && (abs(DIPHI)<=numbercell || ( abs(MaxHit.ietahitm)>20 && abs(DIPHI)<=numbercell+1)) )
	    {
	            
	      if(passCuts && hhit->energy()>0)
		{
		  
		  
		  float factor = 0.0;	  
		  factor = CalibFactors[hhit->id()];
		  
		  
		  //if(i<5){cout<<" calib factors: "<<factor<<"  ij "<<100*i+j<<endl;}
		  
		  if (hhit->id().ieta() == MaxHit.ietahitm && hhit->id().iphi() == MaxHit.iphihitm) CentHitFactor=CalibFactors[hhit->id()];	  
		  
		  if (hhit->id().ieta() == MaxHit.ietahitm && hhit->id().iphi() == MaxHit.iphihitm) iTime = hhit->time();	  
		  
		  if(AxB_!="3x3" && AxB_!="5x5") LogWarning(" AxB ")<<"   Not supported: "<< AxB_;
		  
		  e5x5Before += hhit->energy();
		  e5x5After += hhit->energy()*factor;
		  
		  
		  if(AxB_=="5x5") 
		    {
		      
		      HTime[numHits]=  hhit->time();
		      numHits++;
		      
		      
		      eClustBefore += hhit->energy();
		      eClustAfter += hhit->energy()*factor;
		      
		      
		      if (abs(DIETA)<=1 && (abs(DIPHI)<=1 || (abs(MaxHit.ietahitm)>20 && abs(DIPHI)<=2)) )
			{
			  e3x3Before += hhit->energy();
			  e3x3After += hhit->energy()*factor;
			}
		      
		      
		      if ((hhit->id().depth() == 1) && (abs(hhit->id().ieta()) > 17) && (abs(hhit->id().ieta()) < 29))
			{	
			  eBeforeDepth1 += hhit->energy();
			  eAfterDepth1 += hhit->energy()*factor;
			}
		      else if((hhit->id().depth() == 2) && (abs(hhit->id().ieta()) > 17) && (abs(hhit->id().ieta()) < 29))
			{
			  eBeforeDepth2 += hhit->energy();
			  eAfterDepth2 += hhit->energy()*factor;
			}
		    }//end of 5x5
		  
		  
		  
		  if(AxB_=="3x3") 
		    {
		      if (abs(DIETA)<=1 && (abs(DIPHI)<=1 || ( abs(MaxHit.ietahitm)>20 && abs(DIPHI)<=2) ) )
			{
			  HTime[numHits]=  hhit->time();
			  numHits++;			      
			  
			  
			  eClustBefore += hhit->energy();
			  eClustAfter += hhit->energy() * factor;
			  
			  e3x3Before += hhit->energy();
			  e3x3After += hhit->energy()*factor;
			  
			  if ((hhit->id().depth() == 1) && (abs(hhit->id().ieta()) > 17) && (abs(hhit->id().ieta()) < 29))
			    {	
			      eBeforeDepth1 += hhit->energy();
			      eAfterDepth1 += hhit->energy() * factor;
			    }
			  else if((hhit->id().depth() == 2) && (abs(hhit->id().ieta()) > 17) && (abs(hhit->id().ieta()) < 29))
			    {
			      eBeforeDepth2 += hhit->energy();
			      eAfterDepth2 += hhit->energy() * factor;
			    }
			  
			}
		    }//end of 3x3
		
		}//end of passCuts
	      
	    }//end of DIETA DIPHI
	  
	} //end of associatedcone HBHE hits cycle
      
      
      int dieta_M_P = 100;
      int diphi_M_P = 100;
      if(MaxHit.ietahitm*ietatrue>0) {dieta_M_P = abs (MaxHit.ietahitm-ietatrue);}
      if(MaxHit.ietahitm*ietatrue<0) {dieta_M_P = abs(MaxHit.ietahitm-ietatrue)-1;}
      diphi_M_P = abs(MaxHit.iphihitm-iphitrue);
      diphi_M_P =  diphi_M_P>36 ? 72-diphi_M_P : diphi_M_P; 
      
      
      if(passCuts)
	
	{
	  eventNumber = iEvent.id().event();
	  runNumber = iEvent.id().run();
	  
	  eCentHitBefore = MaxHit.hitenergy;
	  eCentHitAfter = MaxHit.hitenergy*CentHitFactor;
	  eECAL = emEnergy;
	  numValidTrkHits = numVH;
	  numValidTrkStrips = numVS;
	  PtNearBy = ptNear;
	  
	  
	  eTrack = trackE;
	  phiTrack = trackPhi;
	  etaTrack = trackEta;
	  
	  iEta = MaxHit.ietahitm;
	  iPhi = MaxHit.iphihitm;
	  
	  iEtaTr = ietatrue;
	  iPhiTr = iphitrue;
	  iDr = MaxHit.dr;
	  dietatr = dieta_M_P;
	  diphitr = diphi_M_P;
	  
	  fTree -> Fill();
	}

    } //end of isoProdTracks cycle

}


// ------------ method called once each job just before starting event loop  ------------
void 
ValidIsoTrkCalib::beginJob(const edm::EventSetup&)
{

 if(!ReadCalibFactors(calibFactorsFileName_.c_str() )) {cout<<"Cant read file with cailib coefficients!! ---"<<endl;}

  rootFile = new TFile(outputFileName_.c_str(),"RECREATE");
  
  tTree = new TTree("tTree", "Tree for gen info"); 

  if(takeGenTracks_) { 
    tTree->Branch("gen", &gen, "gen/I");
    tTree->Branch("iso", &iso, "iso/I");
    tTree->Branch("pix", &pix, "pix/I");
    tTree->Branch("genPt", genPt, "genPt[gen]/F");
    tTree->Branch("genPhi", genPhi, "genPhi[gen]/F");
    tTree->Branch("genEta", genEta, "genEta[gen]/F");
    
    tTree->Branch("isoPt", isoPt, "isoPt[iso]/F");
    tTree->Branch("isoPhi", isoPhi, "isoPhi[iso]/F");
    tTree->Branch("isoEta", isoEta, "isoEta[iso]/F");
    
    tTree->Branch("pixPt", pixPt, "pixPt[pix]/F");
    tTree->Branch("pixPhi", pixPhi, "pixPhi[pix]/F");
    tTree->Branch("pixEta", pixEta, "pixEta[pix]/F");
  }
  if(takeAllRecHits_) {
    tTree->Branch("Nhits", &Nhits, "Nhits/I");
    tTree->Branch("hbheiEta", hbheiEta, "hbheiEta[Nhits]/I");
    tTree->Branch("hbheiPhi", hbheiPhi, "hbheiPhi[Nhits]/I");
    tTree->Branch("hbheDepth", hbheDepth, "hbheDepth[Nhits]/I");
    tTree->Branch("hbheEnergy", hbheEnergy, "hbheEnergy[Nhits]/F");
  }

  fTree = new TTree("fTree", "Tree for IsoTrack Calibration"); 
  
  fTree->Branch("eventNumber", &eventNumber, "eventNumber/I");
  fTree->Branch("runNumber", &runNumber, "runNumber/I");

  fTree->Branch("eClustBefore", &eClustBefore,"eClustBefore/F");
  fTree->Branch("eClustAfter", &eClustAfter,"eClustAfter/F");
  fTree->Branch("eTrack", &eTrack, "eTrack/F");
  fTree->Branch("etaTrack", &etaTrack, "etaTrack/F");
  fTree->Branch("phiTrack", &phiTrack, "phiTrack/F");
    
  fTree->Branch("numHits", &numHits, "numHits/I");
  fTree->Branch("eECAL", &eECAL, "eECAL/F");
  fTree->Branch("PtNearBy", &PtNearBy, "PtNearBy/F");
  fTree->Branch("numValidTrkHits", &numValidTrkHits, "numValidTrkHits/F");
  fTree->Branch("numValidTrkStrips", &numValidTrkStrips, "numValidTrkStrips/F");
  
  fTree->Branch("eBeforeDepth1", &eBeforeDepth1, "eBeforeDepth1/F");
  fTree->Branch("eBeforeDepth2", &eBeforeDepth2, "eBeforeDepth2/F");
  fTree->Branch("eAfterDepth1", &eAfterDepth1, "eAfterDepth1/F");
  fTree->Branch("eAfterDepth2", &eAfterDepth2, "eAfterDepth2/F");
  
  fTree->Branch("e3x3Before", &e3x3Before, "e3x3Before/F");
  fTree->Branch("e3x3After", &e3x3After, "e3x3After/F");
  fTree->Branch("e5x5Before", &e5x5Before, "e5x5Before/F");
  fTree->Branch("e5x5After", &e5x5After, "e5x5After/F");
  
  fTree->Branch("eCentHitAfter", &eCentHitAfter, "eCentHitAfter/F");
  fTree->Branch("eCentHitBefore", &eCentHitBefore, "eCentHitBefore/F");
  fTree->Branch("iEta", &iEta, "iEta/I");
  fTree->Branch("iPhi", &iPhi, "iPhi/I");


  fTree->Branch("iEtaTr", &iEtaTr, "iEtaTr/I");
  fTree->Branch("iPhiTr", &iPhiTr, "iPhiTr/I");
  fTree->Branch("dietatr", &dietatr, "dietatr/I");
  fTree->Branch("diphitr", &diphitr, "diphitr/I");
  fTree->Branch("iDr", &iDr, "iDr/F");

  fTree->Branch("iTime", &iTime, "iTime/F");
  fTree->Branch("HTime", HTime, "HTime[numHits]/F");


}

// ------------ method called once each job just after ending the event loop  ------------
void 
ValidIsoTrkCalib::endJob() 
{
  rootFile->Write();
  rootFile->Close();
}



//Bool_t ValidIsoTrkCalib::ReadCalibFactors(map<UInt_t, Float_t> CalibFactorsMap, string FileName) {

 Bool_t ValidIsoTrkCalib::ReadCalibFactors(string FileName) {

  //  ifstream CalibFile(calibFactorsFileName_.c_str());

 edm::FileInPath pathto_calibFactorsFileName_(FileName);
 // edm::FileInPath pathto_calibFactorsFileName_( calibFactorsFileName_.c_str() );

 cout<<"  ---- Path to calib.txt FULL:"<<pathto_calibFactorsFileName_.fullPath().c_str() <<endl;

 ifstream CalibFile(pathto_calibFactorsFileName_.fullPath().c_str());


  if (!CalibFile) {
    //cout << "\nERROR: Can not find file with phi symmetry constants " <<  PHI_SYM_COR_FILENAME.Data() << endl;
    cout << "Can't find Calib Corrections File!! ---\n" << endl;
    return kFALSE;
  }

  // assumes the format used in CSA08, first line is a comment

  Int_t   iEta;
  UInt_t  iPhi;
  UInt_t  depth;
  TString sdName;
  UInt_t  detId;

  Float_t value;
  HcalSubdetector sd;

  std::string line;

  while (getline(CalibFile, line)) {

    if(!line.size() || line[0]=='#') continue;


    std::istringstream linestream(line);
    linestream >> iEta >> iPhi >> depth >> sdName >> value >> hex >> detId;


    if (sdName=="HB") sd = HcalBarrel;
    else if (sdName=="HE") sd = HcalEndcap;
    else if (sdName=="HO") sd = HcalOuter;
    else if (sdName=="HF") sd = HcalForward;
    else {
      cout << "\nInvalid detector name in phi symmetri constants file: " << sdName.Data() << endl;
      cout << "Check file and rerun!\n" << endl;
      return kFALSE;
    }

    // check if the data is consistent    
  
    if (HcalDetId(sd, iEta, iPhi, depth) != HcalDetId(detId)) {
      cout << "\nInconsistent info in phi symmetry file: subdet, iEta, iPhi, depth do not match rawId!\n" << endl;
      return kFALSE;    
    }
   
    if (!(HcalDetId::validDetId(sd, iEta, iPhi, depth))) {
      cout << "\nInvalid DetId from: iEta=" << iEta << " iPhi=" << iPhi << " depth=" << depth 
	   << " subdet=" << sdName.Data() << " detId=" << detId << endl << endl; 
      return kFALSE;    
    }
    

    //CalibFactorsMap[HcalDetId(sd, iEta, iPhi, depth)] = value;
    CalibFactors[HcalDetId(sd, iEta, iPhi, depth)] = value;
    //    if (iEta==1){cout<<CalibFactorsMap[HcalDetId(sd, iEta, iPhi, depth)]<<endl;}
  }

  return kTRUE;
}





//define this as a plug-in
DEFINE_FWK_MODULE(ValidIsoTrkCalib);
