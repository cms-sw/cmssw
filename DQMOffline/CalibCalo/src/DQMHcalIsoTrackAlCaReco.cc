// -*- C++ -*-
//
// Package:    DQMOffline/CalibCalo
// Class:      DQMHcalIsoTrackAlCaReco
// 
/**\class DQMHcalIsoTrackAlCaReco DQMHcalIsoTrackAlCaReco.cc DQMOffline/CalibCalo/src/DQMHcalIsoTrackAlCaReco.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Grigory SAFRONOV
//         Created:  Tue Oct  14 16:10:31 CEST 2008
//
//


// system include files
#include <memory>

// user include files

#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/HLTReco/interface/TriggerEvent.h"
#include "DataFormats/L1Trigger/interface/L1JetParticle.h"
#include "DataFormats/L1Trigger/interface/L1JetParticleFwd.h"

#include "CondFormats/L1TObjects/interface/L1GtTriggerMenu.h"
#include "CondFormats/DataRecord/interface/L1GtTriggerMenuRcd.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutSetupFwd.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutSetup.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutRecord.h"

#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"

#include "CondFormats/L1TObjects/interface/L1GtPrescaleFactors.h"
#include "CondFormats/DataRecord/interface/L1GtPrescaleFactorsAlgoTrigRcd.h"
#include "CondFormats/DataRecord/interface/L1GtPrescaleFactorsTechTrigRcd.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "DataFormats/HcalIsolatedTrack/interface/IsolatedPixelTrackCandidate.h"
#include "DataFormats/HcalIsolatedTrack/interface/IsolatedPixelTrackCandidateFwd.h"

#include "DataFormats/Math/interface/deltaR.h"

#include <fstream>

#include "TH1F.h"

class DQMHcalIsoTrackAlCaReco : public edm::EDAnalyzer {
public:
  explicit DQMHcalIsoTrackAlCaReco(const edm::ParameterSet&);
  ~DQMHcalIsoTrackAlCaReco();
  
  
private:

  DQMStore* dbe_;  

  virtual void beginJob() override ;
  virtual void analyze(const edm::Event&, const edm::EventSetup&) override;
  virtual void endJob() override ;

  std::string folderName_;
  bool saveToFile_;
  std::string outRootFileName_;
  edm::EDGetTokenT<trigger::TriggerEvent> hltEventTag_;
  std::string l1FilterTag_;
  std::vector<std::string> hltFilterTag_;
  edm::EDGetTokenT<reco::IsolatedPixelTrackCandidateCollection> arITrLabel_;
  edm::InputTag recoTrLabel_;
  double pThr_;
  double heLow_;
  double heUp_;
  
  MonitorElement* hl3Pt;
  MonitorElement* hl3eta;
  MonitorElement* hl3AbsEta;
  MonitorElement* hl3phi;
  MonitorElement* hOffL3TrackMatch;
  MonitorElement* hOffL3TrackPtRat;

  MonitorElement* hOffP_0005;
  MonitorElement* hOffP_0510;
  MonitorElement* hOffP_1015;
  MonitorElement* hOffP_1520;

  MonitorElement* hOffP;

  MonitorElement* hTracksSumP;
  MonitorElement* hTracksMaxP;

  MonitorElement* hDeposEcalInnerEB;
  MonitorElement* hDeposEcalOuterEB;
  MonitorElement* hDeposEcalInnerEE;
  MonitorElement* hDeposEcalOuterEE;
  
  MonitorElement* hL1jetMatch;

  MonitorElement* hOffEtaFP;
  MonitorElement* hOffAbsEta;
  MonitorElement* hOffPhiFP;

  MonitorElement* hOffEta;
  MonitorElement* hOffPhi;
  
  MonitorElement* hOccupancyFull;
  MonitorElement* hOccupancyHighEn;

  MonitorElement* hPurityEta;
  MonitorElement* hPurityPhi;

  int nTotal;
  int nHLTL3accepts;
  int nameLength_;
  int l1nameLength_;
  
  std::pair<int, int> towerIndex(double eta, double phi);

};

std::pair<int,int> DQMHcalIsoTrackAlCaReco::towerIndex(double eta, double phi) 
{
  int ieta = 0;
  int iphi = 0;
  for (int i=1; i<21; i++)
    {
      if (fabs(eta)<=(i*0.087)&&fabs(eta)>(i-1)*0.087) ieta=int(fabs(eta)/eta)*i;
    }
  if (fabs(eta)>1.740&&fabs(eta)<=1.830) ieta=int(fabs(eta)/eta)*21;
  if (fabs(eta)>1.830&&fabs(eta)<=1.930) ieta=int(fabs(eta)/eta)*22;
  if (fabs(eta)>1.930&&fabs(eta)<=2.043) ieta=int(fabs(eta)/eta)*23;
  if (fabs(eta)>2.043&&fabs(eta)<=2.172) ieta=int(fabs(eta)/eta)*24;

  double delta=phi+0.174532925;
  if (delta<0) delta=delta+2*acos(-1);
  if (fabs(eta)<1.740) 
    {
      for (int i=0; i<72; i++)
	{
	  if (delta<(i+1)*0.087266462&&delta>i*0.087266462) iphi=i;
	}
    }
  else 
    {
      for (int i=0; i<36; i++)
	{
	  if (delta<2*(i+1)*0.087266462&&delta>2*i*0.087266462) iphi=2*i;
	}
    }

  return std::pair<int,int>(ieta,iphi);
}


DQMHcalIsoTrackAlCaReco::DQMHcalIsoTrackAlCaReco(const edm::ParameterSet& iConfig)

{
  folderName_ = iConfig.getParameter<std::string>("folderName");
  saveToFile_=iConfig.getParameter<bool>("saveToFile");
  outRootFileName_=iConfig.getParameter<std::string>("outputRootFileName");
  hltEventTag_= consumes<trigger::TriggerEvent>(iConfig.getParameter<edm::InputTag>("hltTriggerEventLabel"));
  l1FilterTag_=iConfig.getParameter<std::string>("l1FilterLabel");
  hltFilterTag_=iConfig.getParameter<std::vector<std::string> >("hltL3FilterLabels");
  nameLength_=iConfig.getUntrackedParameter<int>("filterNameLength",27);
  l1nameLength_=iConfig.getUntrackedParameter<int>("l1filterNameLength",11);
  arITrLabel_= consumes<reco::IsolatedPixelTrackCandidateCollection>(iConfig.getParameter<edm::InputTag>("alcarecoIsoTracksLabel"));
  recoTrLabel_=iConfig.getParameter<edm::InputTag>("recoTracksLabel");
  pThr_=iConfig.getUntrackedParameter<double>("pThrL3",0);
  heLow_=iConfig.getUntrackedParameter<double>("lowerHighEnergyCut",40);
  heUp_=iConfig.getUntrackedParameter<double>("upperHighEnergyCut",60);

  nTotal=0;
  nHLTL3accepts=0;
}

DQMHcalIsoTrackAlCaReco::~DQMHcalIsoTrackAlCaReco()
{}

void DQMHcalIsoTrackAlCaReco::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{

  nTotal++;

  edm::Handle<trigger::TriggerEvent> trEv;
  iEvent.getByToken(hltEventTag_,trEv);
  
  edm::Handle<reco::IsolatedPixelTrackCandidateCollection> recoIsoTracks;
  iEvent.getByToken(arITrLabel_,recoIsoTracks);

  const trigger::TriggerObjectCollection& TOCol(trEv->getObjects());

  const trigger::size_type nFilt(trEv->sizeFilters());

  //get coordinates of L1 trigger
  trigger::Keys KEYSl1;
  for (trigger::size_type iFilt=0; iFilt!=nFilt; iFilt++)
    {
      if ((trEv->filterTag(iFilt).label()).substr(0,l1nameLength_)==l1FilterTag_) KEYSl1=trEv->filterKeys(iFilt); 
    }
  
  trigger::size_type nRegl1=KEYSl1.size();
  
  double etaTrigl1=-10000;
  double phiTrigl1=-10000;
  double ptMaxl1=0;
  for (trigger::size_type iReg=0; iReg<nRegl1; iReg++)
    {
      const trigger::TriggerObject& TObj(TOCol[KEYSl1[iReg]]);
      if (TObj.pt()>ptMaxl1)
	{
	  etaTrigl1=TObj.eta();
	  phiTrigl1=TObj.phi();
	  ptMaxl1=TObj.pt();
	}
    }
  
  //get coordinates of hlt objects
  std::vector<double> trigEta;
  std::vector<double> trigPhi;
  
  trigger::Keys KEYS;
  for (unsigned l=0; l<hltFilterTag_.size(); l++)
    {
      for (trigger::size_type iFilt=0; iFilt!=nFilt; iFilt++) 
	{
	  if ((trEv->filterTag(iFilt).label()).substr(0,nameLength_)==hltFilterTag_[l]) 
	    {
	      KEYS=trEv->filterKeys(iFilt);
	    }
	}
      
      trigger::size_type nReg=KEYS.size();
      
      //checks with IsoTrack trigger results
      for (trigger::size_type iReg=0; iReg<nReg; iReg++)
	{
	  const trigger::TriggerObject& TObj(TOCol[KEYS[iReg]]);
	  if (TObj.p()<pThr_) continue;
	  hl3eta->Fill(TObj.eta(),1);
	  hl3AbsEta->Fill(fabs(TObj.eta()),1);
	  hl3phi->Fill(TObj.phi(),1);
	  
	  if (recoIsoTracks->size()>0)
	    {
	      double minRecoL3dist=1000;
	      reco::IsolatedPixelTrackCandidateCollection::const_iterator mrtr;
	      for (reco::IsolatedPixelTrackCandidateCollection::const_iterator rtrit=recoIsoTracks->begin(); rtrit!=recoIsoTracks->end(); rtrit++)
		{
		  double R=deltaR(rtrit->eta(),rtrit->phi(),TObj.eta(),TObj.phi()); 
		  if (R<minRecoL3dist) 
		    {
		      mrtr=rtrit;
		      minRecoL3dist=R;
		    }
		}
	      hOffL3TrackMatch->Fill(minRecoL3dist,1);
	      if (minRecoL3dist<0.02) hOffL3TrackPtRat->Fill(TObj.pt()/mrtr->pt(),1);
	    }
	  
	  hl3Pt->Fill(TObj.pt(),1);
	  trigEta.push_back(TObj.eta());
	  trigPhi.push_back(TObj.phi());
	}
    }
  
  //general distributions
  for (reco::IsolatedPixelTrackCandidateCollection::const_iterator itr=recoIsoTracks->begin(); itr!=recoIsoTracks->end(); itr++)
    {
      bool match=false;
      for (unsigned int l=0; l<trigEta.size(); l++)
	{
	  if (deltaR(itr->eta(),itr->phi(),trigEta[l],trigPhi[l])<0.02) match=true;
	}
      if (match)
	{	
	  hOffEtaFP->Fill(itr->eta(),1);
	  hOffPhiFP->Fill(itr->phi(),1);
	}
      
      hOffEta->Fill(itr->eta(),1);
      hOffPhi->Fill(itr->phi(),1);

      hOffAbsEta->Fill(fabs(itr->eta()),1);

      hL1jetMatch->Fill(deltaR(itr->eta(), itr->phi(), etaTrigl1, phiTrigl1),1);  

      if (fabs(itr->eta())<1.479) 
	{
	  hDeposEcalInnerEB->Fill(itr->energyIn(),1);
	  hDeposEcalOuterEB->Fill(itr->energyOut(),1);
	}
      else
	{
	  hDeposEcalInnerEE->Fill(itr->energyIn(),1);
	  hDeposEcalOuterEE->Fill(itr->energyOut(),1);
	}
      
      hTracksSumP->Fill(itr->sumPtPxl(),1);
      if (itr->maxPtPxl()==-10) hTracksMaxP->Fill(0,1);
      else hTracksMaxP->Fill(itr->maxPtPxl(),1);

      if (fabs(itr->eta())<0.5) hOffP_0005->Fill(itr->p(),1);
      if (fabs(itr->eta())>0.5&&fabs(itr->eta())<1.0) hOffP_0510->Fill(itr->p(),1);
      if (fabs(itr->eta())>1.0&&fabs(itr->eta())<1.5) hOffP_1015->Fill(itr->p(),1);
      if (fabs(itr->eta())<1.5&&fabs(itr->eta())<2.0) hOffP_1520->Fill(itr->p(),1);

      hOffP->Fill(itr->p(),1);

      std::pair<int,int> TI=towerIndex(itr->eta(),itr->phi());
      hOccupancyFull->Fill(TI.first,TI.second,1);
      if (itr->p()>heLow_&&itr->p()<heUp_) hOccupancyHighEn->Fill(TI.first,TI.second,1);
    }    
      
}

void DQMHcalIsoTrackAlCaReco::beginJob()
{
  dbe_ = edm::Service<DQMStore>().operator->();
  dbe_->setCurrentFolder(folderName_);

  hl3Pt=dbe_->book1D("hl3Pt","pT of hlt L3 objects",1000,0,1000);
  hl3Pt->setAxisTitle("pT(GeV)",1);

  hl3eta=dbe_->book1D("hl3eta","eta of hlt L3 objects",16,-2,2);
  hl3eta->setAxisTitle("eta",1);
  hl3AbsEta=dbe_->book1D("hl3AbsEta","|eta| of hlt L3 objects",8,0,2);
  hl3AbsEta->setAxisTitle("eta",1);
  hl3phi=dbe_->book1D("hl3phi","phi of hlt L3 objects",16,-3.2,3.2);
  hl3phi->setAxisTitle("phi",1);
  hOffEta=dbe_->book1D("hOffEta","eta of alcareco objects",100,-2,2);
  hOffEta->setAxisTitle("eta",1);
  hOffPhi=dbe_->book1D("hOffPhi","phi of alcareco objects",100,-3.2,3.2);
  hOffPhi->setAxisTitle("phi",1);
  hOffP=dbe_->book1D("hOffP","p of alcareco objects",1000,0,1000);
  hOffP->setAxisTitle("E(GeV)",1);
  hOffP_0005=dbe_->book1D("hOffP_0005","p of alcareco objects, |eta|<0.5",1000,0,1000);
  hOffP_0005->setAxisTitle("E(GeV)",1);
  hOffP_0510=dbe_->book1D("hOffP_0510","p of alcareco objects, 0.5<|eta|<1.0",1000,0,1000);
  hOffP_0510->setAxisTitle("E(GeV)",1);
  hOffP_1015=dbe_->book1D("hOffP_1015","p of alcareco objects, 1.0<|eta|<1.5",1000,0,1000);
  hOffP_1015->setAxisTitle("E(GeV)",1);
  hOffP_1520=dbe_->book1D("hOffP_1520","p of alcareco objects, 1.5<|eta|<2.0",1000,0,1000);
  hOffP_1520->setAxisTitle("E(GeV)",1);
  hOffEtaFP=dbe_->book1D("hOffEtaFP","eta of alcareco objects, FP",16,-2,2);
  hOffEtaFP->setAxisTitle("eta",1);
  hOffAbsEta=dbe_->book1D("hOffAbsEta","|eta| of alcareco objects",8,0,2);
  hOffAbsEta->setAxisTitle("|eta|",1);
  hOffPhiFP=dbe_->book1D("hOffPhiFP","phi of alcareco objects, FP",16,-3.2,3.2);
  hOffPhiFP->setAxisTitle("phi",1);
  hTracksSumP=dbe_->book1D("hTracksSumP","summary p of tracks in the isolation cone",100,0,20);
  hTracksSumP->setAxisTitle("E(GeV)");
  hTracksMaxP=dbe_->book1D("hTracksMaxP","maximum p among tracks in the isolation cone",100,0,20);
  hTracksMaxP->setAxisTitle("E(GeV)");
  hDeposEcalInnerEE=dbe_->book1D("hDeposEcalInnerEE","ecal energy deposition in inner cone around track, EE",20,0,20);
  hDeposEcalInnerEE->setAxisTitle("E(GeV)");
  hDeposEcalOuterEE=dbe_->book1D("hDeposEcalOuterEE","ecal energy deposition in outer cone around track, EE",100,0,100);
  hDeposEcalInnerEB=dbe_->book1D("hDeposEcalInnerEB","ecal energy deposition in inner cone around track, EB",20,0,20);
  hDeposEcalInnerEB->setAxisTitle("E(GeV)");
  hDeposEcalOuterEB=dbe_->book1D("hDeposEcalOuterEB","ecal energy deposition in outer cone around track, EB",100,0,100);
  hDeposEcalOuterEB->setAxisTitle("E(GeV)");
  hOccupancyFull=dbe_->book2D("hOccupancyFull","number of tracks per tower, full energy range",48,-25,25,73,0,73);
  hOccupancyFull->setAxisTitle("ieta",1);
  hOccupancyFull->setAxisTitle("iphi",2);
  hOccupancyFull->getTH2F()->SetOption("colz");
  hOccupancyFull->getTH2F()->SetStats(kFALSE);
  hOccupancyHighEn=dbe_->book2D("hOccupancyHighEn","number of tracks per tower, high energy tracks",48,-25,25,73,0,73);
  hOccupancyHighEn->setAxisTitle("ieta",1);
  hOccupancyHighEn->setAxisTitle("iphi",2);
  hOccupancyHighEn->getTH2F()->SetOption("colz");
  hOccupancyHighEn->getTH2F()->SetStats(kFALSE);
  hOffL3TrackMatch=dbe_->book1D("hOffL3TrackMatch","Distance from L3 object to offline track",40,0,0.2);
  hOffL3TrackMatch->setAxisTitle("R(eta,phi)",1);
  hOffL3TrackPtRat=dbe_->book1D("hOffL3TrackPtRat","Ratio of pT: L3/offline",500,0,3);
  hOffL3TrackPtRat->setAxisTitle("ratio L3/offline",1);

  hL1jetMatch=dbe_->book1D("hL1jetMatch","dR(eta,phi) from leading L1 jet to offline track",100,0,5);

}

void DQMHcalIsoTrackAlCaReco::endJob() {

if(dbe_) 
  {
    if (saveToFile_) dbe_->save(outRootFileName_);
  }
}

