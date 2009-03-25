// -*- C++ -*-
//
// Package:    HLTriggerOffline/special
// Class:      DQMHcalIsoTrackAlCaRaw
// 
/**\class DQMHcalIsoTrackAlCaRaw DQMHcalIsoTrackAlCaRaw.cc HLTriggerOffline/special/src/DQMHcalIsoTrackAlCaRaw.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Grigory SAFRONOV
//         Created:  Mon Oct  6 10:10:22 CEST 2008
// $Id: DQMHcalIsoTrackHLT.cc,v 1.1 2009/03/03 11:09:08 safronov Exp $
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

#include "DataFormats/HLTReco/interface/TriggerEventWithRefs.h"
#include "DataFormats/HLTReco/interface/TriggerEvent.h"

#include "DataFormats/L1Trigger/interface/L1JetParticle.h"
#include "DataFormats/L1Trigger/interface/L1JetParticleFwd.h"

#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "FWCore/ServiceRegistry/interface/Service.h"


#include "DataFormats/HcalIsolatedTrack/interface/IsolatedPixelTrackCandidate.h"
#include "DataFormats/HcalIsolatedTrack/interface/IsolatedPixelTrackCandidateFwd.h"

#include "TH1F.h"
#include "TH2F.h"

#include <fstream>

class DQMHcalIsoTrackHLT : public edm::EDAnalyzer {
public:
  explicit DQMHcalIsoTrackHLT(const edm::ParameterSet&);
  ~DQMHcalIsoTrackHLT();
  double getDist(double,double,double,double);
  
private:

  int evtBuf;

  DQMStore* dbe_;  

  virtual void beginJob(const edm::EventSetup&) ;
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void endJob() ;

  std::string folderName_;
  std::string outRootFileName_;

  std::string hltRAWEventTag_;
  std::string hltAODEventTag_;

  std::string l2collectionLabel_;
  std::string l3collectionLabel_;

  std::string l3filterLabel_;
  std::string l1filterLabel_;
  std::string l2filterLabel_;
  std::string hltProcess_;

  bool useHLTDebug_;

  bool saveToRootFile_;


  //ONLINE
  MonitorElement* hL2TowerOccupancy;
  MonitorElement* hL2L3acc;
  MonitorElement* hL3L2rat;

  //OFFLINE
  //momentum distributions
  MonitorElement* hL3Pt;
  MonitorElement* hL3pVsEta;

  MonitorElement* hL3colP;
  MonitorElement* hL3colEta;
  
  MonitorElement* hL3eta;
  MonitorElement* hL3phi;
  MonitorElement* hL3candL2rat;

  //purity of rate
  MonitorElement* hL3etaAOD;
  MonitorElement* hL3pAOD;
  MonitorElement* hL3etaPureAOD;
  MonitorElement* hL3pPureAOD;
  
  //etc
  MonitorElement* hL1pT;
  MonitorElement* hL2eta;
  MonitorElement* hL2phi;
  MonitorElement* hL2pT;
  MonitorElement* hisopT;
  MonitorElement* hisopTvsEta;
  MonitorElement* hL3L2trackMatch;

};

double DQMHcalIsoTrackHLT::getDist(double eta1, double phi1, double eta2, double phi2)
{
  double dphi = fabs(phi1 - phi2); 
  if(dphi>acos(-1)) dphi = 2*acos(-1)-dphi;
  double dr = sqrt(dphi*dphi + pow(eta1-eta2,2));
  return dr;
}

DQMHcalIsoTrackHLT::DQMHcalIsoTrackHLT(const edm::ParameterSet& iConfig)
{
  folderName_ = iConfig.getParameter<std::string>("folderName");
  outRootFileName_=iConfig.getParameter<std::string>("outputRootFileName");

  useHLTDebug_=iConfig.getParameter<bool>("useHLTDebug");
  hltRAWEventTag_=iConfig.getParameter<std::string>("hltRAWTriggerEventLabel");
  hltAODEventTag_=iConfig.getParameter<std::string>("hltAODTriggerEventLabel");

  l2collectionLabel_=iConfig.getParameter<std::string>("l2collectionLabel");
  l3collectionLabel_=iConfig.getParameter<std::string>("l3collectionLabel");

  l3filterLabel_=iConfig.getParameter<std::string>("hltL3filterLabel");
  l1filterLabel_=iConfig.getParameter<std::string>("hltL1filterLabel");
  l2filterLabel_=iConfig.getParameter<std::string>("hltL2filterLabel");
  hltProcess_=iConfig.getParameter<std::string>("hltProcessName");

  saveToRootFile_=iConfig.getParameter<bool>("SaveToRootFile");

}


DQMHcalIsoTrackHLT::~DQMHcalIsoTrackHLT()
{}

void DQMHcalIsoTrackHLT::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  edm::Handle<trigger::TriggerEventWithRefs> triggerObj;
  edm::InputTag toLab=edm::InputTag(hltRAWEventTag_,"",hltProcess_);
  iEvent.getByLabel(toLab,triggerObj); 
  if(!triggerObj.isValid()) 
    { 
      edm::LogWarning("DQMHcalIsoTrack") << "RAW-type HLT results not found, skipping event";
      return;
    }
  
  std::vector<l1extra::L1JetParticleRef> l1CenJets;
  std::vector<l1extra::L1JetParticleRef> l1ForJets;
  std::vector<l1extra::L1JetParticleRef> l1TauJets;
  edm::InputTag l1Tag = edm::InputTag(l1filterLabel_, "",hltProcess_);
  trigger::size_type l1filterIndex=triggerObj->filterIndex(l1Tag);
  if (l1filterIndex<triggerObj->size())
    {
      triggerObj->getObjects(l1filterIndex, trigger::TriggerL1CenJet, l1CenJets);
      triggerObj->getObjects(l1filterIndex, trigger::TriggerL1ForJet, l1ForJets);
      triggerObj->getObjects(l1filterIndex, trigger::TriggerL1TauJet, l1TauJets);
    }

  std::vector<reco::IsolatedPixelTrackCandidateRef> l2tracks;
  edm::InputTag l2Tag = edm::InputTag(l2filterLabel_,"",hltProcess_);
  trigger::size_type l2filterIndex=triggerObj->filterIndex(l2Tag);
  if (l2filterIndex<triggerObj->size()) triggerObj->getObjects(l2filterIndex, trigger::TriggerTrack, l2tracks);
  
  std::vector<reco::IsolatedPixelTrackCandidateRef> l3tracks;
  edm::InputTag l3Tag = edm::InputTag(l3filterLabel_, "",hltProcess_);
  trigger::size_type l3filterIndex=triggerObj->filterIndex(l3Tag);
  if (l3filterIndex<triggerObj->size()) triggerObj->getObjects(l3filterIndex, trigger::TriggerTrack, l3tracks);

  if (l1CenJets.size()>0||l1ForJets.size()>0||l1CenJets.size()>0) hL2L3acc->Fill(1+0.0001,1);
  
  edm::Handle<reco::IsolatedPixelTrackCandidateCollection> l3col;
  edm::InputTag l3colTag=edm::InputTag(l3collectionLabel_,"",hltProcess_);
  
  if (l2tracks.size()>0) 
    {
      hL2L3acc->Fill(2+0.0001,1);
      if (useHLTDebug_)
	{
	  iEvent.getByLabel(l3collectionLabel_,l3col);
	  
	  for (reco::IsolatedPixelTrackCandidateCollection::const_iterator l3it=l3col->begin(); l3it!=l3col->end(); ++l3it)
	    {
	      hL3colP->Fill(l3it->track()->pt()*cosh(l3it->track()->eta()),1);
	      hL3colEta->Fill(l3it->track()->eta(),1);
	    }
	}
    }
  if (l3tracks.size()>0) hL2L3acc->Fill(3+0.0001,1);

  for (unsigned int i=0; i<l1CenJets.size(); i++)
    {
      hL1pT->Fill(l1CenJets[i]->pt(),1);
    }
  for (unsigned int i=0; i<l1ForJets.size(); i++)
    {
      hL1pT->Fill(l1ForJets[i]->pt(),1);
    }
  for (unsigned int i=0; i<l1TauJets.size(); i++)
    {
      hL1pT->Fill(l1TauJets[i]->pt(),1);
    }
  for (unsigned int i=0; i<l2tracks.size(); i++)
    {
      hL2eta->Fill(l2tracks[i]->track()->eta(),1);
      hL2phi->Fill(l2tracks[i]->track()->phi(),1);
      hL2pT->Fill(l2tracks[i]->track()->pt(),1);
      hL2TowerOccupancy->Fill((l2tracks[i]->towerIndex()).first,(l2tracks[i]->towerIndex()).second,1);
      hisopT->Fill(l2tracks[i]->maxPtPxl(),1);
      hisopTvsEta->Fill(l2tracks[i]->track()->eta(),l2tracks[i]->maxPtPxl(),1);
      if (useHLTDebug_)
	{
	  reco::IsolatedPixelTrackCandidateCollection::const_iterator selTrIt;
	  double drmin=0.3;
	  for (reco::IsolatedPixelTrackCandidateCollection::const_iterator l3it=l3col->begin(); l3it!=l3col->end(); ++l3it)
	    {
	      double drl2l3=getDist(l3it->track()->eta(),l3it->track()->phi(),l2tracks[i]->track()->eta(),l2tracks[i]->track()->phi());
	      if (drl2l3<drmin)
		{
		  drmin=drl2l3;
		  selTrIt=l3it;
		}
	    }
	  if (drmin!=0.3) hL3candL2rat->Fill(selTrIt->track()->p()/l2tracks[i]->track()->p(),1);
	}
    }
  for (unsigned int i=0; i<l3tracks.size(); i++)
    {
      
      hL3Pt->Fill(l3tracks[i]->track()->pt(),1);
      hL3eta->Fill(l3tracks[i]->track()->eta(),1);
      hL3phi->Fill(l3tracks[i]->track()->phi(),1);
      int seltr=-1;
      double drmin=100;
      for (unsigned int j=0; j<l2tracks.size(); j++)
	{
	  double drl2l3=getDist(l3tracks[i]->track()->eta(),l3tracks[i]->track()->phi(),l2tracks[i]->track()->eta(),l2tracks[i]->track()->phi());
	  if (drl2l3<drmin)
	    {
	      drmin=drl2l3;
	      seltr=j;
	    }
	}
      if (seltr!=-1)
	{
	  hL3L2trackMatch->Fill(drmin,1);
	  hL3L2rat->Fill(l3tracks[i]->track()->p()/l2tracks[seltr]->track()->p(),1);
	}
    }
  
  edm::Handle<trigger::TriggerEvent> trevt;
  edm::InputTag taodLab=edm::InputTag(hltAODEventTag_,"",hltProcess_);
  iEvent.getByLabel(taodLab,trevt);
  
  const trigger::TriggerObjectCollection& TOCol(trevt->getObjects());
  
  trigger::Keys KEYS;
  const trigger::size_type nFilt(trevt->sizeFilters());
  
  int nFired=0;
  
  edm::InputTag hltFilterTag_=edm::InputTag(l3filterLabel_,"",hltProcess_);
  for (trigger::size_type iFilt=0; iFilt!=nFilt; iFilt++) 
    {
      trigger::Keys KEYS1=trevt->filterKeys(iFilt);
      if (KEYS1.size()>0) nFired++;
      if (trevt->filterTag(iFilt)==hltFilterTag_) KEYS=trevt->filterKeys(iFilt);
    }
  
  trigger::size_type nReg=KEYS.size();
  
  for (trigger::size_type k=0; k<nReg; k++)
    {
      const trigger::TriggerObject& TObj(TOCol[KEYS[k]]);
      hL3etaAOD->Fill(TObj.eta(),1);
      hL3pAOD->Fill(TObj.pt()*cosh(TObj.eta()),1);
    }
  
  if (nFired==2&&nReg>0)  
    {
      for (trigger::size_type iReg=0; iReg<nReg; iReg++)
	{
	  const trigger::TriggerObject& TObj(TOCol[KEYS[iReg]]);
	  hL3etaPureAOD->Fill(TObj.eta(),1);
	  hL3pPureAOD->Fill(TObj.pt()*cosh(TObj.eta()),1);
	}
    }
}

void DQMHcalIsoTrackHLT::beginJob(const edm::EventSetup&)
{
  dbe_ = edm::Service<DQMStore>().operator->();
  dbe_->setCurrentFolder(folderName_);

  hL2TowerOccupancy=dbe_->book2D("hL2TowerOccupancy","L2 tower occupancy",48,-25,25,73,0,73);
  hL2TowerOccupancy->setAxisTitle("ieta",1);
  hL2TowerOccupancy->setAxisTitle("iphi",2);
  hL2TowerOccupancy->getTH2F()->SetOption("colz");
  hL2TowerOccupancy->getTH2F()->SetStats(kFALSE);

  hL2L3acc=dbe_->book1D("hL2L3acc","number of L1, L2 and L3 accepts",3,1,4);
  hL2L3acc->setAxisTitle("level",1);

  hL3L2trackMatch=dbe_->book1D("hL3L2trackMatch","R from L3 object to L2 object ",1000,0,1);
  hL3L2trackMatch->setAxisTitle("R(eta,phi)",1);
 
  hL3colP=dbe_->book1D("hL3colP","P of L3 candidates",1000,0,100);
  hL3colP->setAxisTitle("P (GeV)",1);

  hL3colEta=dbe_->book1D("hL3colEta","eta of L3 candidates",100,-3,3);
  hL3colEta->setAxisTitle("eta",1);

  hL3candL2rat=dbe_->book1D("hL3candL2rat","ratio of L3 candidate to accepted L2",1000,0,10);  
  hL3candL2rat->setAxisTitle("P_L3/P_L2",1);

  hL3L2rat=dbe_->book1D("hL3L2rat","ratio of L3 to L2 measurement",1000,0,10);
  hL3L2rat->setAxisTitle("pT_L3/pT_L2",1);

  hL3Pt=dbe_->book1D("hl3Pt","pT of L3 objects",1000,0,100);
  hL3Pt->setAxisTitle("pT(GeV)",1);

  hL3eta=dbe_->book1D("hl3eta","eta of L3 objects",50,-2.5,2.5);
  hL3eta->setAxisTitle("eta",1);

  hL3phi=dbe_->book1D("hl3phi","phi of L3 objects",70,-3.5,3.5);
  hL3phi->setAxisTitle("phi(rad)",1);

  hL3etaAOD=dbe_->book1D("hL3etaAOD","eta of L3 objects (AOD)",50,-2.5,2.5);
  hL3etaAOD->setAxisTitle("eta",1);

  hL3etaPureAOD=dbe_->book1D("hL3etaPureAOD","eta of L3 objects (AOD, pure)",50,-2.5,2.5);
  hL3etaPureAOD->setAxisTitle("eta",1);

  hL3pAOD=dbe_->book1D("hl3pAOD","p of L3 objects (AOD)",1000,0,100);
  hL3pAOD->setAxisTitle("p(GeV)",1);

  hL3pPureAOD=dbe_->book1D("hl3pPureAOD","p of L3 objects (AOD, pure)",1000,0,100);
  hL3pPureAOD->setAxisTitle("p(GeV)",1);

  hL1pT=dbe_->book1D("hl1pT","pT of L1 objects",1000,0,1000);
  hL1pT->setAxisTitle("pT(GeV)",1);

  hL2pT=dbe_->book1D("hl2pT","pT of L2 objects",1000,0,1000);
  hL2pT->setAxisTitle("pT(GeV)",1);

  hL2eta=dbe_->book1D("hl2eta","eta of L2 objects",50,-2.5,2.5);
  hL2eta->setAxisTitle("eta",1);

  hL2phi=dbe_->book1D("hl2phi","phi of L2 objects",70,-3.5,3.5);
  hL2phi->setAxisTitle("phi(rad)",1);

  hisopT=dbe_->book1D("hisopT","isolation pT",100,0,5.5);
  hisopT->setAxisTitle("iso pT (GeV)",1);

  hisopTvsEta=dbe_->book2D("hisopTvsEta","isolation pT vs Eta",8,-2,2,100,0,5.5);
  hisopTvsEta->setAxisTitle("eta",1);
  hisopTvsEta->setAxisTitle("iso pT (GeV)",2);
}

void DQMHcalIsoTrackHLT::endJob() {

if(dbe_&&saveToRootFile_) 
  {  
    dbe_->save(outRootFileName_);
  }
}

DEFINE_FWK_MODULE(DQMHcalIsoTrackHLT);
