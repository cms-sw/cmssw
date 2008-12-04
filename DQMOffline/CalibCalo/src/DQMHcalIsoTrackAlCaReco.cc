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
// $Id$
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

#include <fstream>

#include "TH1F.h"

class DQMHcalIsoTrackAlCaReco : public edm::EDAnalyzer {
public:
  explicit DQMHcalIsoTrackAlCaReco(const edm::ParameterSet&);
  ~DQMHcalIsoTrackAlCaReco();
  
  
private:

  DQMStore* dbe_;  

  virtual void beginJob(const edm::EventSetup&) ;
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void endJob() ;

  std::string folderName_;
  std::string outRootFileName_;
  edm::InputTag hltEventTag_;
  edm::InputTag hltFilterTag_;
  edm::InputTag recoTracksLabel_;
  
  MonitorElement* hl3tauMatch;
  MonitorElement* hl1tauMatchPt;
  MonitorElement* hl1tauPt;
  MonitorElement* hl3Pt;
  MonitorElement* hl3eta;
  MonitorElement* hl3AbsEta;
  MonitorElement* hl3phi;
  MonitorElement* hOffL3TrackMatch;
  MonitorElement* hOffL3TrackPtRat;
  MonitorElement* haccepts;
  MonitorElement* hOffPvsEta;
  MonitorElement* hOffEta;
  MonitorElement* hOffAbsEta;
  MonitorElement* hOffPhi;

  MonitorElement* hPurityEta;
  MonitorElement* hPurityPhi;

  int nTotal;
  int nHLTL3accepts;
  
};

double getDist(double eta1, double phi1, double eta2, double phi2)
{
  double dphi = fabs(phi1 - phi2); 
  if(dphi>acos(-1)) dphi = 2*acos(-1)-dphi;
  double dr = sqrt(dphi*dphi + pow(eta1-eta2,2));
  return dr;
}

DQMHcalIsoTrackAlCaReco::DQMHcalIsoTrackAlCaReco(const edm::ParameterSet& iConfig)

{
  folderName_ = iConfig.getParameter<std::string>("folderName");
  outRootFileName_=iConfig.getParameter<std::string>("outputRootFileName");
  hltEventTag_=iConfig.getParameter<edm::InputTag>("hltTriggerEventLabel");
  hltFilterTag_=iConfig.getParameter<edm::InputTag>("hltL3FilterLabel");
  recoTracksLabel_=iConfig.getParameter<edm::InputTag>("recoTracksLabel");
  
  nTotal=0;
  nHLTL3accepts=0;
}


DQMHcalIsoTrackAlCaReco::~DQMHcalIsoTrackAlCaReco()
{}

void DQMHcalIsoTrackAlCaReco::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  nTotal++;

  edm::Handle<trigger::TriggerEvent> trEv;
  iEvent.getByLabel(hltEventTag_,trEv);
  
  edm::Handle<reco::TrackCollection> recoTr;
  iEvent.getByLabel(recoTracksLabel_,recoTr);

  const trigger::TriggerObjectCollection& TOCol(trEv->getObjects());
  
  trigger::Keys KEYS;
  const trigger::size_type nFilt(trEv->sizeFilters());
  for (trigger::size_type iFilt=0; iFilt!=nFilt; iFilt++) 
    {
      if (trEv->filterTag(iFilt)==hltFilterTag_) 
	{
	KEYS=trEv->filterKeys(iFilt);
	}
    }

  trigger::size_type nReg=KEYS.size();
  if (nReg>0) 
    {
      nHLTL3accepts++;
      haccepts->Fill(2+0.0001,1);
    }


  std::vector<double> trigEta;
  std::vector<double> trigPhi;
  bool trig=false;
  for (trigger::size_type iReg=0; iReg<nReg; iReg++)
    {
      const trigger::TriggerObject& TObj(TOCol[KEYS[iReg]]);
      if (TObj.p()<14) continue;
      hl3eta->Fill(TObj.eta(),1);
      hl3AbsEta->Fill(fabs(TObj.eta()),1);
      hl3phi->Fill(TObj.phi(),1);
      l1extra::L1JetParticleCollection::const_iterator mjet;

      if (recoTr->size()>0)
	{
	  double minRecoL3dist=1000;
	  reco::TrackCollection::const_iterator mrtr;
	  for (reco::TrackCollection::const_iterator rtrit=recoTr->begin(); rtrit!=recoTr->end(); rtrit++)
	    {
	      double R=getDist(rtrit->eta(),rtrit->phi(),TObj.eta(),TObj.phi()); 
	      if (R<minRecoL3dist) 
		{
		  mrtr=rtrit;
		  minRecoL3dist=R;
		}
	    }
	  hOffL3TrackMatch->Fill(minRecoL3dist,1);
	  hOffL3TrackPtRat->Fill(TObj.pt()/mrtr->pt(),1);
	  hOffPvsEta->Fill(mrtr->eta(),mrtr->p(),1);
	}
      
      hl3Pt->Fill(TObj.pt(),1);
	trig=true; 
	trigEta.push_back(TObj.eta());
	trigPhi.push_back(TObj.phi());
   }
  for (reco::TrackCollection::const_iterator bl=recoTr->begin(); bl!=recoTr->end(); bl++)
    {
       bool match=false;
	for (unsigned int l=0; l<trigEta.size(); l++)
	{
	if (getDist(bl->eta(),bl->phi(),trigEta[l],trigPhi[l])<0.4) match=true;
	}
      if (match&&trig){	
      hOffEta->Fill(bl->eta(),1);
      hOffPhi->Fill(bl->phi(),1);
      hOffAbsEta->Fill(fabs(bl->eta()),1);
	}
    }
}

void DQMHcalIsoTrackAlCaReco::beginJob(const edm::EventSetup&)
{
  dbe_ = edm::Service<DQMStore>().operator->();
  dbe_->setCurrentFolder(folderName_);

  hl3tauMatch=dbe_->book1D("hl3tauMatch","R from L3 object to L1tauJet ",100,0,5);
  hl3tauMatch->setAxisTitle("R(eta,phi)",1);
 
  hl1tauMatchPt=dbe_->book1D("hl1tauMatchPt","pT of matched L1tauJets",1000,0,100);
  hl1tauMatchPt->setAxisTitle("pT(GeV)",1);

  hl1tauPt=dbe_->book1D("hl1tauPt","pT of all L1tauJets",1000,0,100);
  hl1tauPt->setAxisTitle("pT(GeV)",1);

  hl3Pt=dbe_->book1D("hl3Pt","pT of L3 objects",1000,0,1000);
  hl3Pt->setAxisTitle("pT(GeV)",1);

  hl3eta=dbe_->book1D("hl3eta","eta of L3 objects",8,-2,2);
  hl3eta->setAxisTitle("eta",1);

  hl3AbsEta=dbe_->book1D("hl3AbsEta","|eta| of L3 objects",8,-2,2);
  hl3AbsEta->setAxisTitle("eta",1);

  hl3phi=dbe_->book1D("hl3phi","phi of L3 objects",8,-3.2,3.2);
  hl3phi->setAxisTitle("phi",1);

  hOffEta=dbe_->book1D("hOffEta","eta of offline objects",8,-2,2);
  hOffEta->setAxisTitle("eta",1);

  hOffAbsEta=dbe_->book1D("hOffAbsEta","|eta| of offline objects",8,-2,2);
  hOffAbsEta->setAxisTitle("|eta|",1);

  hOffPhi=dbe_->book1D("hOffPhi","phi of offline objects",8,-3.2,3.2);
  hOffPhi->setAxisTitle("phi",1);

  haccepts=dbe_->book1D("haccepts","Number of accepts at each level",3,0,3);
  haccepts->setAxisTitle("selection level",1);

  hOffL3TrackMatch=dbe_->book1D("hOffL3TrackMatch","Distance from L3 object to offline track",200,0,0.5);
  hOffL3TrackMatch->setAxisTitle("R(eta,phi)",1);

  hOffL3TrackPtRat=dbe_->book1D("hOffL3TrackPtRat","Ratio of pT: L3/offline",100,0,10);
  hOffL3TrackPtRat->setAxisTitle("ratio L3/offline",1);

  hOffPvsEta=dbe_->book2D("hOffPvsEta","Distribution of offline track energy vs eta",25,-2.5,2.5,100,0,100);
  hOffPvsEta->setAxisTitle("eta",1);
  hOffPvsEta->setAxisTitle("E(GeV)",2);

  hPurityEta=dbe_->book1D("hPurityEta","Purity of HLT selection vs Eta",8,-2,2);
  hPurityEta->setAxisTitle("eta",1);
  hPurityEta->setAxisTitle("N(Offline)/N(L3)",2);

  hPurityPhi=dbe_->book1D("hPurityEta","Purity of HLT selection vs Eta",8,-2,2);
  hPurityPhi->setAxisTitle("eta",1);
  hPurityPhi->setAxisTitle("N(Offline)/N(L3)",2);

}

void DQMHcalIsoTrackAlCaReco::endJob() {

if(dbe_) 
  {
    TH1F* hPurEta=new TH1F("hPurEta","hPurEta",8,-2,2);
    TH1F* hPurPhi=new TH1F("hPurPhi","hPurPhi",8,-3.2,3.2);
    
    TH1F* hl3e=hl3eta->getTH1F();
    hl3e->TH1F::Sumw2();

    TH1F* hOffe=hOffEta->getTH1F();
    hOffe->TH1F::Sumw2();

    hPurEta->Divide(hOffe,hl3e,1,1);

    TH1F* hl3p=hl3phi->getTH1F();
    hl3p->TH1F::Sumw2();

    TH1F* hOffp=hOffPhi->getTH1F();
    hOffp->TH1F::Sumw2();

    hPurPhi->Divide(hOffp,hl3p,1,1);
    
    hPurityEta=dbe_->book1D("hPurityEta",hPurEta);
    hPurityEta->setAxisTitle("eta",1);
    hPurityEta->setAxisTitle("N(Offline)/N(L3)",2);
    
    hPurityPhi=dbe_->book1D("hPurityPhi",hPurPhi);
    hPurityPhi->setAxisTitle("phi",1);
    hPurityPhi->setAxisTitle("N(Offline)/N(L3)",2);

    dbe_->save(outRootFileName_);
  }
}

