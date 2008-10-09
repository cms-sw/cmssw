// -*- C++ -*-
//
// Package:    HLTriggerOffline/special
// Class:      HLTHcalIsoTrackVal
// 
/**\class HLTHcalIsoTrackVal HLTHcalIsoTrackVal.cc HLTriggerOffline/special/src/HLTHcalIsoTrackVal.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Grigory SAFRONOV
//         Created:  Mon Oct  6 10:10:22 CEST 2008
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

#include "TH1F.h"
#include "TH2F.h"
#include "TFile.h"

#include <fstream>

class HLTHcalIsoTrackVal : public edm::EDAnalyzer {
public:
  explicit HLTHcalIsoTrackVal(const edm::ParameterSet&);
  ~HLTHcalIsoTrackVal();
  
  
private:
  virtual void beginJob(const edm::EventSetup&) ;
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void endJob() ;

  double sampleXsec_;
  double lumi_;
  std::string outTxtFileName_;
  
  std::string outRootFileName_;
  edm::InputTag hltEventTag_;
  edm::InputTag hltFilterTag_;
  std::vector<edm::InputTag> l1extraJetTag_;
  std::vector<std::string> l1seedNames_;
  edm::InputTag gtDigiLabel_; 
  bool useReco_;
  edm::InputTag recoTracksLabel_;
  
  TFile* hfile;
  TH1F* hl3tauMatch;
  TH1F* hl1tauMatchPt;
  TH1F* hl1tauPt;
  TH1F* hl3Pt;
  TH1F* hl3eta;
  TH1F* hl3phi;
  TH1F* hOffL3TrackMatch;
  TH1F* hOffL3TrackPtRat;
  TH1F* haccepts;
  TH2F* hOffPvsEta;

  std::vector<int> l1counter;

  std::ofstream txtout;

  int nTotal;
  int nL1accepts;
  int nHLTL2accepts;
  int nHLTL3accepts;
  
};

double getDist(double eta1, double phi1, double eta2, double phi2)
{
  double dphi = fabs(phi1 - phi2); 
  if(dphi>acos(-1)) dphi = 2*acos(-1)-dphi;
  double dr = sqrt(dphi*dphi + pow(eta1-eta2,2));
  return dr;
}

HLTHcalIsoTrackVal::HLTHcalIsoTrackVal(const edm::ParameterSet& iConfig)

{
  sampleXsec_=iConfig.getParameter<double>("sampleCrossSection");
  lumi_=iConfig.getParameter<double>("luminosity");
  outTxtFileName_=iConfig.getParameter<std::string>("outputTxtFileName");
  outRootFileName_=iConfig.getParameter<std::string>("outputRootFileName");
  hltEventTag_=iConfig.getParameter<edm::InputTag>("hltTriggerEventLabel");
  hltFilterTag_=iConfig.getParameter<edm::InputTag>("hltL3FilterLabel");
  l1extraJetTag_=iConfig.getParameter<std::vector<edm::InputTag> >("hltL1extraJetLabel");
  gtDigiLabel_=iConfig.getParameter<edm::InputTag>("gtDigiLabel");
  l1seedNames_=iConfig.getParameter<std::vector<std::string> >("l1seedNames");
  useReco_=iConfig.getParameter<bool>("useReco");
  recoTracksLabel_=iConfig.getParameter<edm::InputTag>("recoTracksLabel");
  
  hfile=new TFile(outRootFileName_.c_str(),"RECREATE");
  hl3tauMatch=new TH1F("hl3tauMatch","R from L3 object to L1tauJet ",100,0,5);
  hl1tauMatchPt=new TH1F("hl1tauPt","pT of matched L1tauJets",1000,0,100);
  hl1tauPt=new TH1F("hl1tauPt","pT of all L1tauJets",1000,0,100);
  hl3Pt=new TH1F("hl3Pt","pT of L3 objects",1000,0,1000);
  hl3eta=new TH1F("hl3eta","eta of L3 objects",50,-2.5,2.5);
  hl3phi=new TH1F("hl3phi","phi of L3 objects",70,-3.5,3.5);
  haccepts=new TH1F("haccepts","Number of accepts at each level",3,0,3);
  hOffL3TrackMatch=new TH1F("hOffL3TrackMatch","Distance from L3 object to offline track",200,0,0.5);
  hOffL3TrackPtRat=new TH1F("hOffL3TrackPtRat","Ratio of pT: L3/offline",100,0,10);
  hOffPvsEta=new TH2F("hOffPvsEta","Distribution of offline track energy vs eta",25,-2.5,2.5,100,0,100);

  for (unsigned int i=0; i<l1seedNames_.size(); i++)
    {
      l1counter.push_back(0);
      //      std::cout<<l1counter[i]<<std::endl;
    }

  txtout.open(outTxtFileName_.c_str());

  nTotal=0;
  nL1accepts=0;
  nHLTL2accepts=0;
  nHLTL3accepts=0;
}


HLTHcalIsoTrackVal::~HLTHcalIsoTrackVal()
{
  hfile->cd();
  hl3tauMatch->Write();
  hl1tauMatchPt->Write();
  hl1tauPt->Write();
  hl3Pt->Write();
  hl3eta->Write();
  hl3phi->Write();
  haccepts->Write();
  hOffL3TrackMatch->Write();
  hOffL3TrackPtRat->Write();
  hOffPvsEta->Write();
  hfile->Close();

  double sampleRate=(lumi_)*(sampleXsec_*10E-36);
  double l1Rate=nL1accepts*pow(nTotal,-1)*sampleRate;
  double hltRate=nHLTL3accepts*pow(nL1accepts,-1)*l1Rate;

  double l1rateError=l1Rate/sqrt(nL1accepts);
  double hltRateError=hltRate/sqrt(nHLTL3accepts);

  txtout<<std::setw(40)<<std::left<<"sample xsec(pb)"<<sampleXsec_<<std::endl;
  txtout<<std::setw(40)<<std::left<<"lumi(cm^-2*s^-1)"<<lumi_<<std::endl;
  txtout<<std::setw(40)<<std::left<<"Events processed/rate(Hz)"<<nTotal<<"/"<<sampleRate<<std::endl;
  txtout<<std::setw(40)<<std::left<<"L1 accepts/(rate+-error (Hz))"<<nL1accepts<<"/("<<l1Rate<<"+-"<<l1rateError<<")"<<std::endl;
  txtout<<std::setw(40)<<std::left<<"HLTL3accepts/(rate+-error (Hz))"<<nHLTL3accepts<<"/("<<hltRate<<"+-"<<hltRateError<<")"<<std::endl;
}

void HLTHcalIsoTrackVal::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  nTotal++;
  haccepts->Fill(0.0001,1);
  edm::ESHandle<L1GtTriggerMenu> menuRcd;
  iSetup.get<L1GtTriggerMenuRcd>().get(menuRcd) ;
  const L1GtTriggerMenu* menu = menuRcd.product();
  const AlgorithmMap& bitMap = menu->gtAlgorithmMap();
  
  edm::ESHandle<L1GtPrescaleFactors> l1GtPfAlgo;
  iSetup.get<L1GtPrescaleFactorsAlgoTrigRcd>().get(l1GtPfAlgo);        
  const L1GtPrescaleFactors* preFac = l1GtPfAlgo.product();
  const std::vector< std::vector< int > > prescaleSet=preFac->gtPrescaleFactors(); 
  
  if (prescaleSet.size()>1) std::cout<<"multiple prescale sets"<<std::endl;
  
  edm::Handle< L1GlobalTriggerReadoutRecord > gtRecord;
  iEvent.getByLabel(gtDigiLabel_ ,gtRecord);
  const DecisionWord dWord = gtRecord->decisionWord(); 
  
  bool l1pass=false;
  
  for (unsigned int i=0; i<l1seedNames_.size(); i++)
    {
      int l1seedBitNumber=-10;
      for (CItAlgo itAlgo = bitMap.begin(); itAlgo != bitMap.end(); itAlgo++) 
	{
	  if (itAlgo->first==l1seedNames_[i]) l1seedBitNumber = (itAlgo->second).algoBitNumber();
	}
      int prescale=prescaleSet[0][l1seedBitNumber];
      if (menu->gtAlgorithmResult( l1seedNames_[i], dWord)) 
	{
	  l1counter[i]++;
	  if (l1counter[i]%prescale==0)
	    {
	      l1pass=true;
	      break;
	    }
	}
      
    }
  if (!l1pass) return;
  else haccepts->Fill(1+0.0001,1);
  
  nL1accepts++;
  
  edm::Handle<trigger::TriggerEvent> trEv;
  iEvent.getByLabel(hltEventTag_,trEv);
  
  edm::Handle<l1extra::L1JetParticleCollection> l1cjets;
  iEvent.getByLabel(l1extraJetTag_[1],l1cjets);
  
  edm::Handle<l1extra::L1JetParticleCollection> l1fjets;
  iEvent.getByLabel(l1extraJetTag_[2],l1fjets);
  
  edm::Handle<l1extra::L1JetParticleCollection> l1tjets;
  iEvent.getByLabel(l1extraJetTag_[0],l1tjets);
  
  edm::Handle<reco::TrackCollection> recoTr;
  
  if (useReco_)
    {
      iEvent.getByLabel(recoTracksLabel_,recoTr);
    }
  
  const trigger::TriggerObjectCollection& TOCol(trEv->getObjects());
  
  trigger::Keys KEYS;
  const trigger::size_type nFilt(trEv->sizeFilters());
  for (trigger::size_type iFilt=0; iFilt!=nFilt; iFilt++) 
    {
      if (trEv->filterTag(iFilt)==hltFilterTag_) KEYS=trEv->filterKeys(iFilt);
      
    }
  trigger::size_type nReg=KEYS.size();
  if (nReg>0) 
    {
      nHLTL3accepts++;
      haccepts->Fill(2+0.0001,1);
    }
  for (trigger::size_type iReg=0; iReg<nReg; iReg++)
    {
      const trigger::TriggerObject& TObj(TOCol[KEYS[iReg]]);
      hl3eta->Fill(TObj.eta(),1);
      hl3phi->Fill(TObj.phi(),1);
      l1extra::L1JetParticleCollection::const_iterator mjet;
      double minR=100;
      for (l1extra::L1JetParticleCollection::const_iterator l1tjit=l1tjets->begin(); l1tjit!=l1tjets->end(); l1tjit++)
	{
	  double R=getDist(l1tjit->eta(),l1tjit->phi(),TObj.eta(),TObj.phi());
	  if (R<minR) 
	    {
	      minR=R;
	      mjet=l1tjit;
	    }
	}
      if (recoTr->size()>0)
	{
	  double minRecoL3dist=100;
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
      hl1tauMatchPt->Fill(mjet->pt(),1);
      hl3tauMatch->Fill(minR,1);
    }
  
  for (l1extra::L1JetParticleCollection::const_iterator l1tjit=l1tjets->begin(); l1tjit!=l1tjets->end(); l1tjit++)
    {
      hl1tauPt->Fill(l1tjit->pt(),1);
    }
  
}

void HLTHcalIsoTrackVal::beginJob(const edm::EventSetup&)
{
}

void HLTHcalIsoTrackVal::endJob() {
}

DEFINE_FWK_MODULE(HLTHcalIsoTrackVal);
