//code for Hcal IsoTrack HLT MC validation
//
// Original Author:  Grigory SAFRONOV


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

#include "DataFormats/JetReco/interface/GenJetCollection.h"

#include "DataFormats/HcalIsolatedTrack/interface/IsolatedPixelTrackCandidate.h"
#include "DataFormats/HcalIsolatedTrack/interface/IsolatedPixelTrackCandidateFwd.h"

#include "DataFormats/HLTReco/interface/TriggerEventWithRefs.h"
#include "DataFormats/HLTReco/interface/TriggerEvent.h"

#include "DataFormats/Common/interface/TriggerResults.h"
#include "FWCore/Framework/interface/TriggerNames.h"

#include "DataFormats/Math/interface/deltaR.h"

#include "TH1F.h"
#include "TH2F.h"

#include <fstream>

class ValHcalIsoTrackHLT : public edm::EDAnalyzer {
public:
  explicit ValHcalIsoTrackHLT(const edm::ParameterSet&);
  ~ValHcalIsoTrackHLT();
  double getDist(double,double,double,double);  
  
private:

  int evtBuf;

  DQMStore* dbe_;  

  virtual void beginJob() ;
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void endJob();

  bool produceRates_;
  double sampleXsec_;
  double lumi_;
  std::string outTxtFileName_;
  std::string folderName_;
  bool saveToRootFile_;
  std::string outRootFileName_;

  std::string hltRAWEventLabel_;
  edm::InputTag hlTriggerResults_;
  std::string l3FilterLabelHB_;
  std::string l3FilterLabelHE_;
  std::string l1filterName_;
  std::vector<edm::InputTag> l1extraJetTag_;
  std::vector<std::string> l1seedNames_;
  edm::InputTag gtDigiLabel_; 
  bool useReco_;
  edm::InputTag recoTracksLabel_;
  bool checkL2_;

  edm::InputTag l2colLabelHB_;
  edm::InputTag l2colLabelHE_; 
  edm::InputTag l3colLabelHB_;
  edm::InputTag l3colLabelHE_; 

  std::string trigHB_;
  std::string trigHE_;

  bool testL1_;

  std::string hltProcess_;

  bool checkL1eff_;
  edm::InputTag genJetsLabel_;

  bool produceRatePdep_;

  bool doL1Prescaling_;

  bool usePixelTracks_;
  edm::InputTag pixelTrackLabelHB_;
  edm::InputTag pixelTrackLabelHE_;

  MonitorElement* hNPassedL2heL3acc;
  MonitorElement* hNPassedL2hbL3acc;

  MonitorElement* hNPassedL2heAll;
  MonitorElement* hNPassedL2hbAll;
  
  MonitorElement* hNL2candsHB;
  MonitorElement* hNL2candsHE;

  MonitorElement* hl3PHB;
  MonitorElement* hl3PHE;
  MonitorElement* hL3L2trackRhb;
  MonitorElement* hL3L2trackRhe;

  MonitorElement* hL3L2PdiffHB;
  MonitorElement* hL3L2PdiffHE;

  MonitorElement* hl2phiVSeta;
  MonitorElement* hL2passHB;
  MonitorElement* hL2passHE;
  MonitorElement* hL2L3passHB;
  MonitorElement* hL2L3passHE;

  MonitorElement* hl3P0005;
  MonitorElement* hl3P0510;
  MonitorElement* hl3P10145;
  MonitorElement* hl3P14520;

  MonitorElement* hl3etaHB;
  MonitorElement* hl3phiHB;
  MonitorElement* hl3etaHE;
  MonitorElement* hl3phiHE;

  MonitorElement* hOffL3TrackMatch;
  MonitorElement* hOffL3TrackPtRat;

  MonitorElement* hl2etaHB;
  MonitorElement* hl2phiHB;
  MonitorElement* hl2PHB;
  MonitorElement* hisoPHB;
  MonitorElement* hisoPvsEtaHB;

  MonitorElement* hl2etaHE;
  MonitorElement* hl2phiHE;
  MonitorElement* hl2PHE;
  MonitorElement* hisoPHE;
  MonitorElement* hisoPvsEtaHE;

  MonitorElement* hacceptsHB;
  MonitorElement* hacceptsHE;
  MonitorElement* hOffPvsEta;
  MonitorElement* hpTgenLead;
  MonitorElement* hpTgenLeadL1;
  MonitorElement* hpTgenNext;
  MonitorElement* hpTgenNextL1;
  
  //  MonitorElement* hLeadTurnOn;
  //  MonitorElement* hNextToLeadTurnOn;

  MonitorElement* pixHBPt;
  MonitorElement* pixHBP;
  MonitorElement* pixHBCoreEtaPhi;
  MonitorElement* pixHBIsoEtaPhi;
  MonitorElement* pixHBSoftEtaPhi;
  MonitorElement* pixHBMultCore;
  MonitorElement* pixHBMultIso;
  MonitorElement* pixHBMultSoft;
  
  MonitorElement* pixHEPt;
  MonitorElement* pixHEP;
  MonitorElement* pixHECoreEtaPhi;
  MonitorElement* pixHEIsoEtaPhi;
  MonitorElement* pixHESoftEtaPhi;
  MonitorElement* pixHEMultCore;
  MonitorElement* pixHEMultIso;
  MonitorElement* pixHEMultSoft;

  MonitorElement* hHBRateVsThr;
  MonitorElement* hHERateVsThr;

  MonitorElement* hL2L3minDR;

  MonitorElement* hPhiToGJ;
  MonitorElement* hDistToGJ;

  MonitorElement* hTower18check2030;
  MonitorElement* hTower18check4060;

  std::vector<int> l1counter;

  std::ofstream txtout;

  unsigned int nEvtProc;

  int nTotal;
  int nL1accepts;
  int nHLTL2accepts;
  int nHLTL3accepts;
  int nHLTL3acceptsPure;

  int nL3accHB;
  int nL3accHE;

  int nl3_0005;
  int nl3_0510;
  int nl3_10145;
  int nl3_14520;
  int nl3_overlapHB;
  int nl3_overlapHE;  
  
  int purnl3_0005;
  int purnl3_0510;
  int purnl3_10145;
  int purnl3_14520;
  int purnl3_overlapHE;

  double l3PThr_;
  double l2PThrHB_;
  double l2PThrHE_;
  double l2isoHE_;
  double l2isoHB_;

  int nhbacc, nheacc;

  std::vector<std::string> excludeOverl_;
  std::vector<std::string> overlapNames;
  int overlapsHB[100];
  int overlapsHE[100];

//  std::vector<int> runsAcc;
//  std::vector<int> evtsAcc;
};

double ValHcalIsoTrackHLT::getDist(double eta1, double phi1, double eta2, double phi2)
{
  double dphi = fabs(phi1 - phi2); 
  if(dphi>acos(-1)) dphi = 2*acos(-1)-dphi;
  double dr = sqrt(dphi*dphi + pow(eta1-eta2,2));
  return dr;
}

ValHcalIsoTrackHLT::ValHcalIsoTrackHLT(const edm::ParameterSet& iConfig)

{
  produceRates_=iConfig.getParameter<bool>("produceRates");
  sampleXsec_=iConfig.getParameter<double>("sampleCrossSection");
  nEvtProc=iConfig.getParameter<unsigned int>("numberOfEvents");
  lumi_=iConfig.getParameter<double>("luminosity");
  outTxtFileName_=iConfig.getParameter<std::string>("outputTxtFileName");
  
  folderName_ = iConfig.getParameter<std::string>("folderName");
  saveToRootFile_=iConfig.getParameter<bool>("saveToRootFile");
  outRootFileName_=iConfig.getParameter<std::string>("outputRootFileName");

  hltProcess_=iConfig.getParameter<std::string>("hltProcessName");
  hltRAWEventLabel_=iConfig.getParameter<std::string>("hltTriggerEventLabel");
  hlTriggerResults_=iConfig.getParameter<edm::InputTag>("hlTriggerResultsLabel");

  l3FilterLabelHB_=iConfig.getParameter<std::string>("hltL3FilterLabelHB");
  l3FilterLabelHE_=iConfig.getParameter<std::string>("hltL3FilterLabelHE");
 
  l1extraJetTag_=iConfig.getParameter<std::vector<edm::InputTag> >("hltL1extraJetLabel");
  gtDigiLabel_=iConfig.getParameter<edm::InputTag>("gtDigiLabel");
  checkL1eff_=iConfig.getParameter<bool>("checkL1TurnOn");
  testL1_=iConfig.getParameter<bool>("testL1");
  l1filterName_=iConfig.getParameter<std::string>("L1FilterName");
  genJetsLabel_=iConfig.getParameter<edm::InputTag>("genJetsLabel");
  l1seedNames_=iConfig.getParameter<std::vector<std::string> >("l1seedNames");
  useReco_=iConfig.getParameter<bool>("useReco");
  recoTracksLabel_=iConfig.getParameter<edm::InputTag>("recoTracksLabel");
  checkL2_=iConfig.getParameter<bool>("debugL2");
  l2colLabelHB_=iConfig.getParameter<edm::InputTag>("L2producerLabelHB");
  l2colLabelHE_=iConfig.getParameter<edm::InputTag>("L2producerLabelHE");
  l3colLabelHB_=iConfig.getParameter<edm::InputTag>("L3producerLabelHB");
  l3colLabelHE_=iConfig.getParameter<edm::InputTag>("L3producerLabelHE");
  l2PThrHB_=iConfig.getUntrackedParameter<double>("L2momThresholdHB",8);
  l2PThrHE_=iConfig.getUntrackedParameter<double>("L2momThresholdHE",20);
  l2isoHB_=iConfig.getUntrackedParameter<double>("L2isolationHE",2);
  l2isoHE_=iConfig.getUntrackedParameter<double>("L2isolationHB",2);

  produceRatePdep_=iConfig.getParameter<bool>("produceRatePdep");

  l3PThr_=iConfig.getUntrackedParameter<double>("L3momThreshold",20);

  doL1Prescaling_=iConfig.getParameter<bool>("doL1Prescaling");

  trigHB_=iConfig.getParameter<std::string>("HBtriggerName");
  trigHE_=iConfig.getParameter<std::string>("HEtriggerName");

  usePixelTracks_=iConfig.getParameter<bool>("LookAtPixelTracks");
  pixelTrackLabelHB_=iConfig.getParameter<edm::InputTag>("PixelTrackLabelHB");
  pixelTrackLabelHE_=iConfig.getParameter<edm::InputTag>("PixelTrackLabelHE");

  excludeOverl_=iConfig.getParameter<std::vector<std::string> >("excludeFromOverlap");

  std::string kkk="init";
  overlapNames.push_back(kkk);  

  for (unsigned int i=0; i<l1seedNames_.size(); i++)
    {
      l1counter.push_back(0);
    }

  if (produceRates_) txtout.open(outTxtFileName_.c_str());

  nTotal=0;
  nL1accepts=0;
  nHLTL2accepts=0;
  nHLTL3accepts=0;
  nHLTL3acceptsPure=0;
 
  nhbacc=0;
  nheacc=0;  

  nL3accHB=0;
  nL3accHE=0;

  nl3_0005=0;
  nl3_0510=0;
  nl3_10145=0;
  nl3_14520=0;
  nl3_overlapHB=0;
  nl3_overlapHE=0;
  
  purnl3_0005=0;
  purnl3_0510=0;
  purnl3_10145=0;
  purnl3_14520=0;
  purnl3_overlapHE=0;

  for (int k=0; k<100; k++)
    {
      overlapsHB[k]=0;
      overlapsHE[k]=0;
    }

}


ValHcalIsoTrackHLT::~ValHcalIsoTrackHLT()
{
  if (produceRates_)
    {
      double sampleRate=(lumi_)*(sampleXsec_*1E-36);
      double l1Rate=nL1accepts*pow(nTotal,-1)*sampleRate;
      double hltRate=nHLTL3accepts*pow(nEvtProc,-1)*sampleRate;
      double hltRatePure=nHLTL3acceptsPure*pow(nEvtProc,-1)*sampleRate;

      double l1rateError=l1Rate/sqrt(nL1accepts);
      double hltRateError=hltRate/sqrt(nHLTL3accepts);
      double hltRatePureError=hltRatePure/sqrt(nHLTL3acceptsPure);

      double hbrate=nL3accHB*pow(nEvtProc,-1)*sampleRate;
      double herate=nL3accHE*pow(nEvtProc,-1)*sampleRate;
      
      double rate_0005=nl3_0005*pow(nEvtProc,-1)*sampleRate;
      double rate_0510=nl3_0510*pow(nEvtProc,-1)*sampleRate;
      double rate_10145=nl3_10145*pow(nEvtProc,-1)*sampleRate;
      double rate_14520=nl3_14520*pow(nEvtProc,-1)*sampleRate;
      double rate_overlapHB=nl3_overlapHB*pow(nEvtProc,-1)*sampleRate;
      double rate_overlapHE=nl3_overlapHE*pow(nEvtProc,-1)*sampleRate;

      double prate_0005=purnl3_0005*pow(nEvtProc,-1)*sampleRate;
      double prate_0510=purnl3_0510*pow(nEvtProc,-1)*sampleRate;
      double prate_10145=purnl3_10145*pow(nEvtProc,-1)*sampleRate;
      double prate_14520=purnl3_14520*pow(nEvtProc,-1)*sampleRate;
      double prate_overlapHE=purnl3_overlapHE*pow(nEvtProc,-1)*sampleRate;
      
      txtout<<std::setw(50)<<std::left<<"sample xsec(pb)"<<sampleXsec_<<std::endl;
      txtout<<std::setw(50)<<std::left<<"lumi(cm^-2*s^-1)"<<lumi_<<std::endl;
      txtout<<std::setw(50)<<std::left<<"Events processed/rate(Hz)"<<nTotal<<"/"<<sampleRate<<std::endl;
      txtout<<std::setw(50)<<std::left<<"L1 accepts/(rate+-error (Hz))"<<nL1accepts<<"/("<<l1Rate<<"+-"<<l1rateError<<")"<<std::endl;
      txtout<<std::setw(50)<<std::left<<"HLTL3accepts/(rate+-error (Hz))"<<nHLTL3accepts<<"/("<<hltRate<<"+-"<<hltRateError<<")"<<std::endl;
      txtout<<std::setw(50)<<std::left<<"L3 acc. |eta|<0.5 / rate"<<nl3_0005<<" / "<<rate_0005<<std::endl;
      txtout<<std::setw(50)<<std::left<<"L3 acc. 0.5<|eta|<1.0 / rate"<<nl3_0510<<" / "<<rate_0510<<std::endl;
      txtout<<std::setw(50)<<std::left<<"L3 acc. 1.0<|eta|>1.45 / rate"<<nl3_10145<<" / "<<rate_10145<<std::endl;
      txtout<<std::setw(50)<<std::left<<"L3 acc. 1.45<|eta|>2.0 / rate"<<nl3_14520<<" / "<<rate_14520<<std::endl;
      txtout<<std::setw(50)<<std::left<<"L3 acc. overlap HE rate 1.4<|eta|<1.45 / rate"<<nl3_overlapHB<<" / "<<rate_overlapHB<<std::endl;
      txtout<<std::setw(50)<<std::left<<"L3 acc. overlap HB rate 1.4<|eta|<1.45 / rate"<<nl3_overlapHE<<" / "<<rate_overlapHE<<std::endl;
      txtout<<"\n"<<std::endl;

      txtout<<std::setw(50)<<std::left<<"L3 total acc. HB / rate:  "<<nL3accHB<<" / "<<hbrate<<std::endl;
      txtout<<std::setw(50)<<std::left<<"L3 total acc. HE / rate:  "<<nL3accHE<<" / "<<herate<<std::endl;
      txtout<<"\n"<<std::endl;   

      txtout<<std::setw(50)<<std::left<<"HLTL3acceptsPure/(rate+-error (Hz))"<<nHLTL3acceptsPure<<"/("<<hltRatePure<<"+-"<<hltRatePureError<<")"<<std::endl; 
      txtout<<std::setw(50)<<std::left<<"pure L3 acc. |eta|<0.5 / rate"<<purnl3_0005<<" / "<<prate_0005<<std::endl;
      txtout<<std::setw(50)<<std::left<<"pure L3 acc. 0.5<|eta|<1.0 / rate"<<purnl3_0510<<" / "<<prate_0510<<std::endl;
      txtout<<std::setw(50)<<std::left<<"pure L3 acc. 1.0<|eta|>1.45 / rate"<<purnl3_10145<<" / "<<prate_10145<<std::endl;
      txtout<<std::setw(50)<<std::left<<"pure L3 acc. 1.45<|eta|<2.0 / rate"<<purnl3_14520<<" / "<<prate_14520<<std::endl;
      txtout<<std::setw(50)<<std::left<<"pure overlap. 1.4<|eta|<1.45 / rate"<<purnl3_overlapHE<<" / "<<prate_overlapHE<<std::endl;
      //calculate overlaps:
      
      
      for (int i=0; i<overlapNames.size(); k++)
	{
	  if (nL3accHB>0) overlapsHB[k]=overlapsHB[k]/nL3accHB;
	  else overlapsHB[k]=0;
	  if (nL3accHE>0) overlapsHE[k]=overlapsHE[k]/nL3accHE;
	  else overlapsHE[k]=0;
	}
      
      txtout<<"\n\n"<<std::setw(50)<<std::left<<"-------------------Overlap table------------------"<<std::endl;
      for (unsigned int k=0; k<overlapNames.size(); k++)
	{
	  txtout<<std::setw(50)<<std::left<<overlapNames[k]<<"        "<<overlapsHB[k]<<"          "<<overlapsHE[k]<<std::endl;
	}
    }
  /*
    for (int k=0; k<evtsAcc.size(); k++)
    {
    std::cout<<runsAcc[k]<<"                 "<<evtsAcc[k]<<std::endl;
    }
  */
}

void ValHcalIsoTrackHLT::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  edm::Handle<trigger::TriggerEventWithRefs> triggerObj;
  edm::InputTag toLab=edm::InputTag(hltRAWEventLabel_,"",hltProcess_);
  iEvent.getByLabel(toLab,triggerObj); 
  if(!triggerObj.isValid()) 
    { 
      edm::LogWarning("DQMHcalIsoTrack") << "RAW-type HLT results not found, skipping event";
      return;
    }
  
//  int evtNumm=iEvent.id().event();
//  int runNumm=iEvent.id().run();

  nTotal++;
  hacceptsHB->Fill(0.0001,1);
  hacceptsHE->Fill(0.0001,1);

  double phiGJLead=-10000;
  double etaGJLead=-10000;

  bool l1pass=false;
   
  if (testL1_)
    {
      edm::ESHandle<L1GtTriggerMenu> menuRcd;
      iSetup.get<L1GtTriggerMenuRcd>().get(menuRcd) ;
      const L1GtTriggerMenu* menu = menuRcd.product();
      const AlgorithmMap& bitMap = menu->gtAlgorithmMap();
      
      edm::ESHandle<L1GtPrescaleFactors> l1GtPfAlgo;
      iSetup.get<L1GtPrescaleFactorsAlgoTrigRcd>().get(l1GtPfAlgo);        
      const L1GtPrescaleFactors* preFac = l1GtPfAlgo.product();
      const std::vector< std::vector< int > > prescaleSet=preFac->gtPrescaleFactors(); 
      
      edm::Handle< L1GlobalTriggerReadoutRecord > gtRecord;
      iEvent.getByLabel(gtDigiLabel_ ,gtRecord);
      const DecisionWord dWord = gtRecord->decisionWord(); 
      
      for (unsigned int i=0; i<l1seedNames_.size(); i++)
	{
	  int l1seedBitNumber=-10;
	  for (CItAlgo itAlgo = bitMap.begin(); itAlgo != bitMap.end(); itAlgo++) 
	    {
	      if (itAlgo->first==l1seedNames_[i]) l1seedBitNumber = (itAlgo->second).algoBitNumber();
	    }
	  int prescale=prescaleSet[0][l1seedBitNumber];
	  //          std::cout<<l1seedNames_[i]<<"  "<<prescale<<std::endl;
	  if (menu->gtAlgorithmResult( l1seedNames_[i], dWord)) 
	    {
	      l1counter[i]++;
	      if ((doL1Prescaling_&&l1counter[i]%prescale==0)||(!doL1Prescaling_))
		{
		  l1pass=true;
		  break;
		}
	    }
	}
    }
  else 
    {
      std::vector<l1extra::L1JetParticleRef> l1CenJets;
      std::vector<l1extra::L1JetParticleRef> l1ForJets;
      std::vector<l1extra::L1JetParticleRef> l1TauJets;

      edm::InputTag l1Tag = edm::InputTag(l1filterName_, "",hltProcess_);
      trigger::size_type l1filterIndex=triggerObj->filterIndex(l1Tag);
      if (l1filterIndex<triggerObj->size()) 
	{
	   triggerObj->getObjects(l1filterIndex, trigger::TriggerL1CenJet, l1CenJets);
           triggerObj->getObjects(l1filterIndex, trigger::TriggerL1ForJet, l1ForJets);
           triggerObj->getObjects(l1filterIndex, trigger::TriggerL1TauJet, l1TauJets);
	}
      if (l1CenJets.size()>0||l1ForJets.size()>0||l1CenJets.size()>0) l1pass=true;
    }

  if (checkL1eff_) 
    {
      edm::Handle<reco::GenJetCollection> gjcol;
      iEvent.getByLabel(genJetsLabel_,gjcol);
	  
      reco::GenJetCollection::const_iterator gjit=gjcol->begin();
      if (gjit!=gjcol->end())
        {
          etaGJLead=gjit->eta();
          phiGJLead=gjit->phi();
          hpTgenLead->Fill(gjit->pt(),1);
          if (l1pass) hpTgenLeadL1->Fill(gjit->pt(),1);
          gjit++;
        }
      if (gjit!=gjcol->end())
        {
          hpTgenNext->Fill(gjit->pt(),1);
          if (l1pass) hpTgenNextL1->Fill(gjit->pt(),1);
        }
    }

  l1pass=true;
  if (!l1pass) return;

  hacceptsHB->Fill(1+0.0001,1);
  hacceptsHE->Fill(1+0.0001,1);
  
  nL1accepts++;

  // proceed to L2
  bool hbl2fired=false;
  bool hel2fired=false;

  std::vector<reco::IsolatedPixelTrackCandidateCollection::const_iterator> firedl2hbs;
  std::vector<reco::IsolatedPixelTrackCandidateCollection::const_iterator> firedl2hes;

  edm::Handle<reco::IsolatedPixelTrackCandidateCollection> l2colhb;
  edm::Handle<reco::IsolatedPixelTrackCandidateCollection> l2colhe;

  iEvent.getByLabel(l2colLabelHB_,l2colhb);
  hNL2candsHB->Fill(l2colhb->size()+0.001,1);
  
  for (reco::IsolatedPixelTrackCandidateCollection::const_iterator l2it=l2colhb->begin(); l2it!=l2colhb->end(); l2it++)
    {
      if (l2it->p()>l2PThrHB_&&l2it->maxPtPxl()<l2isoHB_&&fabs(l2it->eta())<1.45)
	{
	  firedl2hbs.push_back(l2it);
	  hbl2fired=true;
	}
      for (int i=0; i<15; i++)
	{
	  double l2thrr=3+i;
	  if (l2it->p()>l2thrr&&l2it->maxPtPxl()<l2isoHB_&&fabs(l2it->eta())<1.45) hL2passHB->Fill(l2thrr+0.001,1);
	}
    }

  iEvent.getByLabel(l2colLabelHE_,l2colhe);
  hNL2candsHE->Fill(l2colhe->size()+0.001,1);

  for (reco::IsolatedPixelTrackCandidateCollection::const_iterator l2it=l2colhe->begin(); l2it!=l2colhe->end(); l2it++)
    {
      if (l2it->p()>l2PThrHE_&&l2it->maxPtPxl()<l2isoHE_&&fabs(l2it->eta())>1.4&&fabs(l2it->eta())<2.0)
	{
	  firedl2hes.push_back(l2it);
	  hel2fired=true;
	}
      for (int i=0; i<15; i++)
	{
	  double l2thrr=3+i;
	  if (l2it->p()>l2thrr&&l2it->maxPtPxl()<l2isoHE_&&fabs(l2it->eta())>1.4&&fabs(l2it->eta())<2.0) hL2passHE->Fill(l2thrr+0.001,1);
	}
    }

  if (hbl2fired) 
    {
      hacceptsHB->Fill(2+0.01,1);
      hNPassedL2hbAll->Fill(firedl2hbs.size()+0.001,1);
      for (unsigned int oo=0; oo<firedl2hbs.size(); oo++)
	{
          hl2phiVSeta->Fill(firedl2hbs[oo]->eta(),firedl2hbs[oo]->phi(),1);
	  hl2PHB->Fill(firedl2hbs[oo]->p(),1);
	  hl2etaHB->Fill(firedl2hbs[oo]->eta(),1);
	  hl2phiHB->Fill(firedl2hbs[oo]->phi(),1);
	  hisoPHB->Fill(firedl2hbs[oo]->maxPtPxl(),1);
	  hisoPvsEtaHB->Fill(firedl2hbs[oo]->eta(),firedl2hbs[oo]->maxPtPxl(),1);
	}
    }
  if (hel2fired) 
    {
      hacceptsHE->Fill(2+0.01,1);
      hNPassedL2heAll->Fill(firedl2hes.size()+0.001,1);
      for (unsigned int oo=0; oo<firedl2hes.size(); oo++)
	{
	  hl2phiVSeta->Fill(firedl2hes[oo]->eta(),firedl2hes[oo]->phi(),1);
	  hl2PHE->Fill(firedl2hes[oo]->p(),1);
	  hl2etaHE->Fill(firedl2hes[oo]->eta(),1);
	  hl2phiHE->Fill(firedl2hes[oo]->phi(),1);
	  hisoPHE->Fill(firedl2hes[oo]->maxPtPxl(),1);
	  hisoPvsEtaHE->Fill(firedl2hes[oo]->eta(),firedl2hes[oo]->maxPtPxl(),1);
	}
    }

    if (usePixelTracks_) 
    {
      edm::Handle<reco::TrackCollection> pixTrHB;
      iEvent.getByLabel(pixelTrackLabelHB_,pixTrHB);

      int nCore=0;
      int nIso=0;
      int nSoft=0;

      for (reco::TrackCollection::const_iterator pit=pixTrHB->begin(); pit!=pixTrHB->end(); pit++)
	{
	  pixHBPt->Fill(pit->pt(),1);
	  pixHBP->Fill(pit->p(),1);
	  if (pit->p()>l2PThrHB_) 
	    {
	      nCore++;
	      pixHBCoreEtaPhi->Fill(pit->eta(), pit->phi(),1);
	    }
	  if (pit->p()<l2PThrHB_&&pit->p()>l2isoHB_) 
	    {
	      nIso++;
	      pixHBIsoEtaPhi->Fill(pit->eta(), pit->phi(),1);
	    }
	  if (pit->p()<l2isoHB_) 
	    {
	      nSoft++;
	      pixHBSoftEtaPhi->Fill(pit->eta(), pit->phi(),1); 
	    }
	}
      
      pixHBMultCore->Fill(nCore,1);
      pixHBMultIso->Fill(nIso,1);
      pixHBMultSoft->Fill(nSoft,1);
      
      nCore=0;
      nIso=0;
      nSoft=0;

      edm::Handle<reco::TrackCollection> pixTrHE;
      iEvent.getByLabel(pixelTrackLabelHE_,pixTrHE);
      
      for (reco::TrackCollection::const_iterator pit=pixTrHE->begin(); pit!=pixTrHE->end(); pit++)
	{
	  pixHEPt->Fill(pit->pt(),1);
	  pixHEP->Fill(pit->p(),1);
	  if (pit->p()>l2PThrHE_) 
	    {
	      nCore++;
	      pixHECoreEtaPhi->Fill(pit->eta(), pit->phi(),1);
	    }
	  if (pit->p()<l2PThrHE_&&pit->p()>l2isoHE_) 
	    {
	      nIso++;
	      pixHEIsoEtaPhi->Fill(pit->eta(), pit->phi(),1);
	    }
	  if (pit->p()<l2isoHE_) 
	    {
	      nSoft++;
	      pixHESoftEtaPhi->Fill(pit->eta(), pit->phi(),1); 
	    }
	}
      
      pixHEMultCore->Fill(nCore,1);
      pixHEMultIso->Fill(nIso,1);
      pixHEMultSoft->Fill(nSoft,1);
      
    }
  
  if (!hbl2fired&&!hel2fired) return;
/////

  edm::Handle<reco::IsolatedPixelTrackCandidateCollection> l3colhb;
  edm::Handle<reco::IsolatedPixelTrackCandidateCollection> l3colhe;

  std::vector<reco::IsolatedPixelTrackCandidateCollection::const_iterator> firedl3hbs;
  std::vector<reco::IsolatedPixelTrackCandidateCollection::const_iterator> firedl3hes;
  
    // proceed to L3
  edm::Handle<reco::TrackCollection> recoTr;
  
  if (useReco_)
    {
      iEvent.getByLabel(recoTracksLabel_,recoTr);
    }

//  int nFired=0;

  bool hbl3fired=false;
  bool hel3fired=false;

  //  if (l3filterIndexHB<triggerObj->size()) triggerObj->getObjects(l3filterIndexHB, trigger::TriggerTrack, l3tracksHB);
  // if (l3filterIndexHE<triggerObj->size()) triggerObj->getObjects(l3filterIndexHE, trigger::TriggerTrack, l3tracksHE);

  /*  
  for (trigger::size_type iFilt=0; iFilt!=nFilt; iFilt++) 
    {
      trigger::Keys KEYS1=trEv->filterKeys(iFilt);
      if (KEYS1.size()>0) nFired++;
      if (trEv->filterTag(iFilt)==l3FilterTagHB_) KEYShb=trEv->filterKeys(iFilt);
      if (trEv->filterTag(iFilt)==l3FilterTagHE_) KEYShe=trEv->filterKeys(iFilt);
    }

  trigger::size_type nRegHB=KEYShb.size();
  trigger::size_type nRegHE=KEYShe.size();	
  */
  //check l3 accept
  if (hbl2fired) 
    {
      iEvent.getByLabel(l3colLabelHB_,l3colhb);
      for (reco::IsolatedPixelTrackCandidateCollection::const_iterator l3it=l3colhb->begin(); l3it!=l3colhb->end(); l3it++)
	{
	  bool l2match=false;
	  double minl2l3R=100;
	  int matchedL2=-1;
	  for (unsigned int p=0; p<firedl2hbs.size(); p++)
	    {
	      double dR=deltaR(float(firedl2hbs[p]->eta()),float(firedl2hbs[p]->phi()),l3it->eta(), l3it->phi());
	      if (dR<minl2l3R) 
		{
		  minl2l3R=dR;
		  matchedL2=p;
		}
	    }
	  hL3L2trackRhb->Fill(minl2l3R,1);
          if (minl2l3R!=100) hL3L2PdiffHB->Fill(firedl2hbs[matchedL2]->p()-l3it->p(),1);
	  
	  if (minl2l3R<0.1) l2match=true;
	  
	  if (!l2match) continue;
	  
	  if (produceRatePdep_){
	    for (int i=0; i<50; i++)
	      {
		double pthr=i;
		if (l3it->p()>pthr&&fabs(l3it->eta())<1.479) hHBRateVsThr->Fill(pthr+0.001,1);
	      }
	  }
	  if (l3it->p()<l3PThr_) continue;
	  
	  for (int ii=0; ii<15; ii++)
	    {
	      double l2thrr=3+ii;
	      if (firedl2hbs[matchedL2]->p()>l2thrr) hL2L3passHB->Fill(l2thrr+0.01,1);
	    }
	  
	  firedl3hbs.push_back(l3it);
	  hbl3fired=true;

	  double dphiGJ=fabs(l3it->phi()-phiGJLead);
	  if (dphiGJ>acos(-1)) dphiGJ=2*acos(-1)-dphiGJ;
	  double dR=sqrt(dphiGJ*dphiGJ+pow(l3it->eta()-etaGJLead,2));
	  hPhiToGJ->Fill(dphiGJ,1);
	  hDistToGJ->Fill(dR,1);
	  	  
	  if (fabs(l3it->eta())<0.5) 
	    {
	      if (l3it->p()>40&&l3it->p()<60) hl3P0005->Fill(l3it->p(),1);
	      nl3_0005++;
	    }
	  if (fabs(l3it->eta())>0.5&&fabs(l3it->eta())<1.0) 
	    {
	      nl3_0510++;
	      if (l3it->p()>40&&l3it->p()<60) hl3P0510->Fill(l3it->p(),1);
	    }
	  if (fabs(l3it->eta())>1.0&&fabs(l3it->eta())<1.45) 
	    {
	      nl3_10145++;
	      if (l3it->p()>40&&l3it->p()<60) hl3P10145->Fill(l3it->p(),1);
	    } 
	  if (fabs(l3it->eta())>1.4&&fabs(l3it->eta())<1.45) nl3_overlapHB++;
	  if (useReco_&&recoTr->size()>0)
	    {
	      double minRecoL3dist=100;
	      reco::TrackCollection::const_iterator mrtr;
	      for (reco::TrackCollection::const_iterator rtrit=recoTr->begin(); rtrit!=recoTr->end(); rtrit++)
		{
		  double R=getDist(rtrit->eta(),rtrit->phi(),l3it->eta(),l3it->phi()); 
		  if (R<minRecoL3dist) 
		    {
		      mrtr=rtrit;
		      minRecoL3dist=R;
		    }
		}
	      hOffL3TrackMatch->Fill(minRecoL3dist,1);
	      hOffL3TrackPtRat->Fill(l3it->pt()/mrtr->pt(),1);
	      hOffPvsEta->Fill(mrtr->eta(),mrtr->p(),1);
	    }
	}
    }

  /*
    for (int k=3; k<15; k++)
	{
	if (l2l3passhb[k]) hL2L3passHB->Fill(3+k+0.01,1);
	}
*/

//  hNPassedL2heL3acc->Fill(firedl2hes.size()+0.001,1);
//  hNPassedL2hbL3acc->Fill(firedl2hbs.size()+0.001,1);
//  std::cout<<"size of l2 HE output: "<<firedl2hes.size()<<std::endl;
//  std::cout<<"size of l2 HB output: "<<firedl2hbs.size()<<std::endl;
  if (hel2fired) 
    {
      iEvent.getByLabel(l3colLabelHE_,l3colhe);
      for (reco::IsolatedPixelTrackCandidateCollection::const_iterator l3it=l3colhe->begin(); l3it!=l3colhe->end(); l3it++)
	{
	  double minl2l3R=100;
	  int matchedL2=-1;
	  bool l2match=false;
	  for (unsigned int p=0; p<firedl2hes.size(); p++)
	    {
	      double dR=deltaR(float(firedl2hes[p]->eta()),float(firedl2hes[p]->phi()),l3it->eta(), l3it->phi());
	      if (dR<minl2l3R)
		{
		  minl2l3R=dR;
		  matchedL2=p;
		}
	    }
	  hL3L2trackRhe->Fill(minl2l3R,1);
          if (minl2l3R!=100) hL3L2PdiffHE->Fill(firedl2hes[matchedL2]->p()-l3it->p(),1);
	  
	  if (minl2l3R<0.1) l2match=true;
	  
	  if (!l2match) continue;
	  
	  if (produceRatePdep_){
	    for (int i=0; i<50; i++)
	      {
		double pthr=i;
		if (l3it->p()>pthr&&fabs(l3it->eta())>1.479&&fabs(l3it->eta())<2.0) hHERateVsThr->Fill(pthr+0.001,1);
	      }
	  }
	  
	  if (l3it->p()<l3PThr_) continue;
	  for (int ii=0; ii<15; ii++)
	    {
	      double l2thrr=3+ii;
	      if (firedl2hes[matchedL2]->p()>l2thrr) hL2L3passHE->Fill(l2thrr+0.01,1);
	    }
	  
	  firedl3hes.push_back(l3it);
	  hel3fired=true;
	  
	  if (l3it->p()>20&&l3it->p()<30&&fabs(l3it->eta())>1.479&&fabs(l3it->eta())<1.566) hTower18check2030->Fill(l3it->p(),1); 
	  if (l3it->p()>40&&l3it->p()<60&&fabs(l3it->eta())>1.479&&fabs(l3it->eta())<1.566) hTower18check4060->Fill(l3it->p(),1);
	  
	  if (fabs(l3it->eta())>1.4&&fabs(l3it->eta())<1.45) nl3_overlapHE++;
	  
	  if (fabs(l3it->eta())<2.0&&fabs(l3it->eta())>1.5) 
	    {
	      nl3_14520++;
	      if (l3it->p()>40&&l3it->p()<60) hl3P14520->Fill(l3it->p(),1);
	    }
	  
	}
    }

  if (hbl3fired) 
    {
      hacceptsHB->Fill(3+0.01,1);
      hNPassedL2hbL3acc->Fill(firedl2hbs.size()+0.001,1);
      for (unsigned int oo=0; oo<firedl3hbs.size(); oo++)
	{
	  hl3PHB->Fill(firedl2hbs[oo]->p(),1);
	  hl3etaHB->Fill(firedl2hbs[oo]->eta(),1);
	  hl3phiHB->Fill(firedl2hbs[oo]->phi(),1);
	}
    }
  if (hel3fired) 
    {
      hacceptsHE->Fill(3+0.01,1);
      hNPassedL2heL3acc->Fill(firedl2hes.size()+0.001,1);
      for (unsigned int oo=0; oo<firedl3hes.size(); oo++)
	{
	  hl3PHE->Fill(firedl2hes[oo]->p(),1);
	  hl3etaHE->Fill(firedl2hes[oo]->eta(),1);
	  hl3phiHE->Fill(firedl2hes[oo]->phi(),1);
	}
    }
  
  // event purity check

  bool pureHB=false;
  bool pureHE=false;
  bool pureHBHE=false;

  edm::Handle<edm::TriggerResults> HLTR;
  iEvent.getByLabel(hlTriggerResults_,HLTR);

  edm::TriggerNames triggerNames_;
  triggerNames_.init(*HLTR);
  std::vector<std::string>  hlNames_=triggerNames_.triggerNames();
  const unsigned int nTrig(hlNames_.size());

  std::vector<bool> hlAccept;
  for (unsigned int k=0; k<nTrig; k++)
    {
      hlAccept.push_back(false);
    }
  int nFiredTrig=0;
  int hbInd=0;
  int heInd=0;
  overlapNames.clear();
  for (unsigned int k=0; k<nTrig; k++)
    {
      if (hlNames_[k]==trigHB_) hbInd=k;
      if (hlNames_[k]==trigHE_) heInd=k;
      if (HLTR->accept(k))
        {
          bool notSkip=true;
	  for (unsigned int kk=0; kk<excludeOverl_.size(); kk++) {if (hlNames_[k]==excludeOverl_[kk]) notSkip=false;} 
	  if (notSkip) nFiredTrig++;
	  //	  if (hbl3fired||hel3fired) std::cout<<hlNames_[k]<<std::endl;
          hlAccept[k]=true;
        }
      if (nTotal==1) overlapNames.push_back(hlNames_[k]);
      if (hbl3fired&&hlAccept[k]) overlapsHB[k]++;
      if (hel3fired&&hlAccept[k]) overlapsHE[k]++;
    }

  if (hbl3fired&&hlAccept[hbInd]&&nFiredTrig==1) pureHB=true;
  if (hel3fired&&hlAccept[heInd]&&nFiredTrig==1) pureHE=true;
  if (hbl3fired&&hel3fired&&hlAccept[hbInd]&&hlAccept[heInd]&&nFiredTrig==2) pureHBHE=true;

  std::vector<reco::IsolatedPixelTrackCandidateRef> l3tracksHB;
  std::vector<reco::IsolatedPixelTrackCandidateRef> l3tracksHE;
  
  edm::InputTag l3TagHB = edm::InputTag(l3FilterLabelHB_, "",hltProcess_);
  edm::InputTag l3TagHE = edm::InputTag(l3FilterLabelHE_, "",hltProcess_);
  
  trigger::size_type l3filterIndexHB=triggerObj->filterIndex(l3TagHB);
  trigger::size_type l3filterIndexHE=triggerObj->filterIndex(l3TagHE);
  
  if ((pureHBHE||pureHE||pureHB)&&(hbl3fired||hel3fired))
    {
      nHLTL3acceptsPure++;
      if (pureHB)
	{
	  for (unsigned int iTr=0; iTr<firedl3hbs.size(); iTr++)      
	    { 
	      if (firedl3hbs[iTr]->p()<l3PThr_) continue;
	      if (fabs(firedl3hbs[iTr]->eta())<0.5) purnl3_0005++;
	      if (fabs(firedl3hbs[iTr]->eta())>0.5&&fabs(firedl3hbs[iTr]->eta())<1.0) purnl3_0510++;
	      if (fabs(firedl3hbs[iTr]->eta())>1.0&&fabs(firedl3hbs[iTr]->eta())<1.45) purnl3_10145++;
	    }
	}
      if (pureHE)
	{
	  for (unsigned int iTr=0; iTr<firedl3hes.size(); iTr++)                   
	    { 
              if (firedl3hes[iTr]->p()<l3PThr_) continue;
	      if (fabs(firedl3hes[iTr]->eta())>1.4&&fabs(firedl3hes[iTr]->eta())<1.45) purnl3_overlapHE++;
	      if (fabs(firedl3hes[iTr]->eta())<2.0&&fabs(firedl3hes[iTr]->eta())>1.45) purnl3_14520++;
	    }
	}
    }

  if (hbl3fired||hel3fired) nHLTL3accepts++;

  if (l1pass&&hbl3fired) 
    {
      nL3accHB++;
      hacceptsHB->Fill(3+0.0001,1);
      hNPassedL2hbL3acc->Fill(firedl2hbs.size()+0.001,1);
    }

  if(l1pass&&hel3fired) 
    {
      hNPassedL2heL3acc->Fill(firedl2hes.size()+0.001,1);
      hacceptsHE->Fill(3+0.0001,1);
      nL3accHE++;
    }  

//  if (l1pass&&(hel3fired||hbl3fired)) 
//	{
//	runsAcc.push_back(runNumm);
//        evtsAcc.push_back(evtNumm);
//	}

  if (!l1pass||!(hel3fired||hbl3fired)) return;
}

void ValHcalIsoTrackHLT::beginJob()
{
  dbe_ = edm::Service<DQMStore>().operator->();
  dbe_->setCurrentFolder(folderName_);

  hTower18check2030=dbe_->book1D("hTower18check2030","hTower18check2030",1000,15,35);
  hTower18check4060=dbe_->book1D("hTower18check4060","hTower18check4060",1000,35,65);

  hL2L3minDR=dbe_->book1D("hL2L3minDR","minimum distance from L2 object to L3 object",1000,0,0.5);

  hHBRateVsThr=dbe_->book1D("hHBRatesVsThr","HB rate vs. L3 threshold",100,0,100);
  hHERateVsThr=dbe_->book1D("hHERatesVsThr","HE rate vs. L3 threshold",100,0,100);
  
  hPhiToGJ=dbe_->book1D("hPhiToGJ","delta phi to nearest genJet",100,0,4);

  hDistToGJ=dbe_->book1D("hDistToGJ","distance to nearest genJet",100,0,10);

  hL3L2trackRhb=dbe_->book1D("hL3L2trackRhb","minimum R from L2 to L3 object in HB region",1000,0,1);
  hL3L2trackRhb->setAxisTitle("R(eta,phi)",1);

  hL3L2trackRhe=dbe_->book1D("hL3L2trackRhe","minimum R from L2 to L3 object in HE region",1000,0,1);
  hL3L2trackRhe->setAxisTitle("R(eta,phi)",1);
 
  hL3L2PdiffHB=dbe_->book1D("hL3L2PdiffHB","ratio of L2 to L3 P measurement in HB region",1000,-50,50);
  hL3L2PdiffHB->setAxisTitle("P_L2-P_L3",1);
  
  hL3L2PdiffHE=dbe_->book1D("hL3L2PdiffHE","ratio of L2 to L3 P measurement in HE region",1000,-50,50);
  hL3L2PdiffHE->setAxisTitle("P_L2-P_L3",1);
  
  hl3PHB=dbe_->book1D("hl3PHB","P of L3 objects, HB region",1000,0,100);
  hl3PHB->setAxisTitle("P(GeV)",1);

  hl3PHE=dbe_->book1D("hl3PHE","P of L3 objects, HE region",1000,0,100);
  hl3PHE->setAxisTitle("P(GeV)",1);

  hl3P0005=dbe_->book1D("hl3P0005","P of L3 objects with |eta|<0.5",1000,0,100);
  hl3P0005->setAxisTitle("P(GeV)",1);

  hl3P0510=dbe_->book1D("hl3P0510","P of L3 objects with 0.5<|eta|<1.0",1000,0,100);
  hl3P0510->setAxisTitle("P(GeV)",1);

  hl3P10145=dbe_->book1D("hl3P10145","P of L3 objects with 1.0<|eta|<1.45",1000,0,100);
  hl3P10145->setAxisTitle("P(GeV)",1);

  hl3P14520=dbe_->book1D("hl3P14520","P of L3 objects with 1.45<|eta|<2.0",1000,0,100);
  hl3P14520->setAxisTitle("P(GeV)",1);

  hl3etaHE=dbe_->book1D("hl3etaHE","eta of L3 objects, HE region",500,-2.5,2.5);
  hl3etaHE->setAxisTitle("eta",1);

  hl3phiHE=dbe_->book1D("hl3phiHE","phi of L3 objects, HE region",70,-3.5,3.5);
  hl3phiHE->setAxisTitle("phi(rad)",1);

  hl2PHE=dbe_->book1D("hl2PHE","P of L2 objects, HE region",1000,0,1000);
  hl2PHE->setAxisTitle("P(GeV)",1);

  hl2etaHE=dbe_->book1D("hl2etaHE","eta of L2 objects, HE region",50,-2.5,2.5);
  hl2etaHE->setAxisTitle("eta",1);

  hl2phiHE=dbe_->book1D("hl2phiHE","phi of L2 objects, HE region",70,-3.5,3.5);
  hl2phiHE->setAxisTitle("phi(rad)",1);

  hl2phiVSeta=dbe_->book2D("hl2phiVSeta","phi vs. eta of L2 objects",50,-2.5,2.5,70,-3.5,3.5);
  hl2phiVSeta->setAxisTitle("eta",1);
  hl2phiVSeta->setAxisTitle("phi(rad)",2);

  hisoPHE=dbe_->book1D("hisoPHE","isolation P, HE region",100,0,5.5);
  hisoPHE->setAxisTitle("iso P (GeV)",1);

  hisoPvsEtaHE=dbe_->book2D("hisoPvsEta","isolation P vs Eta, HE region",8,-2,2,100,0,5.5);
  hisoPvsEtaHE->setAxisTitle("eta",1);
  hisoPvsEtaHE->setAxisTitle("iso P (GeV)",2);

  hl3etaHB=dbe_->book1D("hl3etaHB","eta of L3 objects, HB region",500,-2.5,2.5);
  hl3etaHB->setAxisTitle("eta",1);

  hl3phiHB=dbe_->book1D("hl3phiHB","phi of L3 objects, HB region",70,-3.5,3.5);
  hl3phiHB->setAxisTitle("phi(rad)",1);

  hl2PHB=dbe_->book1D("hl2PHB","P of L2 objects, HB region",1000,0,1000);
  hl2PHB->setAxisTitle("P(GeV)",1);

  hl2etaHB=dbe_->book1D("hl2etaHB","eta of L2 objects, HB region",50,-2.5,2.5);
  hl2etaHB->setAxisTitle("eta",1);

  hl2phiHB=dbe_->book1D("hl2phiHB","phi of L2 objects, HB region",70,-3.5,3.5);
  hl2phiHB->setAxisTitle("phi(rad)",1);

  hisoPHB=dbe_->book1D("hisoPHB","isolation P, HB region",100,0,5.5);
  hisoPHB->setAxisTitle("iso P (GeV)",1);

  hisoPvsEtaHB=dbe_->book2D("hisoPvsEtaHB","isolation P vs Eta, HB region",8,-2,2,100,0,5.5);
  hisoPvsEtaHB->setAxisTitle("eta",1);
  hisoPvsEtaHB->setAxisTitle("iso P (GeV)",2);

  hacceptsHB=dbe_->book1D("hacceptsHB","Number of accepts at each level, HB",4,0,4);
  hacceptsHB->setAxisTitle("selection level",1);

  hacceptsHE=dbe_->book1D("hacceptsHE","Number of accepts at each level, HE",4,0,4);
  hacceptsHE->setAxisTitle("selection level",1);

  hOffL3TrackMatch=dbe_->book1D("hOffL3TrackMatch","Distance from L3 object to nearest reco track, HE",200,0,0.5);
  hOffL3TrackMatch->setAxisTitle("R(eta,phi)",1);

  hOffL3TrackPtRat=dbe_->book1D("hOffL3TrackPtRat","Ratio of pT: L3/reco, HE",100,0,10);
  hOffL3TrackPtRat->setAxisTitle("ratio L3/offline",1);

  hOffPvsEta=dbe_->book2D("hOffPvsEta","Distribution of offline track energy vs eta, HE",25,-2.5,2.5,100,0,100);
  hOffPvsEta->setAxisTitle("eta",1);
  hOffPvsEta->setAxisTitle("E(GeV)",2);

  hL2passHB=dbe_->book1D("hL2passHB","L2 rate vs. P threshold, HB region",15,3,18);

  hL2passHE=dbe_->book1D("hL2passHE","L2 rate vs. P threshold, HE region",15,3,18);
 
  hL2L3passHB=dbe_->book1D("hL2L3passHB","L3 rate vs. L2 P threshold, HB region",15,3,18);

  hL2L3passHE=dbe_->book1D("hL2L3passHE","L3 rate vs. L2 P threshold, HE region",15,3,18);

  hNPassedL2heL3acc=dbe_->book1D("hNPassedL2heL3acc","multiplicity of accepted L2 objects for L3 accepted events, HE region",20,0,20);
  hNPassedL2hbL3acc=dbe_->book1D("hNPassedL2hbL3acc","multiplicity of accepted L2 objects for L3 accepted events, HB region",20,0,20);

  hNPassedL2heAll=dbe_->book1D("hNPassedL2heAll","multiplicity of L2 objects for L2 accepted events, HE region",20,0,20);
  hNPassedL2hbAll=dbe_->book1D("hNPassedL2hbAll","multiplicity of L2 objects for L2 accepted events, HB region",20,0,20);

  hNL2candsHB=dbe_->book1D("hNL2candsHB","multiplicity of L2 candidates, HB region",20,0,20);
  hNL2candsHE=dbe_->book1D("hNL2candsHE","multiplicity of L2 candidates, HE region",20,0,20);
 
  hpTgenLead=dbe_->book1D("hpTgenLead","pT of leading genJet",100,0,100);
  
  hpTgenLeadL1=dbe_->book1D("hpTgenLeadL1","pT of leading genJet, event passing L1 selection",100,0,100);

  hpTgenNext=dbe_->book1D("hpTgenNext","pT of next-to-leadin genJet",100,0,100);

  hpTgenNextL1=dbe_->book1D("hpTgenNextL1","pT of next-to-leading genJets, event passing L1 selection",100,0,100);

  pixHBPt=dbe_->book1D("pixHBPt","pixel track pT distribution, HB region",1000,0,100);
  pixHBP=dbe_->book1D("pixHBP","pixel track P distribution, HB region",1000,0,100);
  pixHBCoreEtaPhi=dbe_->book2D("pixHBCoreEtaPhi","core-like pixel track eta-phi distribution, HB region",50,-2.5,2.5,70,-3.5,3.5);
  pixHBIsoEtaPhi=dbe_->book2D("pixHBIsoEtaPhi","intermediate pixel track eta-phi distribution, HB region",50,-2.5,2.5,70,-3.5,3.5);
  pixHBSoftEtaPhi=dbe_->book2D("pixHBSoftEtaPhi","soft pixel track eta-phi distribution, HB region",50,-2.5,2.5,70,-3.5,3.5);
  pixHBMultCore=dbe_->book1D("pixHBMultiCore","multiplicity of core-like pixel tracks, HB",50,0,50);
  pixHBMultIso=dbe_->book1D("pixHBMultiIso","multiplicity of intermediate pixel tracks, HB",50,0,50);
  pixHBMultSoft=dbe_->book1D("pixHBMultiSoft","multiplicity of soft pixel tracks, HB",50,0,50);
  
  pixHEPt=dbe_->book1D("pixHEPt","pixel track pT distribution, HE region",1000,0,100);
  pixHEP=dbe_->book1D("pixHEP","pixel track P distribution, HE region",1000,0,100);
  pixHECoreEtaPhi=dbe_->book2D("pixHECoreEtaPhi","core-like pixel track eta-phi distribution, HE region",50,-2.5,2.5,70,-3.5,3.5);
  pixHEIsoEtaPhi=dbe_->book2D("pixHEIsoEtaPhi","intermediate pixel track eta-phi distribution, HE region",50,-2.5,2.5,70,-3.5,3.5);
  pixHESoftEtaPhi=dbe_->book2D("pixHESoftEtaPhi","soft pixel track eta-phi distribution, HE region",50,-2.5,2.5,70,-3.5,3.5);
  pixHEMultCore=dbe_->book1D("pixHEMultiCore","multiplicity of core-like pixel tracks, HE",50,0,50);
  pixHEMultIso=dbe_->book1D("pixHEMultiIso","multiplicity of intermediate pixel tracks, HE",50,0,50);
  pixHEMultSoft=dbe_->book1D("pixHEMultiSoft","multiplicity of soft pixel tracks, HE",50,0,50);
}

void ValHcalIsoTrackHLT::endJob() {

if(dbe_) 
  {  
    if (saveToRootFile_) dbe_->save(outRootFileName_);
  }
}

DEFINE_FWK_MODULE(ValHcalIsoTrackHLT);
