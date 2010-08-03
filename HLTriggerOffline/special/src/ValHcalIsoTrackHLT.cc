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
  virtual void endJob() ;

  bool produceRates_;
  double sampleXsec_;
  double lumi_;
  std::string outTxtFileName_;
  std::string folderName_;
  bool saveToRootFile_;
  std::string outRootFileName_;
  edm::InputTag hltEventTag_;
  edm::InputTag hltFilterTag_;
  std::vector<edm::InputTag> l1extraJetTag_;
  std::vector<std::string> l1seedNames_;
  edm::InputTag gtDigiLabel_; 
  bool useReco_;
  edm::InputTag recoTracksLabel_;
  bool checkL2_;
  edm::InputTag l2colLabel_; 

  bool checkL1eff_;
  edm::InputTag genJetsLabel_;

  bool produceRatePdep_;

  bool doL1Prescaling_;
  
  MonitorElement* hl3Pt;
  MonitorElement* hL3L2trackMatch;
  MonitorElement* hL3L2pTrat;
  MonitorElement* hl3Pt0005;
  MonitorElement* hl3Pt0510;
  MonitorElement* hl3Pt1015;
  MonitorElement* hl3Pt1520;
  MonitorElement* hl3P0005;
  MonitorElement* hl3P0510;
  MonitorElement* hl3P1015;
  MonitorElement* hl3P1520;
  MonitorElement* hl3eta;
  MonitorElement* hl3phi;

  MonitorElement* hl3pVsEta;

  MonitorElement* hOffL3TrackMatch;
  MonitorElement* hOffL3TrackPtRat;

  MonitorElement* hl2eta;
  MonitorElement* hl2phi;
  MonitorElement* hl2pT;
  MonitorElement* hl2pVsEta;
  MonitorElement* hisopT;
  MonitorElement* hisopTvsEta;

  MonitorElement* haccepts;
  MonitorElement* hOffPvsEta;
  MonitorElement* hpTgenLead;
  MonitorElement* hpTgenLeadL1;
  MonitorElement* hpTgenNext;
  MonitorElement* hpTgenNextL1;
  MonitorElement* hLeadTurnOn;
  MonitorElement* hNextToLeadTurnOn;

  MonitorElement* hRateVsThr;

  MonitorElement* hPhiToGJ;
  MonitorElement* hDistToGJ;

  std::vector<int> l1counter;

  std::ofstream txtout;

  int nTotal;
  int nL1accepts;
  int nHLTL2accepts;
  int nHLTL3accepts;
  int nHLTL3acceptsPure;

  int nl3_0005;
  int nl3_0510;
  int nl3_1015;
  int nl3_1520;
  
  int purnl3_0005;
  int purnl3_0510;
  int purnl3_1015;
  int purnl3_1520;

  double hltPThr_;
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
  lumi_=iConfig.getParameter<double>("luminosity");
  outTxtFileName_=iConfig.getParameter<std::string>("outputTxtFileName");
  
  folderName_ = iConfig.getParameter<std::string>("folderName");
  saveToRootFile_=iConfig.getParameter<bool>("SaveToRootFile");
  outRootFileName_=iConfig.getParameter<std::string>("outputRootFileName");

  hltEventTag_=iConfig.getParameter<edm::InputTag>("hltTriggerEventLabel");
  hltFilterTag_=iConfig.getParameter<edm::InputTag>("hltL3FilterLabel");

  l1extraJetTag_=iConfig.getParameter<std::vector<edm::InputTag> >("hltL1extraJetLabel");
  gtDigiLabel_=iConfig.getParameter<edm::InputTag>("gtDigiLabel");
  checkL1eff_=iConfig.getParameter<bool>("CheckL1TurnOn");
  genJetsLabel_=iConfig.getParameter<edm::InputTag>("genJetsLabel");
  l1seedNames_=iConfig.getParameter<std::vector<std::string> >("l1seedNames");
  useReco_=iConfig.getParameter<bool>("useReco");
  recoTracksLabel_=iConfig.getParameter<edm::InputTag>("recoTracksLabel");
  checkL2_=iConfig.getParameter<bool>("DebugL2");
  l2colLabel_=iConfig.getParameter<edm::InputTag>("L2producerLabel");
  
  produceRatePdep_=iConfig.getParameter<bool>("produceRatePdep");

  hltPThr_=iConfig.getUntrackedParameter<double>("l3momThreshold",10);

  doL1Prescaling_=iConfig.getParameter<bool>("doL1Prescaling");
  
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
  
  nl3_0005=0;
  nl3_0510=0;
  nl3_1015=0;
  nl3_1520=0;
  
  purnl3_0005=0;
  purnl3_0510=0;
  purnl3_1015=0;
  purnl3_1520=0;
}


ValHcalIsoTrackHLT::~ValHcalIsoTrackHLT()
{
  if (produceRates_)
    {
      double sampleRate=(lumi_)*(sampleXsec_*1E-36);
      double l1Rate=nL1accepts*pow(nTotal,-1)*sampleRate;
      double hltRate=nHLTL3accepts*pow(nL1accepts,-1)*l1Rate;
      double hltRatePure=nHLTL3acceptsPure*pow(nL1accepts,-1)*l1Rate;
      
      double l1rateError=l1Rate/sqrt(nL1accepts);
      double hltRateError=hltRate/sqrt(nHLTL3accepts);
      double hltRatePureError=hltRatePure/sqrt(nHLTL3acceptsPure);
      
      double rate_0005=nl3_0005*pow(nL1accepts,-1)*l1Rate;
      double rate_0510=nl3_0510*pow(nL1accepts,-1)*l1Rate;
      double rate_1015=nl3_1015*pow(nL1accepts,-1)*l1Rate;
      double rate_1520=nl3_1520*pow(nL1accepts,-1)*l1Rate;
      
      double prate_0005=purnl3_0005*pow(nL1accepts,-1)*l1Rate;
      double prate_0510=purnl3_0510*pow(nL1accepts,-1)*l1Rate;
      double prate_1015=purnl3_1015*pow(nL1accepts,-1)*l1Rate;
      double prate_1520=purnl3_1520*pow(nL1accepts,-1)*l1Rate;
      
      txtout<<std::setw(50)<<std::left<<"sample xsec(pb)"<<sampleXsec_<<std::endl;
      txtout<<std::setw(50)<<std::left<<"lumi(cm^-2*s^-1)"<<lumi_<<std::endl;
      txtout<<std::setw(50)<<std::left<<"Events processed/rate(Hz)"<<nTotal<<"/"<<sampleRate<<std::endl;
      txtout<<std::setw(50)<<std::left<<"L1 accepts/(rate+-error (Hz))"<<nL1accepts<<"/("<<l1Rate<<"+-"<<l1rateError<<")"<<std::endl;
      txtout<<std::setw(50)<<std::left<<"HLTL3accepts/(rate+-error (Hz))"<<nHLTL3accepts<<"/("<<hltRate<<"+-"<<hltRateError<<")"<<std::endl;
      txtout<<std::setw(50)<<std::left<<"L3 acc. |eta|<0.5 / rate"<<nl3_0005<<" / "<<rate_0005<<std::endl;
      txtout<<std::setw(50)<<std::left<<"L3 acc. |eta|>0.5 && |eta|<1.0 / rate"<<nl3_0510<<" / "<<rate_0510<<std::endl;
      txtout<<std::setw(50)<<std::left<<"L3 acc. |eta|>1.0 && |eta|<1.5 / rate"<<nl3_1015<<" / "<<rate_1015<<std::endl;
      txtout<<std::setw(50)<<std::left<<"L3 acc. |eta|>1.5 && |eta|<2.0 / rate"<<nl3_1520<<" / "<<rate_1520<<std::endl;
      txtout<<"\n"<<std::endl;
      txtout<<std::setw(50)<<std::left<<"HLTL3acceptsPure/(rate+-error (Hz))"<<nHLTL3acceptsPure<<"/("<<hltRatePure<<"+-"<<hltRatePureError<<")"<<std::endl; 
      txtout<<std::setw(50)<<std::left<<"pure L3 acc. |eta|<0.5 / rate"<<purnl3_0005<<" / "<<prate_0005<<std::endl;
      txtout<<std::setw(50)<<std::left<<"pure L3 acc. |eta|>0.5 && |eta|<1.0 / rate"<<purnl3_0510<<" / "<<prate_0510<<std::endl;
      txtout<<std::setw(50)<<std::left<<"pure L3 acc. |eta|>1.0 && |eta|<1.5 / rate"<<purnl3_1015<<" / "<<prate_1015<<std::endl;
      txtout<<std::setw(50)<<std::left<<"pure L3 acc. |eta|>1.5 && |eta|<2.0 / rate"<<purnl3_1520<<" / "<<prate_1520<<std::endl;
    }
}

void ValHcalIsoTrackHLT::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  nTotal++;
  haccepts->Fill(0.0001,1);

  double phiGJLead=-10000;
  double etaGJLead=-10000;

  bool l1pass=false;

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
     
  if (!l1pass) return;
  else haccepts->Fill(1+0.0001,1);
      
  nL1accepts++;

  edm::Handle<reco::IsolatedPixelTrackCandidateCollection> l2col;
  if (checkL2_)
    {
      iEvent.getByLabel(l2colLabel_,l2col);
    }
  
  edm::Handle<trigger::TriggerEvent> trEv;
  iEvent.getByLabel(hltEventTag_,trEv);
  
  edm::Handle<reco::TrackCollection> recoTr;
  
  if (useReco_)
    {
      iEvent.getByLabel(recoTracksLabel_,recoTr);
    }

  const trigger::TriggerObjectCollection& TOCol(trEv->getObjects());
  
  trigger::Keys KEYS;
  const trigger::size_type nFilt(trEv->sizeFilters());

  int nFired=0;

  bool passl3=false;

  for (trigger::size_type iFilt=0; iFilt!=nFilt; iFilt++) 
    {
      trigger::Keys KEYS1=trEv->filterKeys(iFilt);
      if (KEYS1.size()>0) nFired++;
      if (trEv->filterTag(iFilt)==hltFilterTag_) KEYS=trEv->filterKeys(iFilt);
    }

  trigger::size_type nReg=KEYS.size();

  if (nFired==2&&nReg>0)  
    {
      nHLTL3acceptsPure++;
      for (trigger::size_type iReg=0; iReg<nReg; iReg++)
	{
	  const trigger::TriggerObject& TObj(TOCol[KEYS[iReg]]);
          if (TObj.pt()*cosh(TObj.eta())<10) continue;
	  if (fabs(TObj.eta())<0.5) purnl3_0005++;
	  if (fabs(TObj.eta())>0.5&&fabs(TObj.eta())<1.0) purnl3_0510++;
	  if (fabs(TObj.eta())>1.0&&fabs(TObj.eta())<1.5) purnl3_1015++;
	  if (fabs(TObj.eta())<2.0&&fabs(TObj.eta())>1.5) purnl3_1520++;
	}
    }

  for (trigger::size_type iReg=0; iReg<nReg; iReg++)
    {
      const trigger::TriggerObject& TObj(TOCol[KEYS[iReg]]);

      if (produceRatePdep_){
	for (int i=0; i<50; i++)
	  {
	    double pthr=5+i;
	    if (TObj.pt()*cosh(TObj.eta())>pthr&&fabs(TObj.eta())<1.0) hRateVsThr->Fill(pthr+0.001,1);
	  }
      }

      if (TObj.pt()*cosh(TObj.eta())<hltPThr_) continue;
      
      passl3=true;

      double dphiGJ=fabs(TObj.phi()-phiGJLead);
      if (dphiGJ>acos(-1)) dphiGJ=2*acos(-1)-dphiGJ;
      double dR=sqrt(dphiGJ*dphiGJ+pow(TObj.eta()-etaGJLead,2));
      hPhiToGJ->Fill(dphiGJ,1);
      hDistToGJ->Fill(dR,1);

      hl3eta->Fill(TObj.eta(),1);
      hl3phi->Fill(TObj.phi(),1);
      if (fabs(TObj.eta())<0.5) 
	{
	hl3P0005->Fill(cosh(TObj.eta())*TObj.pt(),1);
	hl3Pt0005->Fill(TObj.pt(),1);
	nl3_0005++;
	}
      if (fabs(TObj.eta())>0.5&&fabs(TObj.eta())<1.0) 
	{
	nl3_0510++;
	hl3P0510->Fill(cosh(TObj.eta())*TObj.pt(),1);
	hl3Pt0510->Fill(TObj.pt(),1);
	}
      if (fabs(TObj.eta())>1.0&&fabs(TObj.eta())<1.5) 
	{
	nl3_1015++;
	hl3P1015->Fill(cosh(TObj.eta())*TObj.pt(),1);
        hl3Pt1015->Fill(TObj.pt(),1);
        } 
      if (fabs(TObj.eta())<2.0&&fabs(TObj.eta())>1.5) 
	{
	nl3_1520++;
	hl3P1520->Fill(cosh(TObj.eta())*TObj.pt(),1);
        hl3Pt1520->Fill(TObj.pt(),1);
        }
      if (l2col->size()==0) continue;
      double l2l3d=100;
      reco::IsolatedPixelTrackCandidateCollection::const_iterator l2match;
      for (reco::IsolatedPixelTrackCandidateCollection::const_iterator eptit=l2col->begin(); eptit!=l2col->end(); eptit++) 
	{
	  hl2pT->Fill(eptit->track()->pt(),1);
	  hl2eta->Fill(eptit->track()->eta(),1);
	  hl2phi->Fill(eptit->track()->phi(),1);
	  hl2pVsEta->Fill(eptit->track()->eta(),eptit->track()->p(),1);
	  hisopT->Fill(eptit->maxPtPxl(),1);
	  hisopTvsEta->Fill(eptit->track()->eta(),eptit->maxPtPxl(),1);

	  double R=getDist(eptit->eta(), eptit->phi(), TObj.eta(), TObj.phi());
	  if (R<l2l3d) 
	    {
	      l2match=eptit;
	      l2l3d=R;
	    }
	}

      hL3L2trackMatch->Fill(l2l3d,1);
      hL3L2pTrat->Fill(l2match->pt()/TObj.pt(),1);
      hl3Pt->Fill(TObj.pt(),1);

      if (useReco_&&recoTr->size()>0)
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
    }

if (passl3) nHLTL3accepts++;
if (passl3) haccepts->Fill(2+0.0001,1);

if (!l1pass||!passl3) return;

}

void ValHcalIsoTrackHLT::beginJob()
{
  dbe_ = edm::Service<DQMStore>().operator->();
  dbe_->setCurrentFolder(folderName_);

  hRateVsThr=dbe_->book1D("hRatesVsThr","hRateVsThr",100,0,100);

  hPhiToGJ=dbe_->book1D("hPhiToGJ","hPhiToGJ",100,0,4);

  hDistToGJ=dbe_->book1D("hDistToGJ","hDistToGJ",100,0,10);

  hL3L2trackMatch=dbe_->book1D("hL3L2trackMatch","R from L3 object to L2 object ",1000,0,1);
  hL3L2trackMatch->setAxisTitle("R(eta,phi)",1);
 
  hL3L2pTrat=dbe_->book1D("hL3L2pTrat","ratio of L2 to L3 measurement",1000,0,10);
  hL3L2pTrat->setAxisTitle("pT_L2/pT_L3",1);

  hl3Pt=dbe_->book1D("hl3Pt","pT of L3 objects",1000,0,100);
  hl3Pt->setAxisTitle("pT(GeV)",1);

  hl3Pt0005=dbe_->book1D("hl3Pt0005","hl3Pt0005",1000,0,100);
  hl3Pt0005->setAxisTitle("pT(GeV)",1);

  hl3Pt0510=dbe_->book1D("hl3Pt0510","hl3Pt0510",1000,0,100);
  hl3Pt0510->setAxisTitle("pT(GeV)",1);

  hl3Pt1015=dbe_->book1D("hl3Pt1015","hl3Pt1015",1000,0,100);
  hl3Pt1015->setAxisTitle("pT(GeV)",1);

  hl3Pt1520=dbe_->book1D("hl3Pt1520","hl3Pt1520",1000,0,100);
  hl3Pt1520->setAxisTitle("pT(GeV)",1);

  hl3P0005=dbe_->book1D("hl3P0005","hl3P0005",1000,0,100);
  hl3P0005->setAxisTitle("P(GeV)",1);

  hl3P0510=dbe_->book1D("hl3P0510","hl3P0510",1000,0,100);
  hl3P0510->setAxisTitle("P(GeV)",1);

  hl3P1015=dbe_->book1D("hl3P1015","hl3P1015",1000,0,100);
  hl3P1015->setAxisTitle("P(GeV)",1);

  hl3P1520=dbe_->book1D("hl3P1520","hl3P1520",1000,0,100);
  hl3P1520->setAxisTitle("P(GeV)",1);

  hl3eta=dbe_->book1D("hl3eta","eta of L3 objects",50,-2.5,2.5);
  hl3eta->setAxisTitle("eta",1);

  hl3phi=dbe_->book1D("hl3phi","phi of L3 objects",70,-3.5,3.5);
  hl3phi->setAxisTitle("phi(rad)",1);

  hl2pT=dbe_->book1D("hl2pT","pT of L2 objects",1000,0,1000);
  hl2pT->setAxisTitle("pT(GeV)",1);

  hl2eta=dbe_->book1D("hl2eta","eta of L2 objects",50,-2.5,2.5);
  hl2eta->setAxisTitle("eta",1);

  hl2phi=dbe_->book1D("hl2phi","phi of L2 objects",70,-3.5,3.5);
  hl2phi->setAxisTitle("phi(rad)",1);

  hisopT=dbe_->book1D("hisopT","isolation pT",100,0,5.5);
  hisopT->setAxisTitle("iso pT (GeV)",1);

  hisopTvsEta=dbe_->book2D("hisopTvsEta","isolation pT vs Eta",8,-2,2,100,0,5.5);
  hisopTvsEta->setAxisTitle("eta",1);
  hisopTvsEta->setAxisTitle("iso pT (GeV)",2);

  hl2pVsEta=dbe_->book2D("hl2pVsEta","Distribution of l2 track energy vs eta",25,-2.5,2.5,100,0,100);
  hl2pVsEta->setAxisTitle("eta",1);
  hl2pVsEta->setAxisTitle("E(GeV)",2);

  haccepts=dbe_->book1D("haccepts","Number of accepts at each level",3,0,3);
  haccepts->setAxisTitle("selection level",1);

  hOffL3TrackMatch=dbe_->book1D("hOffL3TrackMatch","Distance from L3 object to offline track",200,0,0.5);
  hOffL3TrackMatch->setAxisTitle("R(eta,phi)",1);

  hOffL3TrackPtRat=dbe_->book1D("hOffL3TrackPtRat","Ratio of pT: L3/offline",100,0,10);
  hOffL3TrackPtRat->setAxisTitle("ratio L3/offline",1);

  hOffPvsEta=dbe_->book2D("hOffPvsEta","Distribution of offline track energy vs eta",25,-2.5,2.5,100,0,100);
  hOffPvsEta->setAxisTitle("eta",1);
  hOffPvsEta->setAxisTitle("E(GeV)",2);

  hpTgenLead=dbe_->book1D("hpTgenLead","hpTgenLead",100,0,100);
  
  hpTgenLeadL1=dbe_->book1D("hpTgenLeadL1","hpTgenLeadL1",100,0,100);

  hpTgenNext=dbe_->book1D("hpTgenNext","hpTgenNext",100,0,100);

  hpTgenNextL1=dbe_->book1D("hpTgenNextL1","hpTgenNextL1",100,0,100);
}

void ValHcalIsoTrackHLT::endJob() {

if(dbe_) 
  {  
    if (saveToRootFile_) dbe_->save(outRootFileName_);
  }
}

DEFINE_FWK_MODULE(ValHcalIsoTrackHLT);
