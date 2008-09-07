#include "DQMOffline/Trigger/interface/HLTTauPostProcessor.h"
#include <math.h>
HLTTauPostProcessor::HLTTauPostProcessor(const edm::ParameterSet& iConfig):
  L1Folder_(iConfig.getParameter<std::vector<std::string> >("L1Folder")),
  L2Folder_(iConfig.getParameter<std::vector<std::string> >("L2Folder")),
  L25Folder_(iConfig.getParameter<std::vector<std::string> >("L25Folder")),
  L3Folder_(iConfig.getParameter<std::vector< std::string> >("L3Folder")),
  pathValFolder_(iConfig.getParameter<std::vector< std::string> >("HLTPathValidationFolder")),
  pathDQMFolder_(iConfig.getParameter<std::vector<std::string> >("HLTPathDQMFolder"))
  
{
  
}

HLTTauPostProcessor::~HLTTauPostProcessor() {}

void HLTTauPostProcessor::beginJob(const edm::EventSetup& iSetup) 
{
  return;
}

void HLTTauPostProcessor::endJob()
{

  dbe = 0;
  dbe = edm::Service<DQMStore>().operator->();

  if (dbe) {

    //Path Validation
    for(size_t i=0;i<pathValFolder_.size();++i)
    if(pathValFolder_[i].size()>0)
      {
	createEfficiencyHisto(pathValFolder_[i],"L1EffVsEtRef","l1eteff","refEt",dbe);
	createEfficiencyHisto(pathValFolder_[i],"L1EffVsEtaRef","l1etaeff","refEta",dbe);

	createEfficiencyHisto(pathValFolder_[i],"L2EffVsEtL1","l2eteff","l1eteff",dbe);
	createEfficiencyHisto(pathValFolder_[i],"L2EffVsEtaL1","l2etaeff","l1etaeff",dbe);

	createEfficiencyHisto(pathValFolder_[i],"L25EffVsEtL1","l25eteff","l1eteff",dbe);
	createEfficiencyHisto(pathValFolder_[i],"L25EffVsEtaL1","l25etaeff","l1etaeff",dbe);

	createEfficiencyHisto(pathValFolder_[i],"L25EffVsEtL2","l25eteff","l2eteff",dbe);
	createEfficiencyHisto(pathValFolder_[i],"L25EffVsEtaL2","l25etaeff","l2etaeff",dbe);

	createEfficiencyHisto(pathValFolder_[i],"L3EffVsEtL1","l3eteff","l1eteff",dbe);
	createEfficiencyHisto(pathValFolder_[i],"L3EffVsEtaL1","l3etaeff","l1etaeff",dbe);

	createEfficiencyHisto(pathValFolder_[i],"L3EffVsEtL25","l3eteff","l25eteff",dbe);
	createEfficiencyHisto(pathValFolder_[i],"L3EffVsEtaL25","l3etaeff","l25etaeff",dbe);

	calculatePathEfficiencies(pathValFolder_[i],"acceptedEventsMatched",dbe);


      }


    //L1 Harvesting
    for(size_t i=0;i<L1Folder_.size();++i)
    if(L1Folder_[i].size()>0)
      {
	createEfficiencyHisto(L1Folder_[i],"L1TauEtEff","L1RefMatchedTauEt","RefTauHadEt",dbe);
	createEfficiencyHisto(L1Folder_[i],"L1TauPhiEff","L1RefMatchedTauPhi","RefTauHadPhi",dbe);
	createEfficiencyHisto(L1Folder_[i],"L1TauEtaEff","L1RefMatchedTauEta","RefTauHadEta",dbe);

	createEfficiencyHisto(L1Folder_[i],"L1MuonEtEff","L1RefMatchedTauMuonEt","RefTauMuonEt",dbe);
	createEfficiencyHisto(L1Folder_[i],"L1MuonPhiEff","L1RefMatchedTauMuonPhi","RefTauMuonPhi",dbe);
	createEfficiencyHisto(L1Folder_[i],"L1MuonEtaEff","L1RefMatchedTauMuonEta","RefTauMuonEta",dbe);

	createEfficiencyHisto(L1Folder_[i],"L1ElecEtEff","L1RefMatchedTauElecEt","RefTauElecEt",dbe);
	createEfficiencyHisto(L1Folder_[i],"L1ElecPhiEff","L1RefMatchedTauElecPhi","RefTauElecPhi",dbe);
	createEfficiencyHisto(L1Folder_[i],"L1ElecEtaEff","L1RefMatchedTauElecEta","RefTauElecEta",dbe);

	createIntegratedHisto(L1Folder_[i],"L1SingleTauEffEt","nfidCounter",1,dbe);
	createIntegratedHisto(L1Folder_[i],"L1SingleTauEffRefMatchEt","nfidCounter",2,dbe);

	createIntegratedHisto(L1Folder_[i],"L1TauMETfixEffEt","nfidCounter",1,dbe);
	createIntegratedHisto(L1Folder_[i],"L1TauMETfixEffRefMatchEt","nfidCounter",2,dbe);

	createIntegratedHisto(L1Folder_[i],"L1DoubleTauEffEt","nfidCounter",1,dbe);
	createIntegratedHisto(L1Folder_[i],"L1DoubleTauEffRefMatchEt","nfidCounter",3,dbe);

	createIntegratedHisto(L1Folder_[i],"L1TauIsoEgfixEffEt","nfidCounter",1,dbe);
	createIntegratedHisto(L1Folder_[i],"L1TauIsoEgfixEffRefMatchEt","nfidCounter",5,dbe);

	createIntegratedHisto(L1Folder_[i],"L1TauMuonfixEffEt","nfidCounter",1,dbe);
	createIntegratedHisto(L1Folder_[i],"L1TauMuonfixEffRefMatchEt","nfidCounter",4,dbe);


      }

    //L2 Harvesting
    for(size_t i=0;i<L2Folder_.size();++i)
    if(L2Folder_[i].size()>0)
      {
	createEfficiencyHisto(L2Folder_[i],"L2EtEff","L2EtEffNum","L2EtEffDenom",dbe);

      }

    //L25 Harvesting
    for(size_t i=0;i<L25Folder_.size();++i)
    if(L25Folder_[i].size()>0)
      {
	createEfficiencyHisto(L25Folder_[i],"L25EtEff","L25IsoJetEt","L25jetEt",dbe);
	createEfficiencyHisto(L25Folder_[i],"L25EtaEff","L25IsoJetEta","L25jetEta",dbe);
      }

    //L3 Harvesting
    for(size_t i=0;i<L3Folder_.size();++i)
    if(L3Folder_[i].size()>0)
      {
	createEfficiencyHisto(L3Folder_[i],"L3EtEff","L3IsoJetEt","L3jetEt",dbe);
	createEfficiencyHisto(L3Folder_[i],"L3EtaEff","L3IsoJetEta","L3jetEta",dbe);
      }

  }


}      


void HLTTauPostProcessor::beginRun(const edm::Run& iRun, 
				  const edm::EventSetup& iSetup)
{
  return;
}

void HLTTauPostProcessor::endRun(const edm::Run& iRun, 
				const edm::EventSetup& iSetup)
{
  return;
}

void HLTTauPostProcessor::analyze(const edm::Event& iEvent, 
				 const edm::EventSetup& iSetup)
{
  return;
}

void 
HLTTauPostProcessor::createEfficiencyHisto(std::string folder,std::string name,std::string hist1,std::string hist2,DQMStore* dbe)
{
	MonitorElement * effnum = dbe->get(folder+"/"+hist1);
	MonitorElement * effdenom = dbe->get(folder+"/"+hist2);

	if(effnum && effdenom)
	  {
	    dbe->setCurrentFolder(folder);
	    MonitorElement* Eff =  dbe->book1D(name,name,effnum->getTH1F()->GetNbinsX(),effnum->getTH1F()->GetXaxis()->GetXmin(),effnum->getTH1F()->GetXaxis()->GetXmax());
	    Eff->getTH1F()->Divide(effnum->getTH1F(),effdenom->getTH1F(),1.,1.,"B");
	  }

}




void
HLTTauPostProcessor::createIntegratedHisto(std::string folder,std::string histo,std::string nfidh,int bin,DQMStore* dbe)
{
  MonitorElement* eff = dbe->get(folder+"/"+histo);
  MonitorElement* nfid = dbe->get(folder+"/"+nfidh);

  double nGenerated = nfid->getBinContent(bin);

  int nbins = eff->getTH1F()->GetNbinsX();
  double integral = eff->getTH1F()->GetBinContent(nbins+1);  // Initialize to overflow
  if (nGenerated<=0) {
    return;
  }
  for(int i = nbins; i >= 1; i--)
    {
      double thisBin = eff->getBinContent(i);
      integral += thisBin;
      double integralEff;
      double integralError;
      integralEff = (integral / nGenerated);
      eff->setBinContent(i, integralEff);
      integralError = (sqrt(integral) / nGenerated);
      eff->setBinError(i, integralError);
    }
}

void 
HLTTauPostProcessor::calculatePathEfficiencies(std::string folder,std::string histo,DQMStore*dbe)
{
 MonitorElement* eff = dbe->get(folder+"/"+histo);

 //Calculate Efficiencies with ref to truth
 MonitorElement * effRefTruth = dbe->book1D("EffRefTruth","Efficiency with ref to truth",5,0,5);

 effRefTruth->setBinContent(1,calcEfficiency(eff->getBinContent(1),eff->getBinContent(6))[0]);
 effRefTruth->setBinContent(2,calcEfficiency(eff->getBinContent(2),eff->getBinContent(6))[0]);
 effRefTruth->setBinContent(3,calcEfficiency(eff->getBinContent(3),eff->getBinContent(6))[0]);
 effRefTruth->setBinContent(4,calcEfficiency(eff->getBinContent(4),eff->getBinContent(6))[0]);
 effRefTruth->setBinContent(5,calcEfficiency(eff->getBinContent(5),eff->getBinContent(6))[0]);

 effRefTruth->setBinError(1,calcEfficiency(eff->getBinContent(1),eff->getBinContent(6))[1]);
 effRefTruth->setBinError(2,calcEfficiency(eff->getBinContent(2),eff->getBinContent(6))[1]);
 effRefTruth->setBinError(3,calcEfficiency(eff->getBinContent(3),eff->getBinContent(6))[1]);
 effRefTruth->setBinError(4,calcEfficiency(eff->getBinContent(4),eff->getBinContent(6))[1]);
 effRefTruth->setBinError(5,calcEfficiency(eff->getBinContent(5),eff->getBinContent(6))[1]);

 //calculate Efficiency with ref to L1
 MonitorElement * effRefL1 = dbe->book1D("EffRefL1","Efficiency with ref to L1",5,0,5);

 effRefL1->setBinContent(1,calcEfficiency(eff->getBinContent(1),eff->getBinContent(2))[0]);
 effRefL1->setBinContent(2,calcEfficiency(eff->getBinContent(2),eff->getBinContent(2))[0]);
 effRefL1->setBinContent(3,calcEfficiency(eff->getBinContent(3),eff->getBinContent(2))[0]);
 effRefL1->setBinContent(4,calcEfficiency(eff->getBinContent(4),eff->getBinContent(2))[0]);
 effRefL1->setBinContent(5,calcEfficiency(eff->getBinContent(5),eff->getBinContent(2))[0]);

 effRefL1->setBinError(1,calcEfficiency(eff->getBinContent(1),eff->getBinContent(2))[1]);
 effRefL1->setBinError(2,calcEfficiency(eff->getBinContent(2),eff->getBinContent(2))[1]);
 effRefL1->setBinError(3,calcEfficiency(eff->getBinContent(3),eff->getBinContent(2))[1]);
 effRefL1->setBinError(4,calcEfficiency(eff->getBinContent(4),eff->getBinContent(2))[1]);
 effRefL1->setBinError(5,calcEfficiency(eff->getBinContent(5),eff->getBinContent(2))[1]);

 //calculate Efficiency with ref to Previous
 MonitorElement * effRefPrevious = dbe->book1D("EffRefPrevious","Efficiency with ref to Previous",5,0,5);

 effRefPrevious->setBinContent(1,calcEfficiency(eff->getBinContent(1),eff->getBinContent(2))[0]);
 effRefPrevious->setBinContent(2,calcEfficiency(eff->getBinContent(2),eff->getBinContent(6))[0]);
 effRefPrevious->setBinContent(3,calcEfficiency(eff->getBinContent(3),eff->getBinContent(2))[0]);
 effRefPrevious->setBinContent(4,calcEfficiency(eff->getBinContent(4),eff->getBinContent(3))[0]);
 effRefPrevious->setBinContent(5,calcEfficiency(eff->getBinContent(5),eff->getBinContent(4))[0]);

 effRefPrevious->setBinError(1,calcEfficiency(eff->getBinContent(1),eff->getBinContent(2))[1]);
 effRefPrevious->setBinError(2,calcEfficiency(eff->getBinContent(2),eff->getBinContent(6))[1]);
 effRefPrevious->setBinError(3,calcEfficiency(eff->getBinContent(3),eff->getBinContent(2))[1]);
 effRefPrevious->setBinError(4,calcEfficiency(eff->getBinContent(4),eff->getBinContent(3))[1]);
 effRefPrevious->setBinError(5,calcEfficiency(eff->getBinContent(5),eff->getBinContent(4))[1]);

}


std::vector<double>
HLTTauPostProcessor::calcEfficiency(float num,float denom)
{
  std::vector<double> a;
  if(denom==0)
    {
      a.push_back(0.);
      a.push_back(0.);
    }
  else
    {    
      a.push_back(num/denom);
      a.push_back(sqrt(a[0]*(1-a[0])/(denom)));
    }
  return a;
}
