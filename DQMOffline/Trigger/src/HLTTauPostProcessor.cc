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
	calculatePathEfficiencies(pathValFolder_[i],"MatchedTriggers",dbe);
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
  if(dbe->dirExists(folder))
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
}




void
HLTTauPostProcessor::createIntegratedHisto(std::string folder,std::string histo,std::string nfidh,int bin,DQMStore* dbe)
{
  if(dbe->dirExists(folder))
  {

    MonitorElement* eff = dbe->get(folder+"/"+histo);
    MonitorElement* nfid = dbe->get(folder+"/"+nfidh);
  
    if(eff && nfid)
      { 
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
  }
}

void 
HLTTauPostProcessor::calculatePathEfficiencies(std::string folder,std::string histo,DQMStore* dbe)
{
  if(dbe->dirExists(folder))
  {
    dbe->setCurrentFolder(folder);
    MonitorElement* eff = dbe->get(folder+"/"+histo);
   
    if(eff)
      {
	//Calculate Efficiencies with ref to truth
	MonitorElement * effRefTruth = dbe->book1D("PathEffMatchedRef","Efficiency with Matching",eff->getNbinsX()-1,0,eff->getNbinsX()-1);
	for(int i =2;i<=eff->getNbinsX();++i)
	  {
	    effRefTruth->setBinContent(i-1,calcEfficiency(eff->getBinContent(i),eff->getBinContent(1))[0]);
	    effRefTruth->setBinError(i-1,calcEfficiency(eff->getBinContent(i),eff->getBinContent(1))[1]);
	    effRefTruth->setBinLabel(i-1,eff->getTH1F()->GetXaxis()->GetBinLabel(i));

	  }


	//Calculate Efficiencies with ref to L1
	MonitorElement * effRefL1 = dbe->book1D("PathEffMatchedRefL1","Efficiency with Matching Ref to L1",eff->getNbinsX()-2,0,eff->getNbinsX()-2);
	for(int i =3;i<=eff->getNbinsX();++i)
	  {
	    effRefL1->setBinContent(i-2,calcEfficiency(eff->getBinContent(i),eff->getBinContent(2))[0]);
	    effRefL1->setBinError(i-2,calcEfficiency(eff->getBinContent(i),eff->getBinContent(2))[1]);
	    effRefL1->setBinLabel(i-2,eff->getTH1F()->GetXaxis()->GetBinLabel(i));
	  }

	//Calculate Efficiencies with ref to previous
	MonitorElement * effRefPrevious = dbe->book1D("PathEffMatchedRefPrevious","Efficiency with Matching Ref To previous",eff->getNbinsX()-1,0,eff->getNbinsX()-1);
	for(int i = 2;i<=eff->getNbinsX();++i)
	  {
	    effRefPrevious->setBinContent(i-1,calcEfficiency(eff->getBinContent(i),eff->getBinContent(i-1))[0]);
	    effRefPrevious->setBinError(i-1,calcEfficiency(eff->getBinContent(i),eff->getBinContent(i-1))[1]);
	    effRefPrevious->setBinLabel(i-1,eff->getTH1F()->GetXaxis()->GetBinLabel(i));
	  }
      }
  }
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
