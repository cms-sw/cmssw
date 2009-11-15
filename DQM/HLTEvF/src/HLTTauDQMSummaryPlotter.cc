#include "DQM/HLTEvF/interface/HLTTauDQMSummaryPlotter.h"
#include <math.h>
HLTTauDQMSummaryPlotter::HLTTauDQMSummaryPlotter(const edm::ParameterSet& iConfig):
  L1Folder_(iConfig.getParameter<std::vector<std::string> >("L1Dirs")),
  caloFolder_(iConfig.getParameter<std::vector<std::string> >("caloDirs")),
  trackFolder_(iConfig.getParameter<std::vector<std::string> >("trackDirs")),
  pathFolder_(iConfig.getParameter<std::vector< std::string> >("pathDirs")),
  litePathFolder_(iConfig.getParameter<std::vector< std::string> >("pathSummaryDirs"))

{
  dbe = 0;
  dbe = edm::Service<DQMStore>().operator->();

  if (dbe) {

    //Path Summary 
    for(size_t i=0;i<pathFolder_.size();++i)
    if(pathFolder_[i].size()>0)
      {
	bookTriggerBitEfficiencyHistos(pathFolder_[i],"MatchedTriggerBits",dbe);
      }

    //Lite Path Summary 
    for(size_t i=0;i<litePathFolder_.size();++i)
    if(litePathFolder_[i].size()>0)
      {
	bookEfficiencyHisto(litePathFolder_[i],"PathEfficiency","MatchedPathTriggerBits",dbe);
	bookEfficiencyHisto(litePathFolder_[i],"TrigTauEtEff","TrigTauEtEffNum",dbe);
	bookEfficiencyHisto(litePathFolder_[i],"TrigTauEtaEff","TrigTauEtaEffNum",dbe);
	bookEfficiencyHisto(litePathFolder_[i],"TrigTauPhiEff","TrigTauPhiEffNum",dbe);
      }

    //L1 Summary
    for(size_t i=0;i<L1Folder_.size();++i)
    if(L1Folder_[i].size()>0)
      {
	bookEfficiencyHisto(L1Folder_[i],"L1TauEtEff","L1TauEtEffNum",dbe);
	bookEfficiencyHisto(L1Folder_[i],"L1TauEtaEff","L1TauEtaEffNum",dbe);
	bookEfficiencyHisto(L1Folder_[i],"L1TauPhiEff","L1TauPhiEffNum",dbe);

	bookEfficiencyHisto(L1Folder_[i],"L1JetEtEff","L1JetEtEffNum",dbe);
	bookEfficiencyHisto(L1Folder_[i],"L1JetEtaEff","L1JetEtaEffNum",dbe);
	bookEfficiencyHisto(L1Folder_[i],"L1JetPhiEff","L1JetPhiEffNum",dbe);

	bookEfficiencyHisto(L1Folder_[i],"L1LeptonEtEff","L1LeptonEtEffNum",dbe);
	bookEfficiencyHisto(L1Folder_[i],"L1LeptonEtaEff","L1LeptonEtaEffNum",dbe);
	bookEfficiencyHisto(L1Folder_[i],"L1LeptonPhiEff","L1LeptonPhiEffNum",dbe);

      }


    //L2 Summary
    for(size_t i=0;i<caloFolder_.size();++i)
    if(caloFolder_[i].size()>0)
      {
	bookEfficiencyHisto(caloFolder_[i],"L2TauEtEff","L2TauEtEffNum",dbe);
	bookEfficiencyHisto(caloFolder_[i],"L2TauEtaEff","L2TauEtaEffNum",dbe);
	bookEfficiencyHisto(caloFolder_[i],"L2TauPhiEff","L2TauPhiEffNum",dbe);
      }

    //L25/3 Summary
    for(size_t i=0;i<trackFolder_.size();++i)
    if(trackFolder_[i].size()>0)
      {
	bookEfficiencyHisto(trackFolder_[i],"L25TauEtEff","L25TauEtEffNum",dbe);
	bookEfficiencyHisto(trackFolder_[i],"L25TauEtaEff","L25TauEtaEffNum",dbe);
	bookEfficiencyHisto(trackFolder_[i],"L25TauPhiEff","L25TauPhiEffNum",dbe);
	bookEfficiencyHisto(trackFolder_[i],"L3TauEtEff","L3TauEtEffNum",dbe);
	bookEfficiencyHisto(trackFolder_[i],"L3TauEtaEff","L3TauEtaEffNum",dbe);
	bookEfficiencyHisto(trackFolder_[i],"L3TauPhiEff","L3TauPhiEffNum",dbe);
      }
  }

  
}

HLTTauDQMSummaryPlotter::~HLTTauDQMSummaryPlotter() {}


void 
HLTTauDQMSummaryPlotter::plot()
{
  if (dbe) {

    //Path Summary
    for(size_t i=0;i<pathFolder_.size();++i)
    if(pathFolder_[i].size()>0)
      {
	plotTriggerBitEfficiencyHistos(pathFolder_[i],"MatchedTriggerBits",dbe);
      }


    //Lite Path Summary 
    for(size_t i=0;i<litePathFolder_.size();++i)
    if(litePathFolder_[i].size()>0)
      {
	plotEfficiencyHisto(litePathFolder_[i],"PathEfficiency","MatchedPathTriggerBits","RefEvents",dbe);
	plotEfficiencyHisto(litePathFolder_[i],"TrigTauEtEff","TrigTauEtEffNum","TrigTauEtEffDenom",dbe);
	plotEfficiencyHisto(litePathFolder_[i],"TrigTauEtaEff","TrigTauEtaEffNum","TrigTauEtaEffDenom",dbe);
	plotEfficiencyHisto(litePathFolder_[i],"TrigTauPhiEff","TrigTauPhiEffNum","TrigTauPhiEffDenom",dbe);
      }

    //L1 Summary
    for(size_t i=0;i<L1Folder_.size();++i)
    if(L1Folder_[i].size()>0)
      {
	plotEfficiencyHisto(L1Folder_[i],"L1TauEtEff","L1TauEtEffNum","L1TauEtEffDenom",dbe);
	plotEfficiencyHisto(L1Folder_[i],"L1TauEtaEff","L1TauEtaEffNum","L1TauEtaEffDenom",dbe);
	plotEfficiencyHisto(L1Folder_[i],"L1TauPhiEff","L1TauPhiEffNum","L1TauPhiEffDenom",dbe);

	plotEfficiencyHisto(L1Folder_[i],"L1JetEtEff","L1JetEtEffNum","L1JetEtEffDenom",dbe);
	plotEfficiencyHisto(L1Folder_[i],"L1JetEtaEff","L1JetEtaEffNum","L1JetEtaEffDenom",dbe);
	plotEfficiencyHisto(L1Folder_[i],"L1JetPhiEff","L1JetPhiEffNum","L1JetPhiEffDenom",dbe);

	plotEfficiencyHisto(L1Folder_[i],"L1LeptonEtEff","L1LeptonEtEffNum","L1LeptonEtEffDenom",dbe);
	plotEfficiencyHisto(L1Folder_[i],"L1LeptonEtaEff","L1LeptonEtaEffNum","L1LeptonEtaEffDenom",dbe);
	plotEfficiencyHisto(L1Folder_[i],"L1LeptonPhiEff","L1LeptonPhiEffNum","L1LeptonPhiEffDenom",dbe);

      }

    //L2 Summary
    for(size_t i=0;i<caloFolder_.size();++i)
    if(caloFolder_[i].size()>0)
      {
	plotEfficiencyHisto(caloFolder_[i],"L2TauEtEff","L2TauEtEffNum","L2TauEtEffDenom",dbe);
	plotEfficiencyHisto(caloFolder_[i],"L2TauEtaEff","L2TauEtaEffNum","L2TauEtaEffDenom",dbe);
	plotEfficiencyHisto(caloFolder_[i],"L2TauPhiEff","L2TauPhiEffNum","L2TauPhiEffDenom",dbe);
      }


    //L25/3 Summary
    for(size_t i=0;i<trackFolder_.size();++i)
    if(trackFolder_[i].size()>0)
      {
	plotEfficiencyHisto(trackFolder_[i],"L25TauEtEff","L25TauEtEffNum","L25TauEtEffDenom",dbe);
	plotEfficiencyHisto(trackFolder_[i],"L25TauEtaEff","L25TauEtaEffNum","L25TauEtaEffDenom",dbe);
	plotEfficiencyHisto(trackFolder_[i],"L25TauPhiEff","L25TauPhiEffNum","L25TauPhiEffDenom",dbe);
	plotEfficiencyHisto(trackFolder_[i],"L3TauEtEff","L3TauEtEffNum","L3TauEtEffDenom",dbe);
	plotEfficiencyHisto(trackFolder_[i],"L3TauEtaEff","L3TauEtaEffNum","L3TauEtaEffDenom",dbe);
	plotEfficiencyHisto(trackFolder_[i],"L3TauPhiEff","L3TauPhiEffNum","L3TauPhiEffDenom",dbe);

      }
  }
}      



void 
HLTTauDQMSummaryPlotter::bookEfficiencyHisto(std::string folder,std::string name,std::string hist1,DQMStore* dbe)
{
  if(dbe->dirExists(folder))
  {
    MonitorElement * effnum = dbe->get(folder+"/"+hist1);

    if(effnum)
      {
	dbe->setCurrentFolder(folder);
	dbe->book1D(name,name,effnum->getTH1F()->GetNbinsX(),effnum->getTH1F()->GetXaxis()->GetXmin(),effnum->getTH1F()->GetXaxis()->GetXmax());
      }
  }
}


void 
HLTTauDQMSummaryPlotter::plotEfficiencyHisto(std::string folder,std::string name,std::string hist1,std::string hist2,DQMStore* dbe)
{
  if(dbe->dirExists(folder))
  {
    MonitorElement * effnum = dbe->get(folder+"/"+hist1);
    MonitorElement * effdenom = dbe->get(folder+"/"+hist2);
    MonitorElement * eff = dbe->get(folder+"/"+name);
    
    if(effnum && effdenom && eff )
      {
	//	dbe->setCurrentFolder(folder);
	  eff->getTH1F()->Divide(effnum->getTH1F(),effdenom->getTH1F(),1.,1.,"B");
      }

  }
}


void 
HLTTauDQMSummaryPlotter::bookTriggerBitEfficiencyHistos(std::string folder,std::string histo,DQMStore* dbe)
{
  if(dbe->dirExists(folder))
  {
    dbe->setCurrentFolder(folder);
    MonitorElement* eff = dbe->get(folder+"/"+histo);
    dbe->book1D("EfficiencyRefInput","Efficiency with Matching",eff->getNbinsX()-1,0,eff->getNbinsX()-1);
    dbe->book1D("EfficiencyRefL1","Efficiency with Matching Ref to L1",eff->getNbinsX()-2,0,eff->getNbinsX()-2);
    dbe->book1D("EfficiencyRefPrevious","Efficiency with Matching Ref To previous",eff->getNbinsX()-1,0,eff->getNbinsX()-1);

  }

}


void 
HLTTauDQMSummaryPlotter::plotTriggerBitEfficiencyHistos(std::string folder,std::string histo,DQMStore* dbe)
{
  if(dbe->dirExists(folder))
  {
    dbe->setCurrentFolder(folder);
    MonitorElement* eff = dbe->get(folder+"/"+histo);
    MonitorElement * effRefTruth = dbe->get(folder+"/EfficiencyRefInput");
    MonitorElement * effRefL1 = dbe->get(folder+"/EfficiencyRefL1");
    MonitorElement * effRefPrevious = dbe->get(folder+"/EfficiencyRefPrevious");
   

    if(eff)
      {
	//Calculate Efficiencies with ref to Matched Objects
	for(int i =2;i<=eff->getNbinsX();++i)
	  {
	    effRefTruth->setBinContent(i-1,calcEfficiency(eff->getBinContent(i),eff->getBinContent(1))[0]);
	    effRefTruth->setBinError(i-1,calcEfficiency(eff->getBinContent(i),eff->getBinContent(1))[1]);
	    effRefTruth->setBinLabel(i-1,eff->getTH1F()->GetXaxis()->GetBinLabel(i));

	  }


	//Calculate Efficiencies with ref to L1
	for(int i =3;i<=eff->getNbinsX();++i)
	  {
	    effRefL1->setBinContent(i-2,calcEfficiency(eff->getBinContent(i),eff->getBinContent(2))[0]);
	    effRefL1->setBinError(i-2,calcEfficiency(eff->getBinContent(i),eff->getBinContent(2))[1]);
	    effRefL1->setBinLabel(i-2,eff->getTH1F()->GetXaxis()->GetBinLabel(i));
	  }

	//Calculate Efficiencies with ref to previous
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
HLTTauDQMSummaryPlotter::calcEfficiency(float num,float denom)
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
