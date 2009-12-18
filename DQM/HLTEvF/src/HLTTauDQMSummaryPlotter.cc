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
	bookEfficiencyHisto(L1Folder_[i],"L1TauEtEff","EfficiencyHelpers/L1TauEtEffNum",dbe);
	bookEfficiencyHisto(L1Folder_[i],"L1TauEtaEff","EfficiencyHelpers/L1TauEtaEffNum",dbe);
	bookEfficiencyHisto(L1Folder_[i],"L1TauPhiEff","EfficiencyHelpers/L1TauPhiEffNum",dbe);

	bookEfficiencyHisto(L1Folder_[i],"L1JetEtEff","EfficiencyHelpers/L1JetEtEffNum",dbe);
	bookEfficiencyHisto(L1Folder_[i],"L1JetEtaEff","EfficiencyHelpers/L1JetEtaEffNum",dbe);
	bookEfficiencyHisto(L1Folder_[i],"L1JetPhiEff","EfficiencyHelpers/L1JetPhiEffNum",dbe);

	bookEfficiencyHisto(L1Folder_[i],"L1SingleTauEff","L1LeadTauEt",dbe);
	bookEfficiencyHisto(L1Folder_[i],"L1DoubleTauEff","L1SecondTauEt",dbe);

      }


    //L2 Summary
    for(size_t i=0;i<caloFolder_.size();++i)
    if(caloFolder_[i].size()>0)
      {
	bookEfficiencyHisto(caloFolder_[i],"L2RecoTauEtEff","EfficiencyHelpers/L2RecoTauEtEffNum",dbe);
	bookEfficiencyHisto(caloFolder_[i],"L2RecoTauEtaEff","EfficiencyHelpers/L2RecoTauEtaEffNum",dbe);
	bookEfficiencyHisto(caloFolder_[i],"L2RecoTauPhiEff","EfficiencyHelpers/L2RecoTauPhiEffNum",dbe);

	bookEfficiencyHisto(caloFolder_[i],"L2IsoTauEtEff","EfficiencyHelpers/L2IsoTauEtEffNum",dbe);
	bookEfficiencyHisto(caloFolder_[i],"L2IsoTauEtaEff","EfficiencyHelpers/L2IsoTauEtaEffNum",dbe);
	bookEfficiencyHisto(caloFolder_[i],"L2IsoTauPhiEff","EfficiencyHelpers/L2IsoTauPhiEffNum",dbe);
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
	plotEfficiencyHisto(L1Folder_[i],"L1TauEtEff","EfficiencyHelpers/L1TauEtEffNum","EfficiencyHelpers/L1TauEtEffDenom",dbe);
	plotEfficiencyHisto(L1Folder_[i],"L1TauEtaEff","EfficiencyHelpers/L1TauEtaEffNum","EfficiencyHelpers/L1TauEtaEffDenom",dbe);
	plotEfficiencyHisto(L1Folder_[i],"L1TauPhiEff","EfficiencyHelpers/L1TauPhiEffNum","EfficiencyHelpers/L1TauPhiEffDenom",dbe);

	plotEfficiencyHisto(L1Folder_[i],"L1JetEtEff","EfficiencyHelpers/L1JetEtEffNum","EfficiencyHelpers/L1JetEtEffDenom",dbe);
	plotEfficiencyHisto(L1Folder_[i],"L1JetEtaEff","EfficiencyHelpers/L1JetEtaEffNum","EfficiencyHelpers/L1JetEtaEffDenom",dbe);
	plotEfficiencyHisto(L1Folder_[i],"L1JetPhiEff","EfficiencyHelpers/L1JetPhiEffNum","EfficiencyHelpers/L1JetPhiEffDenom",dbe);

	plotEfficiencyHisto(L1Folder_[i],"L1ElectronEtEff","EfficiencyHelpers/L1ElectronEtEffNum","EfficiencyHelpers/L1ElectronEtEffDenom",dbe);
	plotEfficiencyHisto(L1Folder_[i],"L1ElectronEtaEff","EfficiencyHelpers/L1ElectronEtaEffNum","EfficiencyHelpers/L1ElectronEtaEffDenom",dbe);
	plotEfficiencyHisto(L1Folder_[i],"L1ElectronPhiEff","EfficiencyHelpers/L1ElectronPhiEffNum","EfficiencyHelpers/L1ElectronPhiEffDenom",dbe);

	plotEfficiencyHisto(L1Folder_[i],"L1MuonEtEff","EfficiencyHelpers/L1MuonEtEffNum","EfficiencyHelpers/L1MuonEtEffDenom",dbe);
	plotEfficiencyHisto(L1Folder_[i],"L1MuonEtaEff","EfficiencyHelpers/L1MuonEtaEffNum","EfficiencyHelpers/L1MuonEtaEffDenom",dbe);
	plotEfficiencyHisto(L1Folder_[i],"L1MuonPhiEff","EfficiencyHelpers/L1MuonPhiEffNum","EfficiencyHelpers/L1MuonPhiEffDenom",dbe);

	plotIntegratedEffHisto(L1Folder_[i],"L1SingleTauEff","L1LeadTauEt","InputEvents",1,dbe);
	plotIntegratedEffHisto(L1Folder_[i],"L1DoubleTauEff","L1SecondTauEt","InputEvents",2,dbe);
      }

    //L2 Summary
    for(size_t i=0;i<caloFolder_.size();++i)
    if(caloFolder_[i].size()>0)
      {
	plotEfficiencyHisto(caloFolder_[i],"L2RecoTauEtEff","EfficiencyHelpers/L2RecoTauEtEffNum","EfficiencyHelpers/L2RecoTauEtEffDenom",dbe);
	plotEfficiencyHisto(caloFolder_[i],"L2RecoTauEtaEff","EfficiencyHelpers/L2RecoTauEtaEffNum","EfficiencyHelpers/L2RecoTauEtaEffDenom",dbe);
	plotEfficiencyHisto(caloFolder_[i],"L2RecoTauPhiEff","EfficiencyHelpers/L2RecoTauPhiEffNum","EfficiencyHelpers/L2RecoTauPhiEffDenom",dbe);

	plotEfficiencyHisto(caloFolder_[i],"L2IsoTauEtEff","EfficiencyHelpers/L2IsoTauEtEffNum","EfficiencyHelpers/L2IsoTauEtEffDenom",dbe);
	plotEfficiencyHisto(caloFolder_[i],"L2IsoTauEtaEff","EfficiencyHelpers/L2IsoTauEtaEffNum","EfficiencyHelpers/L2IsoTauEtaEffDenom",dbe);
	plotEfficiencyHisto(caloFolder_[i],"L2IsoTauPhiEff","EfficiencyHelpers/L2IsoTauPhiEffNum","EfficiencyHelpers/L2IsoTauPhiEffDenom",dbe);
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
	MonitorElement *tmp = dbe->book1D(name,name,effnum->getTH1F()->GetNbinsX(),effnum->getTH1F()->GetXaxis()->GetXmin(),effnum->getTH1F()->GetXaxis()->GetXmax());
	tmp->getTH1F()->GetYaxis()->SetRangeUser(0.,1.05);
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
HLTTauDQMSummaryPlotter::plotIntegratedEffHisto(std::string folder,std::string name,std::string refHisto,std::string evCount,int bin,DQMStore * dbe)
{
  if(dbe->dirExists(folder))
    {
      MonitorElement * refH = dbe->get(folder+"/"+refHisto);
      MonitorElement * evC  = dbe->get(folder+"/"+evCount);
      MonitorElement * eff = dbe->get(folder+"/"+name);

      if(refH && evC && eff)
	{
	  TH1F *histo = refH->getTH1F();
	  float nGenerated = evC->getTH1F()->GetBinContent(bin);
	  // Assuming that the histogram is incremented with weight=1 for each event
	  // this function integrates the histogram contents above every bin and stores it
	  // in that bin.  The result is plot of integral rate versus threshold plot.
	  int nbins = histo->GetNbinsX();
	  double integral = histo->GetBinContent(nbins+1);  // Initialize to overflow
	  if (nGenerated<=0.0)  {
	    nGenerated=1.0;
	  }
	  for(int i = nbins; i >= 1; i--)
	    {
	      double thisBin = histo->GetBinContent(i);
	      integral += thisBin;
	      double integralEff;
	      double integralError;
	      integralEff = (integral / nGenerated);
	      eff->getTH1F()->SetBinContent(i, integralEff);
	      // error
	      integralError = (sqrt(integral) / nGenerated);
	      eff->getTH1F()->SetBinError(i, integralError);
	    }
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
