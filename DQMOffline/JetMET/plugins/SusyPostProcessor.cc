#include <iostream>

#include "DQMOffline/JetMET/plugins/SusyPostProcessor.h"
#include "DQMOffline/JetMET/interface/SusyDQM/Quantile.h"
#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

using namespace std;

const char* SusyPostProcessor::messageLoggerCatregory = "SusyDQMPostProcessor";

SusyPostProcessor::SusyPostProcessor(const edm::ParameterSet& pSet)
{
  
  dqm = edm::Service<DQMStore>().operator->();
  iConfig = pSet;

  SUSYFolder = iConfig.getParameter<string>("folderName");
  _quantile = iConfig.getParameter<double>("quantile");

}

SusyPostProcessor::~SusyPostProcessor(){
}

void SusyPostProcessor::beginJob(void){
}

void SusyPostProcessor::beginRun(const edm::Run&, const edm::EventSetup& iSetup){
}

void SusyPostProcessor::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup){
}

void  SusyPostProcessor::QuantilePlots(MonitorElement* ME, double q_value)
{
  if(ME->getTH1()->GetEntries()>0.)
    {
      Quantile q(static_cast<const TH1*>(ME->getTH1()));
      Float_t mean=q[q_value].first;
      Float_t RMS=q[q_value].second;
      
      Float_t xLow=-5.5;
      Float_t xUp=9.5;
      Int_t NBin=15;
      
      if(mean>0.)
	{
	  Float_t DBin=RMS*TMath::Sqrt(12.)/2.;
	  xLow=mean-int(mean/DBin+2)*DBin;
	  xUp=int(0.2*mean/DBin)*DBin+mean+5*DBin;
	  NBin=(xUp-xLow)/DBin;
	}

      dqm->setCurrentFolder(ME->getPathname());
      TString name=ME->getTH1()->GetName();
      name+="_quant";
      ME=dqm->book1D(name,"",NBin, xLow, xUp);
      ME->Fill(mean-RMS);
      ME->Fill(mean+RMS);
    }
}



void SusyPostProcessor::endRun(const edm::Run&, const edm::EventSetup&)
{
  // MET
  //----------------------------------------------------------------------------
  dqm->setCurrentFolder("JetMET/MET");

  Dirs = dqm->getSubdirs();

  std::vector<std::string> metFolders;

  metFolders.push_back("All/");
  metFolders.push_back("BasicCleanup/");
  metFolders.push_back("ExtraCleanup/");

  for (int i=0; i<int(Dirs.size()); i++) {

    std::string prefix = "dummy";

    if (size_t(Dirs[i].find("Calo")) != string::npos) prefix = "Calo";
    if (size_t(Dirs[i].find("Pf"))   != string::npos) prefix = "Pf";
    //TCMet related plots are removed
    //if (size_t(Dirs[i].find("Tc"))   != string::npos) prefix = "";

    for (std::vector<std::string>::const_iterator ic=metFolders.begin();
	 ic!=metFolders.end(); ic++) {

      std::string dirName = Dirs[i] + "/" + *ic;

      MEx = dqm->get(dirName + "METTask_" + prefix + "MEx");
      MEy = dqm->get(dirName + "METTask_" + prefix + "MEx");

      if (MEx && MEx->kind() == MonitorElement::DQM_KIND_TH1F) {
	if (MEx->getTH1F()->GetEntries() > 50) MEx->getTH1F()->Fit("gaus", "q");
      }

      if (MEy && MEy->kind() == MonitorElement::DQM_KIND_TH1F) {
	if (MEy->getTH1F()->GetEntries() > 50) MEy->getTH1F()->Fit("gaus", "q");
      }
    }
  }


  // SUSY
  //----------------------------------------------------------------------------
  dqm->setCurrentFolder(SUSYFolder);
  Dirs = dqm->getSubdirs();
  for (int i=0; i<int(Dirs.size()); i++)
    {
      size_t found = Dirs[i].find("Alpha");
      if (found!=string::npos) continue;
      if(!dqm->dirExists(Dirs[i])){
	edm::LogError(messageLoggerCatregory)<< "Directory "<<Dirs[i]<<" doesn't exist!!";
	continue;
      }      
      vector<MonitorElement*> histoVector = dqm->getContents(Dirs[i]);
      for (int i=0; i<int(histoVector.size()); i++) {
	QuantilePlots(histoVector[i],_quantile);
      } 
    }
}


void SusyPostProcessor::endJob(){}
