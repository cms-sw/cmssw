#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include <iostream>
#include <string.h>
#include <iomanip>
#include<fstream>
#include <math.h>


class DQMHcalIsoTrackPostProcessor : public edm::EDAnalyzer {
 public:
  DQMHcalIsoTrackPostProcessor(const edm::ParameterSet& pset);
  ~DQMHcalIsoTrackPostProcessor() {};

  void analyze(const edm::Event& event, const edm::EventSetup& eventSetup) {};
  void endRun(edm::Run const&, edm::EventSetup const&);
  void endJob();

 private:

  std::string subDir_;
  bool saveToFile_;
  std::string outputRootFileName_;

};


DQMHcalIsoTrackPostProcessor::DQMHcalIsoTrackPostProcessor(const edm::ParameterSet& pset)
{
  subDir_ = pset.getUntrackedParameter<std::string>("subDir");
  saveToFile_=pset.getParameter<bool>("saveToFile");
  outputRootFileName_=pset.getParameter<std::string>("outputFile");
}

void DQMHcalIsoTrackPostProcessor::endRun(edm::Run const& run, edm::EventSetup const& es)
{
}

void DQMHcalIsoTrackPostProcessor::endJob()
{

  DQMStore* dqm;
  dqm = 0;
  dqm = edm::Service<DQMStore>().operator->();

  if ( ! dqm ) {
    edm::LogInfo("DQMHcalIsoTrackPostProcessor") << "Cannot create DQMStore instance\n";
    return;
  }

  std::cout<<"endjob"<<std::endl;
  if(dqm->dirExists(subDir_)) dqm->cd(subDir_);
  else {
   edm::LogWarning("DQMHcalIsoTrackPostProcessor") << "cannot find directory: " << subDir_ << " , skipping";
    return;
  }

  MonitorElement* hPurityEta=dqm->book1D("hPurityEta","Purity of sample vs eta",16,-2,2);
  MonitorElement* hPurityPhi=dqm->book1D("hPurityPhi","Purity of sample vs phi",16,-3.2,3.2);

  MonitorElement* hSumOffEta=dqm->book1D("hSumOffEta","hSumOffEta",16,-2,2);
  MonitorElement* hSumOffPhi=dqm->book1D("hSumOffPhi","hSumOffPhi",16,-3.2,3.2);
  
  MonitorElement* hSumL3Eta=dqm->book1D("hSumL3Eta","hSumL3Eta",16,-2,2);
  MonitorElement* hSumL3Phi=dqm->book1D("hSumL3Phi","hSumL3Phi",16,-3.2,3.2);

  hSumOffEta->getTH1F()->Add(dqm->get(dqm->pwd() + "/hOffEtaFP")->getTH1F(),1);
  hSumOffPhi->getTH1F()->Add(dqm->get(dqm->pwd() + "/hOffPhiFP")->getTH1F(),1);
    
  hSumL3Eta->getTH1F()->Add(dqm->get(dqm->pwd() + "/hl3eta")->getTH1F(),1);
  hSumL3Phi->getTH1F()->Add(dqm->get(dqm->pwd() + "/hl3phi")->getTH1F(),1);
 

  hSumOffEta->getTH1F()->TH1F::Sumw2();
  hSumOffPhi->getTH1F()->TH1F::Sumw2();
  hSumL3Eta->getTH1F()->TH1F::Sumw2();
  hSumL3Phi->getTH1F()->TH1F::Sumw2();

  hPurityEta->getTH1F()->Divide(hSumOffEta->getTH1F(),hSumL3Eta->getTH1F(),1,1);
  hPurityPhi->getTH1F()->Divide(hSumOffPhi->getTH1F(),hSumL3Phi->getTH1F(),1,1);

  if (saveToFile_) dqm->save(outputRootFileName_);
}

DEFINE_FWK_MODULE(DQMHcalIsoTrackPostProcessor);


