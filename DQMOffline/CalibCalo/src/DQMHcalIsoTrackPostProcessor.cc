#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMServices/Core/interface/DQMEDHarvester.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include <iostream>
#include <string.h>
#include <iomanip>
#include<fstream>
#include <math.h>


class DQMHcalIsoTrackPostProcessor : public DQMEDHarvester {
 public:
  DQMHcalIsoTrackPostProcessor(const edm::ParameterSet& pset);
  ~DQMHcalIsoTrackPostProcessor() {};

//  void analyze(const edm::Event& event, const edm::EventSetup& eventSetup) override {};
  virtual void dqmEndJob(DQMStore::IBooker &, DQMStore::IGetter &) override; //performed in the endJob

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


void DQMHcalIsoTrackPostProcessor::dqmEndJob(DQMStore::IBooker & ibooker, DQMStore::IGetter & igetter)
{

  if(igetter.dirExists(subDir_)) igetter.cd(subDir_);
  else {
   edm::LogWarning("DQMHcalIsoTrackPostProcessor") << "cannot find directory: " << subDir_ << " , skipping";
    return;
  }

  MonitorElement* hPurityEta=ibooker.book1D("hPurityEta","Purity of sample vs eta",16,-2,2);
  MonitorElement* hPurityPhi=ibooker.book1D("hPurityPhi","Purity of sample vs phi",16,-3.2,3.2);

  MonitorElement* hSumOffEta=ibooker.book1D("hSumOffEta","hSumOffEta",16,-2,2);
  MonitorElement* hSumOffPhi=ibooker.book1D("hSumOffPhi","hSumOffPhi",16,-3.2,3.2);
  
  MonitorElement* hSumL3Eta=ibooker.book1D("hSumL3Eta","hSumL3Eta",16,-2,2);
  MonitorElement* hSumL3Phi=ibooker.book1D("hSumL3Phi","hSumL3Phi",16,-3.2,3.2);

  hSumOffEta->getTH1F()->Add(igetter.get(ibooker.pwd() + "/hOffEtaFP")->getTH1F(),1);
  hSumOffPhi->getTH1F()->Add(igetter.get(ibooker.pwd() + "/hOffPhiFP")->getTH1F(),1);
    
  hSumL3Eta->getTH1F()->Add(igetter.get(ibooker.pwd() + "/hl3eta")->getTH1F(),1);
  hSumL3Phi->getTH1F()->Add(igetter.get(ibooker.pwd() + "/hl3phi")->getTH1F(),1);
 

  hSumOffEta->getTH1F()->TH1F::Sumw2();
  hSumOffPhi->getTH1F()->TH1F::Sumw2();
  hSumL3Eta->getTH1F()->TH1F::Sumw2();
  hSumL3Phi->getTH1F()->TH1F::Sumw2();

  hPurityEta->getTH1F()->Divide(hSumOffEta->getTH1F(),hSumL3Eta->getTH1F(),1,1);
  hPurityPhi->getTH1F()->Divide(hSumOffPhi->getTH1F(),hSumL3Phi->getTH1F(),1,1);

//  if (saveToFile_) igetter.save(outputRootFileName_);
}

DEFINE_FWK_MODULE(DQMHcalIsoTrackPostProcessor);


