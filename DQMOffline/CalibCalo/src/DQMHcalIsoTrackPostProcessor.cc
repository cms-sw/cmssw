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

//#define DebugLog

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


DQMHcalIsoTrackPostProcessor::DQMHcalIsoTrackPostProcessor(const edm::ParameterSet& pset) {
  subDir_     = pset.getUntrackedParameter<std::string>("subDir");
  saveToFile_ = pset.getParameter<bool>("saveToFile");
  outputRootFileName_=pset.getParameter<std::string>("outputFile");
}


void DQMHcalIsoTrackPostProcessor::dqmEndJob(DQMStore::IBooker & ibooker, DQMStore::IGetter & igetter) {

  if (igetter.dirExists(subDir_)) {
    igetter.cd(subDir_);
  } else {
    edm::LogWarning("DQMHcalIsoTrackPostProcessor") << "cannot find directory: " << subDir_ << " , skipping";
    return;
  }

  MonitorElement  *hSumEta[4], *hSumPhi[4], *hPurityEta[3], *hPurityPhi[3];
  std::string      types[4] = {"L2","L2.5","L3","Off"};
  char             name[20], title[100];
  for (int i=0; i<4; ++i) {
    sprintf (name, "hSum%sEta", types[i].c_str()); 
    hSumEta[i] = ibooker.book1D(name,name,16,-2,2);
    hSumEta[i]->getTH1F()->TH1F::Sumw2();
    hSumPhi[i] = ibooker.book1D(name,name,16,-3.2,3.2);
    hSumPhi[i]->getTH1F()->TH1F::Sumw2();
    if (i < 3) {
      sprintf (name, "hPurity%sEta", types[i].c_str()); 
      sprintf (title,"Purity of %s sample vs #eta", types[i].c_str()); 
      hPurityEta[i] = ibooker.book1D(name, title,16,-2,2);
      sprintf (name, "hPurity%sPhi", types[i].c_str()); 
      sprintf (title,"Purity of %s sample vs #phi", types[i].c_str()); 
      hPurityPhi[i] = ibooker.book1D(name, title,16,-3.2,3.2);
    }
  }

  std::string hname;
  for (int i=0; i<4; ++i) {
    sprintf (name, "/heta%s", types[i].c_str()); 
    hname = ibooker.pwd() + std::string(name);
#ifdef DebugLog
    std::cout << "PostProcesor " << hname << " " << igetter.get(hname) << std::endl;
#endif
    hSumEta[i]->getTH1F()->Add(igetter.get(hname)->getTH1F(),1);
    sprintf (name, "/hphi%s", types[i].c_str()); 
    hname = ibooker.pwd() + std::string(name);
#ifdef DebugLog
    std::cout << "PostProcesor " << hname << " " << igetter.get(hname) << std::endl;
#endif
    hSumPhi[i]->getTH1F()->Add(igetter.get(hname)->getTH1F(),1);
  }

  for (int i=0; i<3; ++i) {
    hPurityEta[i]->getTH1F()->Divide(hSumEta[i+1]->getTH1F(),hSumEta[i]->getTH1F(),1,1);
    hPurityPhi[i]->getTH1F()->Divide(hSumPhi[i+1]->getTH1F(),hSumPhi[i]->getTH1F(),1,1);
  }

//if (saveToFile_) igetter.save(outputRootFileName_);
}

DEFINE_FWK_MODULE(DQMHcalIsoTrackPostProcessor);


