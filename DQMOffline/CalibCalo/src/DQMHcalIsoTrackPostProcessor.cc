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

#define DebugLog

class DQMHcalIsoTrackPostProcessor : public DQMEDHarvester {

public:
  DQMHcalIsoTrackPostProcessor(const edm::ParameterSet& pset);
  ~DQMHcalIsoTrackPostProcessor() {};

//  void analyze(const edm::Event& event, const edm::EventSetup& eventSetup) override {};
  virtual void dqmEndJob(DQMStore::IBooker &, DQMStore::IGetter &) override; //performed in the endJob

private:

  std::string subDir_;
};


DQMHcalIsoTrackPostProcessor::DQMHcalIsoTrackPostProcessor(const edm::ParameterSet& pset) {
  subDir_     = pset.getUntrackedParameter<std::string>("subDir");
}


void DQMHcalIsoTrackPostProcessor::dqmEndJob(DQMStore::IBooker & ibooker, DQMStore::IGetter & igetter) {

  if (igetter.dirExists(subDir_)) {
    igetter.cd(subDir_);
  } else {
    edm::LogWarning("DQMHcalIsoTrackPostProcessor") << "cannot find directory: " << subDir_ << " , skipping";
    return;
  }

  MonitorElement  *hSumEta[4], *hSumPhi[4], *hPurityEta[3], *hPurityPhi[3];
  std::string      types[4] = {"L2","L2x","L3","Off"};
  char             name[100], title[200];
  for (int i=0; i<4; ++i) {
    sprintf (name, "/heta%s", types[i].c_str()); 
    std::string hname1 = ibooker.pwd() + std::string(name);
    int    nbinEta = igetter.get(hname1)->getTH1F()->GetNbinsX();
    double xminEta = igetter.get(hname1)->getTH1F()->GetBinLowEdge(1);
    double xmaxEta = igetter.get(hname1)->getTH1F()->GetBinLowEdge(nbinEta) +
      igetter.get(hname1)->getTH1F()->GetBinWidth(nbinEta);
    sprintf (name, "/hphi%s", types[i].c_str()); 
    std::string hname2 = ibooker.pwd() + std::string(name);
    int    nbinPhi = igetter.get(hname2)->getTH1F()->GetNbinsX();
    double xminPhi = igetter.get(hname2)->getTH1F()->GetBinLowEdge(1);
    double xmaxPhi = igetter.get(hname2)->getTH1F()->GetBinLowEdge(nbinEta) +
      igetter.get(hname2)->getTH1F()->GetBinWidth(nbinEta);
    sprintf (name, "hSum%sEta", types[i].c_str()); 
    hSumEta[i] = ibooker.book1D(name,name,nbinEta,xminEta,xmaxEta);
    sprintf (name, "hSum%sPhi", types[i].c_str()); 
    hSumPhi[i] = ibooker.book1D(name,name,nbinPhi,xminPhi,xmaxPhi);
    if (i < 3) {
      sprintf (name, "hPurity%sEta", types[i].c_str()); 
      sprintf (title,"Purity of %s sample vs #eta", types[i].c_str()); 
      hPurityEta[i] = ibooker.book1D(name, title,nbinEta,xminEta,xmaxEta);
      sprintf (name, "hPurity%sPhi", types[i].c_str()); 
      sprintf (title,"Purity of %s sample vs #phi", types[i].c_str()); 
      hPurityPhi[i] = ibooker.book1D(name, title,nbinPhi,xminPhi,xmaxPhi);
    }
  }

  for (int i=0; i<4; ++i) {
    sprintf (name, "/heta%s", types[i].c_str()); 
    std::string hname1 = ibooker.pwd() + std::string(name);
#ifdef DebugLog
    std::cout << "PostProcesor " << hname1 << " " << igetter.get(hname1) << std::endl;
#endif
    hSumEta[i]->getTH1F()->Add(igetter.get(hname1)->getTH1F(),1);
    sprintf (name, "/hphi%s", types[i].c_str()); 
    std::string hname2 = ibooker.pwd() + std::string(name);
#ifdef DebugLog
    std::cout << "PostProcesor " << hname2 << " " << igetter.get(hname2) << std::endl;
#endif
    hSumPhi[i]->getTH1F()->Add(igetter.get(hname2)->getTH1F(),1);
  }

  for (int i=0; i<3; ++i) {
    hPurityEta[i]->getTH1F()->Divide(hSumEta[i+1]->getTH1F(),hSumEta[i]->getTH1F(),1,1);
    hPurityPhi[i]->getTH1F()->Divide(hSumPhi[i+1]->getTH1F(),hSumPhi[i]->getTH1F(),1,1);
  }

}

DEFINE_FWK_MODULE(DQMHcalIsoTrackPostProcessor);
