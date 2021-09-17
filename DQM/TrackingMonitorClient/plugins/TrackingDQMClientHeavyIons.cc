
#include "DQM/TrackingMonitorClient/plugins/TrackingDQMClientHeavyIons.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <TH1F.h>
#include <TClass.h>
#include <TString.h>
#include <string>
#include <cmath>
#include <climits>
#include <boost/tokenizer.hpp>

using namespace std;
using namespace edm;

typedef dqm::harvesting::MonitorElement ME;

TrackingDQMClientHeavyIons::TrackingDQMClientHeavyIons(const edm::ParameterSet& pset) {
  TopFolder_ = pset.getParameter<std::string>("FolderName");
}

void TrackingDQMClientHeavyIons::dqmEndJob(DQMStore::IBooker& ibooker, DQMStore::IGetter& igetter) {
  typedef vector<string> vstring;

  // Update 2014-04-02
  // Migrated back to the endJob. the DQMFileSaver logic has
  // to be reviewed to guarantee that the endJob is properly
  // considered. The splitting per run is done centrally when
  // running the harvesting in production

  // Update 2009-09-23
  // Migrated all code from endJob to this function
  // endJob is not necessarily called in the proper sequence
  // and does not necessarily book histograms produced in
  // that step.
  // It more robust to do the histogram manipulation in
  // this endRun function
  // needed to access the DQMStore::save method

  ibooker.cd();
  ibooker.setCurrentFolder(TopFolder_);

  histName = "DCAStats_";
  DCAStats = ibooker.book2D(histName, histName, 2, 0, 2, 4, 0, 4);
  DCAStats->getTH2F()->GetYaxis()->SetBinLabel(1, "Mean");
  DCAStats->getTH2F()->GetYaxis()->SetBinLabel(2, "RMS, #sigma");
  DCAStats->getTH2F()->GetYaxis()->SetBinLabel(3, "Skewness ,#gamma_{1}");
  DCAStats->getTH2F()->GetYaxis()->SetBinLabel(4, "Kurtosis, #gamma_{2}");
  DCAStats->setBinLabel(1, "Longitudinal");
  DCAStats->setBinLabel(2, "Transverse");
  DCAStats->setOption("text");

  histName = "LongDCASig_HeavyIonTk";
  ME* element = igetter.get(TopFolder_ + "/" + histName);
  //Longitudinal First
  DCAStats->setBinContent(1, 1, element->getTH1F()->GetMean());      //mean
  DCAStats->setBinContent(1, 2, element->getTH1F()->GetRMS());       //rms
  DCAStats->setBinContent(1, 3, element->getTH1F()->GetSkewness());  //skewness
  DCAStats->setBinContent(1, 4, element->getTH1F()->GetKurtosis());  //kurtosis
  //Transverse
  histName = "TransDCASig_HeavyIonTk";
  ME* element1 = igetter.get(TopFolder_ + "/" + histName);
  //Longitudinal First
  DCAStats->setBinContent(2, 1, element1->getTH1F()->GetMean());      //mean
  DCAStats->setBinContent(2, 2, element1->getTH1F()->GetRMS());       //rms
  DCAStats->setBinContent(2, 3, element1->getTH1F()->GetSkewness());  //skewness
  DCAStats->setBinContent(2, 4, element1->getTH1F()->GetKurtosis());  //kurtosis
}
