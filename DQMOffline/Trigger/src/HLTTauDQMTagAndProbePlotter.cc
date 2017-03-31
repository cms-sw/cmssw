#include "DQMOffline/Trigger/interface/HLTTauDQMTagAndProbePlotter.h"
#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"
#include "DataFormats/HLTReco/interface/TriggerEvent.h"
#include "DataFormats/HLTReco/interface/TriggerTypeDefs.h"

#include <boost/algorithm/string.hpp>

namespace {
  std::string stripVersion(const std::string& pathName) {
    size_t versionStart = pathName.rfind("_v");
    if(versionStart == std::string::npos)
      return pathName;
    return pathName.substr(0, versionStart);
  }
}

HLTTauDQMTagAndProbePlotter::HLTTauDQMTagAndProbePlotter(const std::string& pathNameNum, const std::string& pathNameDen, const HLTConfigProvider& HLTCP,
                                           bool doRefAnalysis, const std::string& dqmBaseFolder,
                                           const std::string& hltProcess, int nbins,
                                           double xmax,
                                           std::string& xvariableName):
  HLTTauDQMPlotter(stripVersion(pathNameNum), dqmBaseFolder),
  nbins_(nbins),
  xmax_(xmax),
  doRefAnalysis_(doRefAnalysis),
  xvariable(xvariableName),
  hltDenominatorPath_(pathNameDen, hltProcess, doRefAnalysis_, HLTCP),
  hltNumeratorPath_(pathNameNum, hltProcess, doRefAnalysis_, HLTCP)
{
  configValid_ = configValid_ && hltDenominatorPath_.isValid() && hltNumeratorPath_.isValid();
  boost::algorithm::to_lower(xvariable);
}

#include <algorithm>
void HLTTauDQMTagAndProbePlotter::bookHistograms(DQMStore::IBooker &iBooker) {
  if(!isValid())
    return;

  // Book histograms

  // Efficiency helpers
  if(doRefAnalysis_) {
    iBooker.setCurrentFolder(triggerTag()+"/helpers");
    h_num = iBooker.book1D(xvariable+"EtEffNum",    "", nbins_, 0, xmax_);
    h_den = iBooker.book1D(xvariable+"EtEffDenom",    "", nbins_, 0, xmax_);
    iBooker.setCurrentFolder(triggerTag());
  }
}


HLTTauDQMTagAndProbePlotter::~HLTTauDQMTagAndProbePlotter() {}

void HLTTauDQMTagAndProbePlotter::analyze(const edm::TriggerResults& triggerResults, const trigger::TriggerEvent& triggerEvent, const HLTTauDQMOfflineObjects& refCollection) {

  if(doRefAnalysis_) {

    if(xvariable == "tau"){
      for(const LV& offlineObject: refCollection.taus) {
        double xvar = offlineObject.pt();
        if(hltDenominatorPath_.fired(triggerResults)) {
          h_den->Fill(xvar);
          if(hltNumeratorPath_.fired(triggerResults)) {
            h_num->Fill(xvar);
          }
        }
      }
    }

    if(xvariable == "muon"){
      for(const LV& offlineObject: refCollection.muons) {
        double xvar = offlineObject.pt();
        if(hltDenominatorPath_.fired(triggerResults)) {
          h_den->Fill(xvar);
          if(hltNumeratorPath_.fired(triggerResults)) {
            h_num->Fill(xvar);
          }
        }
      }
    }

    if(xvariable == "electron"){
      for(const LV& offlineObject: refCollection.electrons) {
        double xvar = offlineObject.pt();
        if(hltDenominatorPath_.fired(triggerResults)) {
          h_den->Fill(xvar);
          if(hltNumeratorPath_.fired(triggerResults)) {
            h_num->Fill(xvar);
          }
        }
      }
    }

    if(xvariable == "met"){
      for(const LV& offlineObject: refCollection.met) {
        double xvar = offlineObject.pt();
        if(hltDenominatorPath_.fired(triggerResults)) {
          h_den->Fill(xvar);
          if(hltNumeratorPath_.fired(triggerResults)) {
            h_num->Fill(xvar);
          }
        }
      }
    }

  }

}
