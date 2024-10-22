#ifndef HLTrigReport_h
#define HLTrigReport_h

/** \class HLTrigReport
 *
 *  
 *  This class is an EDAnalyzer implementing TrigReport (statistics
 *  printed to log file) for HL triggers
 *
 *
 *  \author Martin Grunewald
 *
 */

#include <string>
#include <vector>

#include "DataFormats/Common/interface/TriggerResults.h"
#include "FWCore/Common/interface/TriggerNames.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"

//
// class declaration
//

namespace hltrigreport {
  struct Accumulate {
    Accumulate();
    Accumulate(size_t numHLNames,
               std::vector<std::vector<unsigned int>> const& hlIndex,
               std::vector<std::vector<unsigned int>> const& dsIndex);

    unsigned int nEvents_;  // number of events processed
    unsigned int nWasRun_;  // # where at least one HLT was run
    unsigned int nAccept_;  // # of accepted events
    unsigned int nErrors_;  // # where at least one HLT had error

    std::vector<unsigned int> hlWasRun_;  // # where HLT[i] was run
    std::vector<unsigned int> hltL1s_;    // # of events after L1 seed
    std::vector<unsigned int> hltPre_;    // # of events after HLT prescale
    std::vector<unsigned int> hlAccept_;  // # of events accepted by HLT[i]
    std::vector<unsigned int> hlAccTot_;  // # of events accepted by HLT[0] OR ... OR HLT[i]
    std::vector<unsigned int> hlErrors_;  // # of events with error in HLT[i]

    std::vector<std::vector<unsigned int>>
        hlAccTotDS_;  // hlAccTotDS_[ds][p] stores the # of accepted events by the 0-th to p-th paths in the ds-th dataset
    std::vector<unsigned int> hlAllTotDS_;  // hlAllTotDS_[ds] stores the # of accepted events in the ds-th dataset
    std::vector<std::vector<unsigned int>>
        dsAccTotS_;  // dsAccTotS_[s][ds] stores the # of accepted events by the 0-th to ds-th dataset in the s-th stream
    std::vector<unsigned int> dsAllTotS_;  // dsAllTotS_[s] stores the # of accepted events in the s-th stream

    void accumulate(Accumulate const&);
    void reset();
  };

}  // namespace hltrigreport

class HLTrigReport
    : public edm::one::EDAnalyzer<edm::one::WatchRuns, edm::LuminosityBlockCache<hltrigreport::Accumulate>> {
private:
  enum ReportEvery { NEVER = 0, EVERY_EVENT = 1, EVERY_LUMI = 2, EVERY_RUN = 3, EVERY_JOB = 4 };

public:
  using Accumulate = hltrigreport::Accumulate;

  explicit HLTrigReport(const edm::ParameterSet&);
  ~HLTrigReport() override;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  static ReportEvery decode(const std::string& value);

  void beginJob() override;
  void endJob() override;

  void beginRun(edm::Run const&, edm::EventSetup const&) override;
  void endRun(edm::Run const&, edm::EventSetup const&) override;

  std::shared_ptr<Accumulate> globalBeginLuminosityBlock(edm::LuminosityBlock const&,
                                                         edm::EventSetup const&) const override;
  void globalEndLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override;

  void analyze(const edm::Event&, const edm::EventSetup&) override;

  // names and event counts
  const std::vector<std::string>& streamNames() const;
  const std::vector<std::string>& datasetNames() const;

private:
  void reset();  // reset all counters

  hltrigreport::Accumulate& chooseAccumulate(edm::LuminosityBlockIndex index) {
    if (useLumiCache()) {
      return *luminosityBlockCache(index);
    }
    return accumulate_;
  }
  bool useLumiCache() const { return reportBy_ == EVERY_LUMI or serviceBy_ == EVERY_LUMI or resetBy_ == EVERY_LUMI; }
  bool readAfterLumi() const {
    return (reportBy_ == EVERY_RUN or reportBy_ == EVERY_JOB or serviceBy_ == EVERY_RUN or serviceBy_ == EVERY_JOB);
  }

  void updateConfigCache();

  void dumpReport(hltrigreport::Accumulate const& accumulate, std::string const& header = std::string()) const;
  void updateService(Accumulate const& accumulate) const;

  const edm::InputTag hlTriggerResults_;  // Input tag for TriggerResults
  const edm::EDGetTokenT<edm::TriggerResults> hlTriggerResultsToken_;
  bool configured_;  // is HLTConfigProvider configured ?

  std::vector<std::string> hlNames_;  // name of each HLT algorithm

  std::vector<std::vector<unsigned int>>
      hlIndex_;  // hlIndex_[ds][p] stores the hlNames_ index of the p-th path of the ds-th dataset

  std::vector<int> posL1s_;  // pos # of last L1 seed
  std::vector<int> posPre_;  // pos # of last HLT prescale

  std::vector<std::string> datasetNames_;                  // list of dataset names
  std::vector<std::vector<std::string>> datasetContents_;  // list of path names for each dataset
  bool isCustomDatasets_;  // true if the user overwrote the dataset definitions of the provenance with the CustomDatasets parameter
  std::vector<std::vector<unsigned int>>
      dsIndex_;  // dsIndex_[s][ds] stores the datasetNames_ index of the ds-th dataset of the s-th stream
  std::vector<std::string> streamNames_;                  // list of stream names
  std::vector<std::vector<std::string>> streamContents_;  // list of dataset names for each stream
  bool isCustomStreams_;  // true if the user overwrote the stream definitions of the provenance with the CustomSterams parameter
  std::string refPath_;    // name of the reference path for rate calculation
  unsigned int refIndex_;  // index of the reference path for rate calculation
  const double refRate_;   // rate of the reference path, the rate of all other paths will be normalized to this

  const ReportEvery reportBy_;   // dump report for every never/event/lumi/run/job
  const ReportEvery resetBy_;    // reset counters  every never/event/lumi/run/job
  const ReportEvery serviceBy_;  // call to service every never/event/lumi/run/job
  HLTConfigProvider hltConfig_;  // to get configuration for L1s/Pre

  hltrigreport::Accumulate accumulate_;
};

#endif  //HLTrigReport_h
