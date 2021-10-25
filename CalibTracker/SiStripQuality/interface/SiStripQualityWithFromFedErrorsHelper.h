#ifndef CALIBTRACKER_SISTRIPESPRODUCERS_INTERFACE_SISTRIPQUALITYHELPERS_H
#define CALIBTRACKER_SISTRIPESPRODUCERS_INTERFACE_SISTRIPQUALITYHELPERS_H

#include <memory>
#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/ESWatcher.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "CalibFormats/SiStripObjects/interface/SiStripQuality.h"
#include "CalibTracker/Records/interface/SiStripQualityRcd.h"
#include "CondFormats/SiStripObjects/interface/SiStripFedCabling.h"

class SiStripFedCablingRcd;

namespace sistrip {
  /**
   * Create a SiStripQuality record from the list of detected Fed errors
   * in the SiStrip/ReadoutView/FedIdVsApvId DQM histogram
   */
  std::unique_ptr<SiStripQuality> badStripFromFedErr(dqm::harvesting::DQMStore::IGetter& dqmStore,
                                                     const SiStripFedCabling& fedCabling,
                                                     float cutoff);

  /**
   * Create a SiStripQuality record from the list of detected Fed errors
   * in the SiStrip/ReadoutView/FedIdVsApvId DQM histogram, reading from
   * a DQM file in legacy TDirectory from instead of the DQM store
   */
  std::unique_ptr<SiStripQuality> badStripFromFedErrLegacyDQMFile(const std::string& fileName,
                                                                  unsigned int runNumber,
                                                                  const SiStripFedCabling& fedCabling,
                                                                  float cutoff);

  /**
   * Helper class for making the merged SiStripQuality available in dqmEndJob,
   * and optionally merge the bad components from FED errors
   * (from the DQM store or a legacy TDirectory DQM file).
   */
}  // namespace sistrip

class SiStripQualityWithFromFedErrorsHelper {
public:
  SiStripQualityWithFromFedErrorsHelper(const edm::ParameterSet& iConfig,
                                        edm::ConsumesCollector iC,
                                        bool keepCopy = false) {
    const auto& fedErrConfig = iConfig.getParameter<edm::ParameterSet>("BadComponentsFromFedErrors");
    addBadCompFromFedErr_ = fedErrConfig.getParameter<bool>("Add");
    fedErrCutoff_ = float(fedErrConfig.getParameter<double>("Cutoff"));
    fedErrLegacyFile_ = fedErrConfig.getParameter<std::string>("LegacyDQMFile");
    fedErrFileRunNumber_ = fedErrConfig.getParameter<unsigned int>("FileRunNumber");
    stripQualityToken_ = iC.esConsumes<edm::Transition::EndRun>(
        edm::ESInputTag{"", iConfig.getParameter<std::string>("StripQualityLabel")});
    if (addBadCompFromFedErr_) {
      fedCablingToken_ = iC.esConsumes<edm::Transition::EndRun>();
    }
    // can be set if a copy should be made even if BadComponentsFromFedErrors is false
    // (e.g. for producing a new payloads)
    keepCopy_ = keepCopy || addBadCompFromFedErr_;
  }

  static void fillDescription(edm::ParameterSetDescription& desc) {
    desc.add<std::string>("StripQualityLabel", "");
    edm::ParameterSetDescription descFedErr;
    descFedErr.add<bool>("Add", false);
    descFedErr.add<double>("Cutoff", 0.8);
    descFedErr.add<std::string>("LegacyDQMFile", "");
    descFedErr.add<unsigned int>("FileRunNumber", -1);
    desc.add<edm::ParameterSetDescription>("BadComponentsFromFedErrors", descFedErr);
  }

  bool endRun(const edm::EventSetup&);
  const SiStripQuality& getMergedQuality(dqm::harvesting::DQMStore::IGetter& getter);

  bool addBadCompFromFedErr() const { return addBadCompFromFedErr_; }
  const edm::ESGetToken<SiStripQuality, SiStripQualityRcd>& qualityToken() const { return stripQualityToken_; }

private:
  bool addBadCompFromFedErr_ = false;
  float fedErrCutoff_;
  std::string fedErrLegacyFile_;
  unsigned int fedErrFileRunNumber_;
  edm::ESGetToken<SiStripQuality, SiStripQualityRcd> stripQualityToken_;
  edm::ESGetToken<SiStripFedCabling, SiStripFedCablingRcd> fedCablingToken_;
  edm::ESWatcher<SiStripQualityRcd> stripQualityWatcher_;
  std::unique_ptr<SiStripFedCabling> fedCabling_;
  std::unique_ptr<SiStripQuality> mergedQuality_;
  bool merged_ = false;
  bool keepCopy_;
};

#endif  // CALIBTRACKER_SISTRIPESPRODUCERS_INTERFACE_SISTRIPQUALITYHELPERS_H
