// -*- C++ -*-
//
// Package:    Alignment/CommonAlignment
// Class:      APVModeFilter
//
/**\class APVModeFilter APVModeFilter.cc Alignment/CommonAlignment/plugins/APVModeFilter.cc

 Description: Plugin to filter events based on the APV mode

 Implementation:
     The filter checks the bit configuration used for a given run and selects
     only events according to the configured APV mode.

     General reference:
     https://twiki.cern.ch/twiki/bin/view/CMS/SiStripConditionObjects#SiStripLatency

     Document describing the bit configuration (section 5.5):
     https://cds.cern.ch/record/1069892/files/cer-002725643.pdf

     Summary given here:
     https://hypernews.cern.ch/HyperNews/CMS/get/recoTracking/1590/1/1/1.html

     bit 1: 0 = 3-sample,      1 = 1-sample
     bit 3: 0 = deconvolution, 1 = peak

     if both bits are zero: deco
     if both bits are one: peak
     if 1 is zero and bit 3 is one: multi (not used in actual data taking)

*/
//
// Original Author:  Gregor Mittag
//         Created:  Thu, 03 Dec 2015 16:51:33 GMT
//
//


// system include files
#include <bitset>
#include <array>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "CondFormats/SiStripObjects/interface/SiStripLatency.h"
#include "CondFormats/DataRecord/interface/SiStripCondDataRecords.h"

//
// class declaration
//

class APVModeFilter : public edm::stream::EDFilter<> {
public:
  explicit APVModeFilter(const edm::ParameterSet&);
  ~APVModeFilter() = default;

  static void fillDescriptions(edm::ConfigurationDescriptions&);

private:
  virtual bool filter(edm::Event&, const edm::EventSetup&) override;
  virtual void beginRun(const edm::Run&, const edm::EventSetup&) override;

  using BitMask = std::bitset<16>; /// APV mode is encoded in uin16_t

  /// converts configuration parameter into type used for APV mode filtering
  BitMask convertMode(const std::string& mode) const;

  /// converts latency record content into type used for APV mode filtering
  BitMask convertMode(const uint16_t& mode) const;


  // ----------member data ---------------------------

  /// bits of interest for the APV mode
  static constexpr std::array<size_t, 2> bits_ = {{1, 3}};
  static constexpr BitMask deco_ = BitMask(0);  /// deco mode bit mask (0000)
  static constexpr BitMask peak_ = BitMask(10); /// peak mode bit mask (1010)
  static constexpr BitMask multi_ = BitMask(8); /// multi mode bit mask (1000)

  const BitMask mode_;          /// APV mode that is filtered
  BitMask modeCurrentRun_;      /// APV mode of the current run
};


//
// static data member definitions
//
constexpr std::array<size_t, 2> APVModeFilter::bits_;
constexpr APVModeFilter::BitMask APVModeFilter::deco_;
constexpr APVModeFilter::BitMask APVModeFilter::peak_;
constexpr APVModeFilter::BitMask APVModeFilter::multi_;


//
// constructors and destructor
//
APVModeFilter::APVModeFilter(const edm::ParameterSet& iConfig) :
  mode_(convertMode(iConfig.getUntrackedParameter<std::string>("apvMode"))) {
}


//
// member functions
//

// ------------ method called on each new Event  ------------
bool
APVModeFilter::filter(edm::Event&, const edm::EventSetup&) {
  return mode_ == modeCurrentRun_;
}

// ------------ method called when starting to processes a run  ------------
void
APVModeFilter::beginRun(const edm::Run&, const edm::EventSetup& iSetup) {
  edm::ESHandle<SiStripLatency> siStripLatency;
  iSetup.get<SiStripLatencyRcd>().get(siStripLatency);
  auto product = siStripLatency.product();
  modeCurrentRun_ = convertMode(product->singleMode());
}

APVModeFilter::BitMask
APVModeFilter::convertMode(const std::string& mode) const {
  if (mode == "deco") {
    return deco_;
  } else if (mode == "peak") {
    return peak_;
  } else if (mode == "multi") {
    return multi_;
  } else {
    throw cms::Exception("BadConfig")
      << "Your choice for the APV mode ('" << mode
      << "') is invalid.\nValid APV modes: deco, peak, multi" << std::endl;
  }
}

APVModeFilter::BitMask
APVModeFilter::convertMode(const uint16_t& mode) const {
  BitMask input(mode);
  BitMask result;
  for (const auto& bit: bits_) result.set(bit, input[bit]);
  return result;
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void
APVModeFilter::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.setComment("Filters events with the APV mode 'apvMode' (deco/peak/multi).");
  desc.addUntracked<std::string>("apvMode", "deco");
  descriptions.add("apvModeFilter", desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(APVModeFilter);
