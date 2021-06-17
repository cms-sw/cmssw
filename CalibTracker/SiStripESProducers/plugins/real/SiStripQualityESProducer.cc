// -*- C++ -*-
//
// Package:    SiStripQualityESProducer
// Class:      SiStripQualityESProducer
//
/**\class SiStripQualityESProducer SiStripQualityESProducer.h CalibTracker/SiStripESProducers/plugins/real/SiStripQualityESProducer.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Domenico GIORDANO
//         Created:  Wed Oct  3 12:11:10 CEST 2007
//
//

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/ModuleFactory.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/SiStripCommon/interface/SiStripConstants.h"
#include "CalibFormats/SiStripObjects/interface/SiStripQuality.h"
#include "CalibTracker/Records/interface/SiStripDependentRecords.h"
#include "CalibTracker/SiStripCommon/interface/SiStripDetInfoFileReader.h"
#include "CondFormats/RunInfo/interface/RunInfo.h"

namespace {
  class ProductAdder {
  public:
    virtual ~ProductAdder() = default;
    virtual void add(const SiStripQualityRcd& iRecord, SiStripQuality& quality) const = 0;
  };

  template <typename Product, typename RealRecord>
  class ProductAdderT : public ProductAdder {
  public:
    ProductAdderT(edm::ESConsumesCollector& cc, const std::string& label)
        : token_{cc.consumesFrom<Product, RealRecord>(edm::ESInputTag{"", label})} {}
    void add(const SiStripQualityRcd& iRecord, SiStripQuality& quality) const override {
      quality.add(&iRecord.get(token_));
    }

  private:
    edm::ESGetToken<Product, RealRecord> token_;
  };

  template <typename Product, typename RealRecord>
  auto make_ProductAdder(edm::ESConsumesCollector& cc, const std::string& label) {
    return std::make_unique<ProductAdderT<Product, RealRecord>>(cc, label);
  }
}  // namespace

class SiStripQualityESProducer : public edm::ESProducer {
public:
  SiStripQualityESProducer(const edm::ParameterSet&);
  ~SiStripQualityESProducer() override{};

  std::unique_ptr<SiStripQuality> produce(const SiStripQualityRcd&);

private:
  std::vector<std::unique_ptr<const ProductAdder>> productAdders_;
  edm::ESGetToken<RunInfo, RunInfoRcd> runInfoToken_;

  const double thresholdForReducedGranularity_;
  const bool printDebugOutput_;
  const bool useEmptyRunInfo_;
  const bool reduceGranularity_;
};

SiStripQualityESProducer::SiStripQualityESProducer(const edm::ParameterSet& iConfig)
    : thresholdForReducedGranularity_{iConfig.getParameter<double>("ThresholdForReducedGranularity")},
      printDebugOutput_{iConfig.getParameter<bool>("PrintDebugOutput")},
      useEmptyRunInfo_{iConfig.getParameter<bool>("UseEmptyRunInfo")},
      reduceGranularity_{iConfig.getParameter<bool>("ReduceGranularity")} {
  auto cc = setWhatProduced(this);

  edm::LogInfo("SiStripQualityESProducer") << "ctor";

  bool doRunInfo = false;
  std::string runInfoTagName = "";

  auto toGet = iConfig.getParameter<std::vector<edm::ParameterSet>>("ListOfRecordToMerge");

  for (const auto& toGetPSet : toGet) {
    auto tagName = toGetPSet.getParameter<std::string>("tag");
    auto recordName = toGetPSet.getParameter<std::string>("record");

    edm::LogInfo("SiStripQualityESProducer")
        << "[SiStripQualityESProducer::ctor] Going to get data from record " << recordName << " with tag " << tagName;

    if (recordName == "SiStripBadModuleRcd") {
      productAdders_.emplace_back(make_ProductAdder<SiStripBadStrip, SiStripBadModuleRcd>(cc, tagName));
    } else if (recordName == "SiStripBadFiberRcd") {
      productAdders_.emplace_back(make_ProductAdder<SiStripBadStrip, SiStripBadFiberRcd>(cc, tagName));
    } else if (recordName == "SiStripBadChannelRcd") {
      productAdders_.emplace_back(make_ProductAdder<SiStripBadStrip, SiStripBadChannelRcd>(cc, tagName));
    } else if (recordName == "SiStripBadStripRcd") {
      productAdders_.emplace_back(make_ProductAdder<SiStripBadStrip, SiStripBadStripRcd>(cc, tagName));
    } else if (recordName == "SiStripDetCablingRcd") {
      productAdders_.emplace_back(make_ProductAdder<SiStripDetCabling, SiStripDetCablingRcd>(cc, tagName));
    } else if (recordName == "SiStripDetVOffRcd") {
      productAdders_.emplace_back(make_ProductAdder<SiStripDetVOff, SiStripDetVOffRcd>(cc, tagName));
    } else if (recordName == "RunInfoRcd") {
      runInfoTagName = tagName;
      doRunInfo = true;
    } else {
      // Would it make sense to elevate this to an exception?
      edm::LogError("SiStripQualityESProducer")
          << "[SiStripQualityESProducer::ctor] Skipping the requested data for unexisting record " << recordName
          << " with tag " << tagName << std::endl;
      continue;
    }
  }

  if (doRunInfo) {
    runInfoToken_ = cc.consumes(edm::ESInputTag{"", runInfoTagName});
  }
}

std::unique_ptr<SiStripQuality> SiStripQualityESProducer::produce(const SiStripQualityRcd& iRecord) {
  const auto detInfo =
      SiStripDetInfoFileReader::read(edm::FileInPath{SiStripDetInfoFileReader::kDefaultFile}.fullPath());
  auto quality = std::make_unique<SiStripQuality>(detInfo);
  edm::LogInfo("SiStripQualityESProducer") << "produce called";

  // Set the debug output level
  quality->setPrintDebugOutput(printDebugOutput_);
  // Set the protection against empty RunInfo objects
  quality->setUseEmptyRunInfo(useEmptyRunInfo_);

  for (const auto& adder : productAdders_) {
    adder->add(iRecord, *quality);
  }

  // We do this after all the others so we know it is done after the DetCabling (if any)
  if (runInfoToken_.isInitialized()) {
    quality->add(&iRecord.get(runInfoToken_));
  }

  quality->cleanUp();

  if (reduceGranularity_) {
    quality->ReduceGranularity(thresholdForReducedGranularity_);
    quality->cleanUp(true);
  }

  quality->fillBadComponents();

  return quality;
}

DEFINE_FWK_EVENTSETUP_MODULE(SiStripQualityESProducer);
