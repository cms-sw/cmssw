// system include files
#include <memory>
#include <utility>

// user include files
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "CalibFormats/SiStripObjects/interface/SiStripGain.h"
#include "CondFormats/SiStripObjects/interface/SiStripApvGain.h"
#include "CondFormats/DataRecord/interface/SiStripCondDataRecords.h"
#include "CalibTracker/Records/interface/SiStripDependentRecords.h"
#include "CalibTracker/SiStripCommon/interface/SiStripDetInfoFileReader.h"

#include "SiStripGainFactor.h"

//
// class declaration
//
class SiStripGainESProducer : public edm::ESProducer {
public:
  SiStripGainESProducer(const edm::ParameterSet&);
  ~SiStripGainESProducer() override{};

  std::unique_ptr<SiStripGain> produce(const SiStripGainRcd&);

private:
  class GainGetter {
  public:
    GainGetter(std::string record, std::string label) : recordLabel_{std::move(record), std::move(label)} {}
    virtual ~GainGetter() = default;
    virtual const SiStripApvGain& gain(const SiStripGainRcd& rcd) const = 0;

    const auto& recordLabel() const { return recordLabel_; }

  private:
    std::pair<std::string, std::string> recordLabel_;
  };

  template <typename Record>
  class GainGetterT : public GainGetter {
  public:
    GainGetterT(edm::ESConsumesCollector& cc, std::string record, std::string label)
        : GainGetter(std::move(record), std::move(label)),
          token_{cc.consumesFrom<SiStripApvGain, Record>(edm::ESInputTag{"", recordLabel().second})} {}

    const SiStripApvGain& gain(const SiStripGainRcd& rcd) const override { return rcd.get(token_); }

  private:
    edm::ESGetToken<SiStripApvGain, Record> token_;
  };

  template <typename Record>
  auto make_GainGetter(edm::ESConsumesCollector& cc, std::string record, std::string label) {
    return std::make_unique<GainGetterT<Record>>(cc, std::move(record), std::move(label));
  }

  std::vector<std::unique_ptr<GainGetter>> gainGetters_;

  SiStripGainFactor factor_;
};

SiStripGainESProducer::SiStripGainESProducer(const edm::ParameterSet& iConfig) : factor_{iConfig} {
  auto cc = setWhatProduced(this);

  auto apvGainLabels = iConfig.getParameter<std::vector<edm::ParameterSet>>("APVGain");
  if (apvGainLabels.empty()) {
    throw cms::Exception("Configuration") << "Got empty APVGain vector, but need at least one entry";
  }

  // Fill the vector of apv labels
  for (const auto& gainPSet : apvGainLabels) {
    // Shouldn't all these parameters be tracked?
    auto record = gainPSet.getParameter<std::string>("Record");
    auto label = gainPSet.getUntrackedParameter<std::string>("Label", "");
    if (record == "SiStripApvGainRcd")
      gainGetters_.emplace_back(make_GainGetter<SiStripApvGainRcd>(cc, record, label));
    else if (record == "SiStripApvGain2Rcd")
      gainGetters_.emplace_back(make_GainGetter<SiStripApvGain2Rcd>(cc, record, label));
    else if (record == "SiStripApvGain3Rcd")
      gainGetters_.emplace_back(make_GainGetter<SiStripApvGain3Rcd>(cc, record, label));
    else
      throw cms::Exception("Configuration")
          << "SiStripGainESProducer::ctor ERROR: unrecognized record name " << record << std::endl
          << "please specify one of: SiStripApvGainRcd, SiStripApvGain2Rcd, SiStripApvGain3Rcd";
    factor_.push_back_norm(gainPSet.getUntrackedParameter<double>("NormalizationFactor", 1.));
  }
  factor_.resetIfBadNorm();
}

std::unique_ptr<SiStripGain> SiStripGainESProducer::produce(const SiStripGainRcd& iRecord) {
  const auto detInfo =
      SiStripDetInfoFileReader::read(edm::FileInPath{SiStripDetInfoFileReader::kDefaultFile}.fullPath());

  const auto& apvGain = gainGetters_[0]->gain(iRecord);
  // Create a new gain object and insert the ApvGain
  auto gain = std::make_unique<SiStripGain>(apvGain, factor_.get(apvGain, 0), gainGetters_[0]->recordLabel(), detInfo);

  for (unsigned int i = 1; i < gainGetters_.size(); ++i) {
    const auto& apvGain = gainGetters_[i]->gain(iRecord);
    // Add the new ApvGain to the gain object
    gain->multiply(apvGain, factor_.get(apvGain, i), gainGetters_[i]->recordLabel(), detInfo);
  }

  return gain;
}

#include "FWCore/Framework/interface/ModuleFactory.h"

DEFINE_FWK_EVENTSETUP_MODULE(SiStripGainESProducer);
