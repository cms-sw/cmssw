/** \class AutoMagneticFieldESProducer
 *
 *  Produce a magnetic field map corresponding to the current
 *  recorded in the condidtion DB.
 *
 *  \author Nicola Amapane 11/08
 */

#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/ESProductTag.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/GeomBuilder/src/MagGeoBuilderFromDDD.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

#include "CondFormats/RunInfo/interface/RunInfo.h"
#include "CondFormats/DataRecord/interface/RunSummaryRcd.h"

#include <string>
#include <sstream>
#include <iostream>
#include <vector>

namespace magneticfield {
  class AutoMagneticFieldESProducer : public edm::ESProducer {
  public:
    AutoMagneticFieldESProducer(const edm::ParameterSet&);
    ~AutoMagneticFieldESProducer() override;

    std::unique_ptr<MagneticField> produce(const IdealMagneticFieldRecord&);

  private:
    const std::string& closerModel(float current) const;
    const std::vector<int> nominalCurrents_;
    const std::vector<std::string> maps_;
    const int overrideCurrent_;
    edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> magFieldToken_;
  };
}  // namespace magneticfield

using namespace std;
using namespace edm;
using namespace magneticfield;

AutoMagneticFieldESProducer::AutoMagneticFieldESProducer(const edm::ParameterSet& iConfig)
    : nominalCurrents_{iConfig.getUntrackedParameter<vector<int>>("nominalCurrents")},
      maps_{iConfig.getUntrackedParameter<vector<string>>("mapLabels")},
      overrideCurrent_{iConfig.getParameter<int>("valueOverride")} {
  auto cc = setWhatProduced(this, iConfig.getUntrackedParameter<std::string>("label", ""));

  if (maps_.empty() || (maps_.size() != nominalCurrents_.size())) {
    throw cms::Exception("InvalidParameter") << "Invalid values for parameters \"nominalCurrents\" and \"maps\"";
  }

  if (overrideCurrent_ < 0) {
    cc.setMayConsume(
        magFieldToken_,
        [this](const auto& get, edm::ESTransientHandle<RunInfo> runInfo) {
          const auto current = runInfo->m_avg_current;
          const auto& model = closerModel(current);
          edm::LogInfo("MagneticField|AutoMagneticField")
              << "Current: " << current << "(from RunInfo DB); using map with label: " << model;
          return get("", model);
        },
        edm::ESProductTag<RunInfo, RunInfoRcd>("", ""));
  } else {
    const auto& model = closerModel(overrideCurrent_);
    edm::LogInfo("MagneticField|AutoMagneticField")
        << "Current: " << overrideCurrent_ << "(from valueOverride card); using map with label: " << model;
    cc.setConsumes(magFieldToken_, edm::ESInputTag{"", model});
  }
}

AutoMagneticFieldESProducer::~AutoMagneticFieldESProducer() {}

std::unique_ptr<MagneticField> AutoMagneticFieldESProducer::produce(const IdealMagneticFieldRecord& iRecord) {
  return std::unique_ptr<MagneticField>(iRecord.get(magFieldToken_).clone());
}
const std::string& AutoMagneticFieldESProducer::closerModel(float current) const {
  int i = 0;
  for (; i < (int)maps_.size() - 1; i++) {
    if (2 * current < nominalCurrents_[i] + nominalCurrents_[i + 1])
      return maps_[i];
  }
  return maps_[i];
}

#include "FWCore/Framework/interface/ModuleFactory.h"
DEFINE_FWK_EVENTSETUP_MODULE(AutoMagneticFieldESProducer);
