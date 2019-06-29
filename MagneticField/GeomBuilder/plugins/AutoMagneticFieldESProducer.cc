/** \class AutoMagneticFieldESProducer
 *
 *  Produce a magnetic field map corresponding to the current
 *  recorded in the condidtion DB.
 *
 *  \author Nicola Amapane 11/08
 */

#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EventSetup.h"
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

class IdealMagneticFieldRecord;

namespace magneticfield {
  class AutoMagneticFieldESProducer : public edm::ESProducer {
  public:
    AutoMagneticFieldESProducer(const edm::ParameterSet&);
    ~AutoMagneticFieldESProducer() override;

    std::unique_ptr<MagneticField> produce(const IdealMagneticFieldRecord&);

  private:
    std::tuple<const MagneticField*, const std::string&> closerModel(const IdealMagneticFieldRecord& iRecord,
                                                                     float current) const;
    const std::vector<int> nominalCurrents_;
    const std::vector<std::string> maps_;
    const int overrideCurrent_;
    edm::ESGetToken<RunInfo, RunInfoRcd> runInfoToken_;
    std::vector<edm::ESGetToken<MagneticField, IdealMagneticFieldRecord>> magFieldTokens_;
  };
}  // namespace magneticfield

using namespace std;
using namespace edm;
using namespace magneticfield;

AutoMagneticFieldESProducer::AutoMagneticFieldESProducer(const edm::ParameterSet& iConfig)
    : nominalCurrents_{iConfig.getUntrackedParameter<vector<int>>("nominalCurrents")},
      maps_{iConfig.getUntrackedParameter<vector<string>>("mapLabels")},
      overrideCurrent_{iConfig.getParameter<int>("valueOverride")},
      magFieldTokens_(maps_.size()) {
  auto cc = setWhatProduced(this, iConfig.getUntrackedParameter<std::string>("label", ""));

  if (maps_.empty() || (maps_.size() != nominalCurrents_.size())) {
    throw cms::Exception("InvalidParameter") << "Invalid values for parameters \"nominalCurrents\" and \"maps\"";
  }

  if (overrideCurrent_ < 0) {
    cc.setConsumes(runInfoToken_);
  }

  for (size_t i = 0; i < maps_.size(); ++i) {
    cc.setConsumes(magFieldTokens_[i], edm::ESInputTag{maps_[i]});
  }
}

AutoMagneticFieldESProducer::~AutoMagneticFieldESProducer() {}

std::unique_ptr<MagneticField> AutoMagneticFieldESProducer::produce(const IdealMagneticFieldRecord& iRecord) {
  float current = overrideCurrent_;

  string message;

  if (overrideCurrent_ < 0) {
    current = iRecord.get(runInfoToken_).m_avg_current;
    message = " (from RunInfo DB)";
  } else {
    message = " (from valueOverride card)";
  }

  const auto& model = closerModel(iRecord, current);

  edm::LogInfo("MagneticField|AutoMagneticField")
      << "Current: " << current << message << "; using map with label: " << std::get<1>(model);

  MagneticField* result = std::get<0>(model)->clone();

  return std::unique_ptr<MagneticField>(result);
}
std::tuple<const MagneticField*, const std::string&> AutoMagneticFieldESProducer::closerModel(
    const IdealMagneticFieldRecord& iRecord, float current) const {
  int i = 0;
  for (; i < (int)maps_.size() - 1; i++) {
    if (2 * current < nominalCurrents_[i] + nominalCurrents_[i + 1])
      break;
  }
  return std::make_tuple(&iRecord.get(magFieldTokens_[i]), maps_[i]);
}

#include "FWCore/Framework/interface/ModuleFactory.h"
DEFINE_FWK_EVENTSETUP_MODULE(AutoMagneticFieldESProducer);
