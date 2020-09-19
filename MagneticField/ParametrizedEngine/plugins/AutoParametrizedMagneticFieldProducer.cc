/** \class AutoParametrizedMagneticFieldProducer
 *
 *   Description: Producer for parametrized Magnetics Fields, with value scaled depending on current.
 *
 */

#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/ParametrizedEngine/interface/ParametrizedMagneticFieldFactory.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

#include "CondFormats/RunInfo/interface/RunInfo.h"
#include "CondFormats/DataRecord/interface/RunSummaryRcd.h"

#include <string>

using namespace std;
using namespace edm;

namespace magneticfield {
  class AutoParametrizedMagneticFieldProducer : public edm::ESProducer {
  public:
    AutoParametrizedMagneticFieldProducer(const edm::ParameterSet&);
    ~AutoParametrizedMagneticFieldProducer() override {}

    std::unique_ptr<MagneticField> produce(const IdealMagneticFieldRecord&);

    int closerNominaCurrent(float current) const;
    const std::string version_;
    const int currentOverride_;
    const std::array<int, 7> nominalCurrents_;
    //  std::vector<std::string> nominalLabels_;
    edm::ESGetToken<RunInfo, RunInfoRcd> runInfoToken_;
  };
}  // namespace magneticfield

using namespace magneticfield;

AutoParametrizedMagneticFieldProducer::AutoParametrizedMagneticFieldProducer(const edm::ParameterSet& iConfig)
    : version_{iConfig.getParameter<string>("version")},
      currentOverride_{iConfig.getParameter<int>("valueOverride")},
      nominalCurrents_{{-1, 0, 9558, 14416, 16819, 18268, 19262}}
//  nominalLabels_{["3.8T","0T","2T", "3T", "3.5T", "3.8T", "4T"}}
{
  auto cc = setWhatProduced(this, iConfig.getUntrackedParameter<std::string>("label", ""));
  if (currentOverride_ < 0) {
    runInfoToken_ = cc.consumes();
  }
}

std::unique_ptr<MagneticField> AutoParametrizedMagneticFieldProducer::produce(const IdealMagneticFieldRecord& iRecord) {
  // Get value of the current from condition DB
  float current = currentOverride_;
  string message;
  if (current < 0) {
    current = iRecord.get(runInfoToken_).m_avg_current;
    message = " (from RunInfo DB)";
  } else {
    message = " (from valueOverride card)";
  }
  float cnc = closerNominaCurrent(current);

  edm::LogInfo("MagneticField") << "Current: " << current << message << "; using map for: " << cnc;

  vector<double> parameters;

  auto version = version_;
  if (cnc == 0) {
    version = "Uniform";
    parameters.push_back(0);
  } else if (version == "Parabolic") {
    parameters.push_back(3.8114);        //c1
    parameters.push_back(-3.94991e-06);  //b0
    parameters.push_back(7.53701e-06);   //b1
    parameters.push_back(2.43878e-11);   //a
    if (cnc !=
        18268) {  // Linear scaling for B!= 3.8T; note that just c1, b0 and b1 have to be scaled to get linear scaling
      double scale = double(cnc) / double(18268);
      parameters[0] *= scale;
      parameters[1] *= scale;
      parameters[2] *= scale;
    }
  } else {
    //Other parametrizations are not relevant here and not supported
    throw cms::Exception("InvalidParameter") << "version " << version << " is not supported";
  }

  return ParametrizedMagneticFieldFactory::get(version, parameters);
}

int AutoParametrizedMagneticFieldProducer::closerNominaCurrent(float current) const {
  int i = 0;
  for (; i < (int)nominalCurrents_.size() - 1; i++) {
    if (2 * current < nominalCurrents_[i] + nominalCurrents_[i + 1])
      return nominalCurrents_[i];
  }
  return nominalCurrents_[i];
}

#include "FWCore/Framework/interface/ModuleFactory.h"
DEFINE_FWK_EVENTSETUP_MODULE(AutoParametrizedMagneticFieldProducer);
