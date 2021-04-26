//#define EDM_ML_DEBUG

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "Geometry/MTDNumberingBuilder/interface/MTDTopology.h"
#include "Geometry/Records/interface/MTDTopologyRcd.h"
#include "CondFormats/GeometryObjects/interface/PMTDParameters.h"
#include "Geometry/Records/interface/PMTDParametersRcd.h"

#include <memory>

class MTDTopologyEP : public edm::ESProducer {
public:
  MTDTopologyEP(const edm::ParameterSet&);

  using ReturnType = std::unique_ptr<MTDTopology>;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  ReturnType produce(const MTDTopologyRcd&);

private:
  void fillParameters(const PMTDParameters&, int& mtdTopologyMode, MTDTopology::ETLValues&);

  const edm::ESGetToken<PMTDParameters, PMTDParametersRcd> token_;
};

MTDTopologyEP::MTDTopologyEP(const edm::ParameterSet& conf)
    : token_{setWhatProduced(this).consumesFrom<PMTDParameters, PMTDParametersRcd>(edm::ESInputTag())} {}

void MTDTopologyEP::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription ttc;
  descriptions.add("mtdTopology", ttc);
}

MTDTopologyEP::ReturnType MTDTopologyEP::produce(const MTDTopologyRcd& iRecord) {
  int mtdTopologyMode;
  MTDTopology::ETLValues etlVals;

  fillParameters(iRecord.get(token_), mtdTopologyMode, etlVals);

  return std::make_unique<MTDTopology>(mtdTopologyMode, etlVals);
}

void MTDTopologyEP::fillParameters(const PMTDParameters& ptp, int& mtdTopologyMode, MTDTopology::ETLValues& etlVals) {
  mtdTopologyMode = ptp.topologyMode_;

  // Check on the internal consistency of thr ETL layour information provided by parameters

  for (size_t it = 3; it <= 9; it++) {
    if (ptp.vitems_[it].vpars_.size() != ptp.vitems_[2].vpars_.size()) {
      throw cms::Exception("MTDTopologyEP") << "Inconsistent size of ETL structure arrays";
    }
  }

  MTDTopology::ETLfaceLayout tmpFace;

  // Front Face (0), starting with type Right (2)

  tmpFace.idDiscSide_ = 0;  // ETL front side
  tmpFace.idDetType1_ = 2;  // ETL module type right

  tmpFace.start_copy_1_ = ptp.vitems_[3].vpars_;  // start_copy_FR
  tmpFace.start_copy_2_ = ptp.vitems_[2].vpars_;  // start_copy_FL
  tmpFace.offset_1_ = ptp.vitems_[7].vpars_;      // offset_FR
  tmpFace.offset_2_ = ptp.vitems_[6].vpars_;      // offset_FL

  etlVals.emplace_back(tmpFace);

  // Back Face (1), starting with type Left (1)

  tmpFace.idDiscSide_ = 1;  // ETL back side
  tmpFace.idDetType1_ = 1;  // ETL module type left

  tmpFace.start_copy_1_ = ptp.vitems_[4].vpars_;  // start_copy_BL
  tmpFace.start_copy_2_ = ptp.vitems_[5].vpars_;  // start_copy_BR
  tmpFace.offset_1_ = ptp.vitems_[8].vpars_;      // offset_BL
  tmpFace.offset_2_ = ptp.vitems_[9].vpars_;      // offset_BR

  etlVals.emplace_back(tmpFace);

#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("MTDTopologyEP") << " Topology mode = " << mtdTopologyMode << "\n";
  auto print_array = [&](std::vector<int> vector) {
    std::stringstream ss;
    for (auto const& elem : vector) {
      ss << " " << elem;
    }
    ss << "\n";
    return ss.str();
  };

  for (auto const& ilay : etlVals) {
    edm::LogVerbatim("MTDTopologyEP") << " disc face = " << ilay.idDiscSide_ << " start det type = " << ilay.idDetType1_
                                      << "\n start_copy_1 = " << print_array(ilay.start_copy_1_)
                                      << "\n start_copy_2 = " << print_array(ilay.start_copy_2_)
                                      << "\n offset_1 = " << print_array(ilay.offset_1_)
                                      << "\n offset_2 = " << print_array(ilay.offset_2_);
  }

#endif
}

DEFINE_FWK_EVENTSETUP_MODULE(MTDTopologyEP);
