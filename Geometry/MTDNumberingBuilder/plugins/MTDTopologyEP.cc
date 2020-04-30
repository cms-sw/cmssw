#include "FWCore/Utilities/interface/Exception.h"
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
//#define EDM_ML_DEBUG

class MTDTopologyEP : public edm::ESProducer {
public:
  MTDTopologyEP(const edm::ParameterSet&);

  using ReturnType = std::unique_ptr<MTDTopology>;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  ReturnType produce(const MTDTopologyRcd&);

private:
  void fillParameters(const PMTDParameters&, int&, MTDTopology::BTLValues&, MTDTopology::ETLValues&);

  const edm::ESGetToken<PMTDParameters, PMTDParametersRcd> token_;
};

MTDTopologyEP::MTDTopologyEP(const edm::ParameterSet& conf)
    : token_{setWhatProduced(this).consumesFrom<PMTDParameters, PMTDParametersRcd>(edm::ESInputTag())} {
  edm::LogInfo("MTD") << "MTDTopologyEP::MTDTopologyEP";
}

void MTDTopologyEP::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription ttc;
  descriptions.add("mtdTopology", ttc);
}

MTDTopologyEP::ReturnType MTDTopologyEP::produce(const MTDTopologyRcd& iRecord) {
  edm::LogInfo("MTDTopologyEP") << "MTDTopologyEP::produce(const MTDTopologyRcd& iRecord)";

  int mtdTopologyMode;
  MTDTopology::BTLValues btlVals;
  MTDTopology::ETLValues etlVals;

  fillParameters(iRecord.get(token_), mtdTopologyMode, btlVals, etlVals);

  return std::make_unique<MTDTopology>(mtdTopologyMode, btlVals, etlVals);
}

void MTDTopologyEP::fillParameters(const PMTDParameters& ptp,
                                   int& mtdTopologyMode,
                                   MTDTopology::BTLValues& btlVals,
                                   MTDTopology::ETLValues& etlVals) {
  mtdTopologyMode = ptp.topologyMode_;

  btlVals.sideStartBit_ = ptp.vitems_[0].vpars_[0];    // 16
  btlVals.layerStartBit_ = ptp.vitems_[0].vpars_[1];   // 16
  btlVals.trayStartBit_ = ptp.vitems_[0].vpars_[2];    // 8
  btlVals.moduleStartBit_ = ptp.vitems_[0].vpars_[3];  // 2
  btlVals.sideMask_ = ptp.vitems_[0].vpars_[4];        // 0xF
  btlVals.layerMask_ = ptp.vitems_[0].vpars_[5];       // 0xF
  btlVals.trayMask_ = ptp.vitems_[0].vpars_[6];        // 0xFF
  btlVals.moduleMask_ = ptp.vitems_[0].vpars_[7];      // 0x3F

  etlVals.sideStartBit_ = ptp.vitems_[1].vpars_[0];
  etlVals.layerStartBit_ = ptp.vitems_[1].vpars_[1];
  etlVals.ringStartBit_ = ptp.vitems_[1].vpars_[2];
  etlVals.moduleStartBit_ = ptp.vitems_[1].vpars_[3];
  etlVals.sideMask_ = ptp.vitems_[1].vpars_[4];
  etlVals.layerMask_ = ptp.vitems_[1].vpars_[5];
  etlVals.ringMask_ = ptp.vitems_[1].vpars_[6];
  etlVals.moduleMask_ = ptp.vitems_[1].vpars_[7];

#ifdef EDM_ML_DEBUG

  edm::LogInfo("MTDTopologyEP") << "BTL values = " << btlVals.sideStartBit_ << " " << btlVals.layerStartBit_ << " "
                                << btlVals.trayStartBit_ << " " << btlVals.moduleStartBit_ << " " << std::hex
                                << btlVals.sideMask_ << " " << std::hex << btlVals.layerMask_ << " " << std::hex
                                << btlVals.trayMask_ << " " << std::hex << btlVals.moduleMask_ << " ";
  edm::LogInfo("MTDTopologyEP") << "ETL values = " << etlVals.sideStartBit_ << " " << etlVals.layerStartBit_ << " "
                                << etlVals.ringStartBit_ << " " << etlVals.moduleStartBit_ << " " << std::hex
                                << etlVals.sideMask_ << " " << std::hex << etlVals.layerMask_ << " " << std::hex
                                << etlVals.ringMask_ << " " << std::hex << etlVals.moduleMask_ << " ";

#endif
}

DEFINE_FWK_EVENTSETUP_MODULE(MTDTopologyEP);
