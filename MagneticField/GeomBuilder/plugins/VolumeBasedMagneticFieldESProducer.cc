/** \class VolumeBasedMagneticFieldESProducer
 *
 *  Producer for the VolumeBasedMagneticField.
 *
 */

#include "FWCore/Framework/interface/ESTransientHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EventSetupRecordIntervalFinder.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/ModuleFactory.h"

#include "MagneticField/VolumeBasedEngine/interface/VolumeBasedMagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "MagneticField/Engine/interface/MagneticField.h"

#include "DetectorDescription/Core/interface/DDCompactView.h"
#include "MagneticField/GeomBuilder/src/MagGeoBuilderFromDDD.h"
#include "CondFormats/MFObjects/interface/MagFieldConfig.h"

#include <string>
#include <vector>
#include <iostream>

namespace magneticfield {
  class VolumeBasedMagneticFieldESProducer : public edm::ESProducer {
  public:
    VolumeBasedMagneticFieldESProducer(const edm::ParameterSet& iConfig);

    std::unique_ptr<MagneticField> produce(const IdealMagneticFieldRecord& iRecord);

    // forbid copy ctor and assignment op.
    VolumeBasedMagneticFieldESProducer(const VolumeBasedMagneticFieldESProducer&) = delete;
    const VolumeBasedMagneticFieldESProducer& operator=(const VolumeBasedMagneticFieldESProducer&) = delete;

  private:
    const bool debug_;
    const bool useParametrizedTrackerField_;
    const MagFieldConfig conf_;
    const std::string version_;
    edm::ESGetToken<DDCompactView, IdealMagneticFieldRecord> cpvToken_;
    edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> paramFieldToken_;
  };
}  // namespace magneticfield

using namespace std;
using namespace magneticfield;

VolumeBasedMagneticFieldESProducer::VolumeBasedMagneticFieldESProducer(const edm::ParameterSet& iConfig)
    : debug_{iConfig.getUntrackedParameter<bool>("debugBuilder", false)},
      useParametrizedTrackerField_{iConfig.getParameter<bool>("useParametrizedTrackerField")},
      conf_{iConfig, debug_},
      version_{iConfig.getParameter<std::string>("version")} {
  auto cc = setWhatProduced(this, iConfig.getUntrackedParameter<std::string>("label", ""));
  cpvToken_ = cc.consumes(edm::ESInputTag{"", "magfield"});
  if (useParametrizedTrackerField_) {
    paramFieldToken_ = cc.consumes(edm::ESInputTag{"", iConfig.getParameter<string>("paramLabel")});
  }
}

// ------------ method called to produce the data  ------------
std::unique_ptr<MagneticField> VolumeBasedMagneticFieldESProducer::produce(const IdealMagneticFieldRecord& iRecord) {
  LogTrace("MagGeoBuilder") << "VolumeBasedMagneticFieldESProducer::produce() " << version_;

  auto cpv = iRecord.getTransientHandle(cpvToken_);
  MagGeoBuilderFromDDD builder(conf_.version, conf_.geometryVersion, debug_);

  // Set scaling factors
  if (!conf_.keys.empty()) {
    builder.setScaling(conf_.keys, conf_.values);
  }

  // Set specification for the grid tables to be used.
  if (!conf_.gridFiles.empty()) {
    builder.setGridFiles(conf_.gridFiles);
  }

  builder.build(*cpv);

  // Get slave field (from ES)
  const MagneticField* paramField = nullptr;
  if (useParametrizedTrackerField_) {
    paramField = &iRecord.get(paramFieldToken_);
  }
  return std::make_unique<VolumeBasedMagneticField>(conf_.geometryVersion,
                                                    builder.barrelLayers(),
                                                    builder.endcapSectors(),
                                                    builder.barrelVolumes(),
                                                    builder.endcapVolumes(),
                                                    builder.maxR(),
                                                    builder.maxZ(),
                                                    paramField,
                                                    false);
}

DEFINE_FWK_EVENTSETUP_MODULE(VolumeBasedMagneticFieldESProducer);
