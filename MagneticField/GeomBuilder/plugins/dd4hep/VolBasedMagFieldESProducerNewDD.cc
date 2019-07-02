/** \class VolBasedMagFieldESProducerNewDD
 *
 *  Producer for the VolumeBasedMagneticField.
 *
 */

#include "FWCore/Framework/interface/ESTransientHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EventSetupRecordIntervalFinder.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/ModuleFactory.h"

#include "MagneticField/VolumeBasedEngine/interface/VolumeBasedMagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "MagneticField/Engine/interface/MagneticField.h"

#include "MagneticField/GeomBuilder/plugins/dd4hep/MagGeoBuilder.h"
#include "CondFormats/MFObjects/interface/MagFieldConfig.h"
#include "Geometry/Records/interface/GeometryFileRcd.h"
#include "DetectorDescription/DDCMS/interface/BenchmarkGrd.h"
#include "DetectorDescription/DDCMS/interface/DDCompactView.h"
#include "DetectorDescription/DDCMS/interface/DDDetector.h"

#include <iostream>
#include <string>
#include <vector>

using namespace cms;
using namespace std;
using namespace magneticfield;

namespace magneticfield {
  class VolBasedMagFieldESProducerNewDD : public edm::ESProducer {
  public:
    VolBasedMagFieldESProducerNewDD(const edm::ParameterSet& iConfig);

    // forbid copy ctor and assignment op.
    VolBasedMagFieldESProducerNewDD(const VolBasedMagFieldESProducerNewDD&) = delete;
    const VolBasedMagFieldESProducerNewDD& operator=(const VolBasedMagFieldESProducerNewDD&) = delete;

    std::unique_ptr<MagneticField> produce(const IdealMagneticFieldRecord& iRecord);

  private:
    edm::ParameterSet pset_;
    const bool debug_;
    const bool useParametrizedTrackerField_;
    const MagFieldConfig conf_;
    const std::string version_;
    edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> paramFieldToken_;
    edm::ESGetToken<DDCompactView, IdealMagneticFieldRecord> cpvToken_;
    const edm::ESInputTag tag_;
  };
}  // namespace magneticfield

VolBasedMagFieldESProducerNewDD::VolBasedMagFieldESProducerNewDD(const edm::ParameterSet& iConfig)
    : pset_{iConfig},
      debug_{iConfig.getUntrackedParameter<bool>("debugBuilder", false)},
      useParametrizedTrackerField_{iConfig.getParameter<bool>("useParametrizedTrackerField")},
      conf_{iConfig, debug_},
      version_{iConfig.getParameter<std::string>("version")},
      tag_{iConfig.getParameter<edm::ESInputTag>("DDDetector")} {
  // LogInfo used because LogDebug messages don't appear even when fully enabled.
  edm::LogInfo("VolBasedMagFieldESProducerNewDD") << "info:Constructing a VolBasedMagFieldESProducerNewDD" << endl;

  auto cc = setWhatProduced(this, iConfig.getUntrackedParameter<std::string>("label", ""));
  cc.setConsumes(cpvToken_, edm::ESInputTag{"", "magfield"});
  if (useParametrizedTrackerField_) {
    cc.setConsumes(paramFieldToken_, edm::ESInputTag{"", iConfig.getParameter<string>("paramLabel")});
  }
}

// ------------ method called to produce the data  ------------
std::unique_ptr<MagneticField> VolBasedMagFieldESProducerNewDD::produce(const IdealMagneticFieldRecord& iRecord) {
  if (debug_) {
    edm::LogInfo("VolBasedMagFieldESProducerNewDD") << "VolBasedMagFieldESProducerNewDD::produce() " << version_;
  }

  MagGeoBuilder builder(conf_.version, conf_.geometryVersion, debug_);

  // Set scaling factors
  if (!conf_.keys.empty()) {
    builder.setScaling(conf_.keys, conf_.values);
  }

  // Set specification for the grid tables to be used.
  if (!conf_.gridFiles.empty()) {
    builder.setGridFiles(conf_.gridFiles);
  }

  auto cpv = iRecord.getTransientHandle(cpvToken_);
  const DDCompactView* cpvPtr = cpv.product();
  const DDDetector* det = cpvPtr->detector();
  builder.build(det);
  edm::LogInfo("VolBasedMagFieldESProducerNewDD") << "produce() finished build";

  // Get slave field (from ES)
  const MagneticField* paramField = nullptr;
  if (useParametrizedTrackerField_) {
    edm::LogInfo("VolBasedMagFieldESProducerNewDD") << "Getting MF for parametrized field";
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

DEFINE_FWK_EVENTSETUP_MODULE(VolBasedMagFieldESProducerNewDD);
