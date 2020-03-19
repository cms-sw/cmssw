/** \class DD4hep_VolumeBasedMagneticFieldESProducer
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

#include "MagneticField/GeomBuilder/src/DD4hep_MagGeoBuilder.h"
#include "CondFormats/MFObjects/interface/MagFieldConfig.h"
#include "DetectorDescription/DDCMS/interface/BenchmarkGrd.h"
#include "DetectorDescription/DDCMS/interface/DDCompactView.h"
#include "DetectorDescription/DDCMS/interface/DDDetector.h"

#include <string>

namespace magneticfield {

  class DD4hep_VolumeBasedMagneticFieldESProducer : public edm::ESProducer {
  public:
    DD4hep_VolumeBasedMagneticFieldESProducer(const edm::ParameterSet& iConfig);

    // forbid copy ctor and assignment op.
    DD4hep_VolumeBasedMagneticFieldESProducer(const DD4hep_VolumeBasedMagneticFieldESProducer&) = delete;
    const DD4hep_VolumeBasedMagneticFieldESProducer& operator=(const DD4hep_VolumeBasedMagneticFieldESProducer&) =
        delete;

    std::unique_ptr<MagneticField> produce(const IdealMagneticFieldRecord& iRecord);

  private:
    edm::ParameterSet pset_;
    const bool debug_;
    const bool useParametrizedTrackerField_;
    const MagFieldConfig conf_;
    const std::string version_;
    edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> paramFieldToken_;
    edm::ESGetToken<cms::DDCompactView, IdealMagneticFieldRecord> cpvToken_;
  };
}  // namespace magneticfield

using namespace magneticfield;

DD4hep_VolumeBasedMagneticFieldESProducer::DD4hep_VolumeBasedMagneticFieldESProducer(const edm::ParameterSet& iConfig)
    : pset_{iConfig},
      debug_{iConfig.getUntrackedParameter<bool>("debugBuilder", false)},
      useParametrizedTrackerField_{iConfig.getParameter<bool>("useParametrizedTrackerField")},
      conf_{iConfig, debug_},
      version_{iConfig.getParameter<std::string>("version")} {
  // LogVerbatim used because LogTrace messages don't seem to appear even when fully enabled.
  edm::LogVerbatim("DD4hep_VolumeBasedMagneticFieldESProducer")
      << "info:Constructing a DD4hep_VolumeBasedMagneticFieldESProducer";

  auto cc = setWhatProduced(this, iConfig.getUntrackedParameter<std::string>("label", ""));
  cc.setConsumes(cpvToken_, edm::ESInputTag{"", "magfield"});
  if (useParametrizedTrackerField_) {
    cc.setConsumes(paramFieldToken_, edm::ESInputTag{"", iConfig.getParameter<std::string>("paramLabel")});
  }
}

// ------------ method called to produce the data  ------------
std::unique_ptr<MagneticField> DD4hep_VolumeBasedMagneticFieldESProducer::produce(
    const IdealMagneticFieldRecord& iRecord) {
  if (debug_) {
    edm::LogVerbatim("DD4hep_VolumeBasedMagneticFieldESProducer")
        << "DD4hep_VolumeBasedMagneticFieldESProducer::produce() " << version_;
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
  const cms::DDCompactView* cpvPtr = cpv.product();
  const cms::DDDetector* det = cpvPtr->detector();
  builder.build(det);
  edm::LogVerbatim("DD4hep_VolumeBasedMagneticFieldESProducer") << "produce() finished build";

  // Get slave field (from ES)
  const MagneticField* paramField = nullptr;
  if (useParametrizedTrackerField_) {
    edm::LogVerbatim("DD4hep_VolumeBasedMagneticFieldESProducer") << "Getting MF for parametrized field";
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

DEFINE_FWK_EVENTSETUP_MODULE(DD4hep_VolumeBasedMagneticFieldESProducer);
