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

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  private:
    edm::ParameterSet pset_;
    const bool debug_;
    const bool useParametrizedTrackerField_;
    const bool useMergeFileIfAvailable_;
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
      useMergeFileIfAvailable_{iConfig.getParameter<bool>("useMergeFileIfAvailable")},
      conf_{iConfig, debug_},
      version_{iConfig.getParameter<std::string>("version")} {
  LogTrace("MagGeoBuilder") << "info:Constructing a DD4hep_VolumeBasedMagneticFieldESProducer";

  auto cc = setWhatProduced(this, iConfig.getUntrackedParameter<std::string>("label", ""));
  cpvToken_ = cc.consumes(edm::ESInputTag{"", "magfield"});
  if (useParametrizedTrackerField_) {
    paramFieldToken_ = cc.consumes(edm::ESInputTag{"", iConfig.getParameter<std::string>("paramLabel")});
  }
}

// ------------ method called to produce the data  ------------
std::unique_ptr<MagneticField> DD4hep_VolumeBasedMagneticFieldESProducer::produce(
    const IdealMagneticFieldRecord& iRecord) {
  if (debug_) {
    LogTrace("MagGeoBuilder") << "DD4hep_VolumeBasedMagneticFieldESProducer::produce() " << version_;
  }

  MagGeoBuilder builder(conf_.version, conf_.geometryVersion, debug_, useMergeFileIfAvailable_);

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
  LogTrace("MagGeoBuilder") << "produce() finished build";

  // Get slave field (from ES)
  const MagneticField* paramField = nullptr;
  if (useParametrizedTrackerField_) {
    LogTrace("MagGeoBuilder") << "Getting MF for parametrized field";
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

void DD4hep_VolumeBasedMagneticFieldESProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.addUntracked<bool>("debugBuilder", false);
  desc.add<bool>("useMergeFileIfAvailable", true);
  desc.add<bool>("useParametrizedTrackerField");
  desc.addUntracked<std::string>("label", "");
  desc.add<std::string>("version");
  desc.add<std::string>("paramLabel");

  //from MagFieldConfig
  desc.add<int>("geometryVersion");
  {
    edm::ParameterSetDescription sub;
    sub.add<std::string>("volumes");
    sub.add<std::string>("sectors");
    sub.add<int>("master");
    sub.add<std::string>("path");
    desc.addVPSet("gridFiles", sub);
  }
  desc.add<std::vector<int> >("scalingVolumes");
  desc.add<std::vector<double> >("scalingFactors");
  //default used to be compatible with older configurations
  desc.add<std::vector<double> >("paramData", std::vector<double>());

  descriptions.addDefault(desc);
}

DEFINE_FWK_EVENTSETUP_MODULE(DD4hep_VolumeBasedMagneticFieldESProducer);
