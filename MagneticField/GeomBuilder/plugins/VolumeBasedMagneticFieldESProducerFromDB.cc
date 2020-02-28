/** \class VolumeBasedMagneticFieldESProducerFromDB
 *
 *  Producer for the VolumeBasedMagneticField, taking all inputs (geometry, etc) from DB
 *
 */

#include "FWCore/Framework/interface/EventSetupRecordIntervalFinder.h"
#include "FWCore/Framework/interface/ESProducer.h"

#include "CondFormats/RunInfo/interface/RunInfo.h"
#include "CondFormats/DataRecord/interface/RunSummaryRcd.h"

#include "MagneticField/Engine/interface/MagneticField.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "MagneticField/VolumeBasedEngine/interface/VolumeBasedMagneticField.h"
#include "MagneticField/ParametrizedEngine/interface/ParametrizedMagneticFieldFactory.h"

#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

#include "FWCore/Framework/interface/ESTransientHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/ESProductTag.h"

#include "FWCore/Framework/interface/ModuleFactory.h"
#include "MagneticField/GeomBuilder/src/MagGeoBuilderFromDDD.h"

#include "DetectorDescription/Core/interface/DDCompactView.h"
#include "DetectorDescription/Core/interface/DDRoot.h"
#include "DetectorDescription/Parser/interface/DDLParser.h"
#include "CondFormats/Common/interface/FileBlob.h"
#include "CondFormats/DataRecord/interface/MFGeometryFileRcd.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "CondFormats/MFObjects/interface/MagFieldConfig.h"
#include "CondFormats/DataRecord/interface/MagFieldConfigRcd.h"

#include <string>
#include <vector>
#include <iostream>
#include <memory>

using namespace std;
using namespace magneticfield;
using namespace edm;

namespace magneticfield {
  class VolumeBasedMagneticFieldESProducerFromDB : public edm::ESProducer {
  public:
    VolumeBasedMagneticFieldESProducerFromDB(const edm::ParameterSet& iConfig);
    // forbid copy ctor and assignment op.
    VolumeBasedMagneticFieldESProducerFromDB(const VolumeBasedMagneticFieldESProducerFromDB&) = delete;
    const VolumeBasedMagneticFieldESProducerFromDB& operator=(const VolumeBasedMagneticFieldESProducerFromDB&) = delete;

    std::shared_ptr<MagFieldConfig const> chooseConfigViaParameter(const IdealMagneticFieldRecord& iRecord);
    std::shared_ptr<MagFieldConfig const> chooseConfigAtRuntime(const IdealMagneticFieldRecord& iRecord);

    std::unique_ptr<MagneticField> produce(const IdealMagneticFieldRecord& iRecord);

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  private:
    static std::string_view closerNominalLabel(float current);

    edm::ESGetToken<MagFieldConfig, MagFieldConfigRcd> mayGetConfigToken_;
    edm::ESGetToken<MagFieldConfig, MagFieldConfigRcd> knownFromParamConfigToken_;

    //NOTE: change of record since this MagFieldConfig was chosen based on data
    // from the record RunInfoRcd so therefore has a dependency upon that record
    edm::ESGetToken<MagFieldConfig, IdealMagneticFieldRecord> chosenConfigToken_;

    edm::ESGetToken<FileBlob, MFGeometryFileRcd> mayConsumeBlobToken_;
    const bool debug_;
  };
}  // namespace magneticfield

VolumeBasedMagneticFieldESProducerFromDB::VolumeBasedMagneticFieldESProducerFromDB(const edm::ParameterSet& iConfig)
    : debug_(iConfig.getUntrackedParameter<bool>("debugBuilder")) {
  std::string const myConfigLabel = "VBMFESChoice";

  //Based on configuration, pick algorithm to produce the proper MagFieldConfig with a specific label
  const int current = iConfig.getParameter<int>("valueOverride");
  if (current < 0) {
    //We do not know what to get until we first read RunInfo
    setWhatProduced(
        this, &VolumeBasedMagneticFieldESProducerFromDB::chooseConfigAtRuntime, edm::es::Label(myConfigLabel))
        .setMayConsume(
            mayGetConfigToken_,
            [](auto const& iGet, edm::ESTransientHandle<RunInfo> iHandle) {
              auto const label = closerNominalLabel(iHandle->m_avg_current);
              edm::LogInfo("MagneticFieldDB") << "Current :" << iHandle->m_avg_current
                                              << " (from RunInfo DB); using map configuration with label: " << label;
              return iGet("", label);
            },
            edm::ESProductTag<RunInfo, RunInfoRcd>("", ""));

  } else {
    //we know exactly what we are going to get
    auto const label = closerNominalLabel(current);
    edm::LogInfo("MagneticFieldDB") << "Current :" << current
                                    << " (from valueOverride card); using map configuration with label: " << label;
    setWhatProduced(
        this, &VolumeBasedMagneticFieldESProducerFromDB::chooseConfigViaParameter, edm::es::Label(myConfigLabel))
        .setConsumes(knownFromParamConfigToken_, edm::ESInputTag(""s, std::string(label)));
  }

  auto const label = iConfig.getUntrackedParameter<std::string>("label");
  auto const myConfigTag = edm::ESInputTag(iConfig.getParameter<std::string>("@module_label"), myConfigLabel);

  //We use the MagFieldConfig created above to decide which FileBlob to use
  setWhatProduced(this, label)
      .setMayConsume(
          mayConsumeBlobToken_,
          [](auto const& iGet, edm::ESTransientHandle<MagFieldConfig> iConfig) {
            if (iConfig->version == "parametrizedMagneticField") {
              return iGet.nothing();
            }
            return iGet("", std::to_string(iConfig->geometryVersion));
          },
          edm::ESProductTag<MagFieldConfig, IdealMagneticFieldRecord>(myConfigTag))
      .setConsumes(chosenConfigToken_, myConfigTag);  //Use same tag as the choice
}

std::shared_ptr<MagFieldConfig const> VolumeBasedMagneticFieldESProducerFromDB::chooseConfigAtRuntime(
    IdealMagneticFieldRecord const& iRcd) {
  edm::ESHandle<MagFieldConfig> config = iRcd.getHandle(mayGetConfigToken_);

  //just forward what we just got but do not take ownership
  return std::shared_ptr<MagFieldConfig const>(config.product(), [](auto*) {});
}

std::shared_ptr<MagFieldConfig const> VolumeBasedMagneticFieldESProducerFromDB::chooseConfigViaParameter(
    const IdealMagneticFieldRecord& iRecord) {
  auto config = iRecord.getHandle(knownFromParamConfigToken_);

  //just forward what we just got but do not take ownership
  return std::shared_ptr<MagFieldConfig const>(config.product(), [](auto*) {});
}

// ------------ method called to produce the data  ------------
std::unique_ptr<MagneticField> VolumeBasedMagneticFieldESProducerFromDB::produce(
    const IdealMagneticFieldRecord& iRecord) {
  auto const& conf = iRecord.getTransientHandle(chosenConfigToken_);

  std::unique_ptr<MagneticField> paramField =
      ParametrizedMagneticFieldFactory::get(conf->slaveFieldVersion, conf->slaveFieldParameters);

  edm::LogInfo("MagneticFieldDB") << "Version: " << conf->version << " geometryVersion: " << conf->geometryVersion
                                  << " slaveFieldVersion: " << conf->slaveFieldVersion;

  if (conf->version == "parametrizedMagneticField") {
    // The map consist of only the parametrization in this case
    return paramField;
  }

  // Full VolumeBased map + parametrization
  MagGeoBuilderFromDDD builder(conf->version, conf->geometryVersion, debug_);

  // Set scaling factors
  if (!conf->keys.empty()) {
    builder.setScaling(conf->keys, conf->values);
  }

  // Set specification for the grid tables to be used.
  if (!conf->gridFiles.empty()) {
    builder.setGridFiles(conf->gridFiles);
  }

  // Build the geometry (DDDCompactView) from the DB blob
  // (code taken from GeometryReaders/XMLIdealGeometryESSource/src/XMLIdealMagneticFieldGeometryESProducer.cc)

  auto const& blob = iRecord.getTransientHandle(mayConsumeBlobToken_);

  DDCompactView cpv{DDName("cmsMagneticField:MAGF")};
  DDLParser parser(cpv);
  parser.getDDLSAX2FileHandler()->setUserNS(true);
  parser.clearFiles();
  std::unique_ptr<std::vector<unsigned char> > tb = blob->getUncompressedBlob();
  parser.parse(*tb, tb->size());
  cpv.lockdown();

  builder.build(cpv);

  // Build the VB map. Ownership of the parametrization is transferred to it
  return std::make_unique<VolumeBasedMagneticField>(conf->geometryVersion,
                                                    builder.barrelLayers(),
                                                    builder.endcapSectors(),
                                                    builder.barrelVolumes(),
                                                    builder.endcapVolumes(),
                                                    builder.maxR(),
                                                    builder.maxZ(),
                                                    paramField.release(),
                                                    true);
}

std::string_view VolumeBasedMagneticFieldESProducerFromDB::closerNominalLabel(float current) {
  constexpr std::array<int, 7> nominalCurrents = {{-1, 0, 9558, 14416, 16819, 18268, 19262}};
  constexpr std::array<std::string_view, 7> nominalLabels = {{"3.8T", "0T", "2T", "3T", "3.5T", "3.8T", "4T"}};

  int i = 0;
  for (; i < (int)nominalLabels.size() - 1; i++) {
    if (2 * current < nominalCurrents[i] + nominalCurrents[i + 1])
      return nominalLabels[i];
  }
  return nominalLabels[i];
}

void VolumeBasedMagneticFieldESProducerFromDB::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.addUntracked<bool>("debugBuilder", false);
  desc.add<int>("valueOverride", -1)->setComment("Force value of current (in A); take the value from DB if < 0.");
  desc.addUntracked<std::string>("label", "");

  descriptions.addDefault(desc);
}

DEFINE_FWK_EVENTSETUP_MODULE(VolumeBasedMagneticFieldESProducerFromDB);
