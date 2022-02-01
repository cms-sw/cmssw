// -*- C++ -*-
//
// Package:    SiPixelFakeLorentzAngleESSource
// Class:      SiPixelFakeLorentzAngleESSource
//
/**\class SiPixelFakeLorentzAngleESSource SiPixelFakeLorentzAngleESSource.h CalibTracker/SiPixelESProducer/src/SiPixelFakeLorentzAngleESSource.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Lotte Wilke
//         Created:  Jan 31 2008
//
//

// user include files

#include "CalibTracker/SiPixelESProducers/interface/SiPixelDetInfoFileReader.h"
#include "CalibTracker/SiPixelESProducers/interface/SiPixelFakeLorentzAngleESSource.h"
#include "CalibTracker/StandaloneTrackerTopology/interface/StandaloneTrackerTopology.h"
#include "DataFormats/TrackerCommon/interface/PixelBarrelName.h"
#include "DataFormats/TrackerCommon/interface/PixelEndcapName.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"

//
// constructors and destructor
//
SiPixelFakeLorentzAngleESSource::SiPixelFakeLorentzAngleESSource(const edm::ParameterSet& conf_)
    : fp_(conf_.getParameter<edm::FileInPath>("file")),
      t_topo_fp_(conf_.getParameter<edm::FileInPath>("topologyInput")),
      myLabel_(conf_.getParameter<std::string>("appendToDataLabel")),
      BPixParameters_(conf_.getParameter<Parameters>("BPixParameters")),
      FPixParameters_(conf_.getParameter<Parameters>("FPixParameters")),
      ModuleParameters_(conf_.getParameter<Parameters>("ModuleParameters")),
      bPixLorentzAnglePerTesla_((float)conf_.getUntrackedParameter<double>("bPixLorentzAnglePerTesla", -9999.)),
      fPixLorentzAnglePerTesla_((float)conf_.getUntrackedParameter<double>("fPixLorentzAnglePerTesla", -9999.)) {
  edm::LogInfo("SiPixelFakeLorentzAngleESSource::SiPixelFakeLorentzAngleESSource");
  // the following line is needed to tell the framework what data is being produced
  setWhatProduced(this);
  findingRecord<SiPixelLorentzAngleRcd>();
}

std::unique_ptr<SiPixelLorentzAngle> SiPixelFakeLorentzAngleESSource::produce(const SiPixelLorentzAngleRcd&) {
  using namespace edm::es;
  unsigned int nmodules = 0;
  SiPixelLorentzAngle* obj = new SiPixelLorentzAngle();
  SiPixelDetInfoFileReader reader(fp_.fullPath());
  const std::vector<uint32_t>& DetIds = reader.getAllDetIds();

  TrackerTopology tTopo = StandaloneTrackerTopology::fromTrackerParametersXMLFile(t_topo_fp_.fullPath());

  // Loop over detectors
  for (const auto& detit : DetIds) {
    nmodules++;
    const DetId detid(detit);
    auto rawId = detid.rawId();
    int found = 0;
    int side = tTopo.side(detid);  // 1:-z 2:+z for fpix, for bpix gives 0

    // fill bpix values for LA
    if (detid.subdetId() == static_cast<int>(PixelSubdetector::PixelBarrel)) {
      int layer = tTopo.pxbLayer(detid);
      // Barrel ladder id 1-20,32,44.
      int ladder = tTopo.pxbLadder(detid);
      // Barrel Z-index=1,8
      int module = tTopo.pxbModule(detid);
      if (module < 5) {
        side = 1;
      } else {
        side = 2;
      }

      LogDebug("SiPixelFakeLorentzAngleESSource") << " pixel barrel:"
                                                  << " layer=" << layer << " ladder=" << ladder << " module=" << module
                                                  << "  rawId=" << rawId << " side " << side;

      // use a commmon value (e.g. for MC)
      if (bPixLorentzAnglePerTesla_ != -9999.) {  // use common value for all
        edm::LogInfo("SiPixelFakeLorentzAngleESSource")
            << " LA = " << bPixLorentzAnglePerTesla_ << " common for all bpix" << std::endl;
        if (!obj->putLorentzAngle(detid.rawId(), bPixLorentzAnglePerTesla_))
          edm::LogError("SiPixelFakeLorentzAngleESSource")
              << "ERROR!: detid " << rawId << " already exists" << std::endl;
      } else {
        //first individuals are put
        for (const auto& it : ModuleParameters_) {
          if (it.getParameter<unsigned int>("rawid") == detid.rawId()) {
            float lorentzangle = (float)it.getParameter<double>("angle");
            if (!found) {
              obj->putLorentzAngle(detid.rawId(), lorentzangle);
              edm::LogInfo("SiPixelFakeLorentzAngleESSource")
                  << " LA= " << lorentzangle << " individual value " << detid.rawId() << std::endl;
              found = 1;
            } else
              edm::LogError("SiPixelFakeLorentzAngleESSource") << "ERROR!: detid already exists" << std::endl;
          }
        }

        //modules already put are automatically skipped
        for (const auto& it : BPixParameters_) {
          if (it.exists("layer"))
            if (it.getParameter<int>("layer") != layer)
              continue;
          if (it.exists("ladder"))
            if (it.getParameter<int>("ladder") != ladder)
              continue;
          if (it.exists("module"))
            if (it.getParameter<int>("module") != module)
              continue;
          if (it.exists("side"))
            if (it.getParameter<int>("side") != side)
              continue;
          if (!found) {
            float lorentzangle = (float)it.getParameter<double>("angle");
            obj->putLorentzAngle(detid.rawId(), lorentzangle);
            edm::LogInfo("SiPixelFakeLorentzAngleESSource") << " LA= " << lorentzangle << std::endl;
            found = 2;
          } else if (found == 1) {
            edm::LogWarning("SiPixelFakeLorentzAngleESSource")
                << "detid already given in ModuleParameters, skipping ..." << std::endl;
          } else
            edm::LogError("SiPixelFakeLorentzAngleESSource")
                << " ERROR!: detid " << rawId << " already exists" << std::endl;
        }
      }
    } else if (detid.subdetId() == static_cast<int>(PixelSubdetector::PixelEndcap)) {
      // fill fpix values for LA (for phase2 fpix & epix)

      // For fpix we also need to find the ring number which is not available from topology
      // Convert to online
      PixelEndcapName pen(detid, &tTopo, true);  // use det-id phaseq

      //PixelEndcapName::HalfCylinder sh = pen.halfCylinder(); //enum
      //string nameF = pen.name();
      //int plaquetteName = pen.plaquetteName();
      int disk = pen.diskName();
      int blade = pen.bladeName();
      int panel = pen.pannelName();
      int ring = pen.ringName();

      LogDebug("SiPixelFakeLorentzAngleESSource") << " pixel endcap:"
                                                  << " side=" << side << " disk=" << disk << " blade =" << blade
                                                  << " pannel=" << panel << " ring=" << ring << "  rawId=" << rawId;

      // use a commmon value (e.g. for MC)
      if (fPixLorentzAnglePerTesla_ != -9999.) {  // use common value for all
        edm::LogInfo("SiPixelFakeLorentzAngleESSource")
            << " LA = " << fPixLorentzAnglePerTesla_ << " common for all fpix" << std::endl;
        if (!obj->putLorentzAngle(detid.rawId(), fPixLorentzAnglePerTesla_))
          edm::LogError("SiPixelFakeLorentzAngleESSource") << " ERROR! detid already exists" << std::endl;
      } else {
        //first individuals are put
        for (const auto& it : ModuleParameters_) {
          if (it.getParameter<unsigned int>("rawid") == detid.rawId()) {
            float lorentzangle = (float)it.getParameter<double>("angle");
            if (!found) {
              obj->putLorentzAngle(detid.rawId(), lorentzangle);
              edm::LogInfo("SiPixelFakeLorentzAngleESSource")
                  << " LA= " << lorentzangle << " individual value " << detid.rawId() << std::endl;
              found = 1;
            } else
              edm::LogError("SiPixelFakeLorentzAngleESSource")
                  << "ERROR!: detid " << rawId << " already exists" << std::endl;
          }  // if
        }    // for

        //modules already put are automatically skipped
        for (const auto& it : FPixParameters_) {
          if (it.exists("side"))
            if (it.getParameter<int>("side") != side)
              continue;
          if (it.exists("disk"))
            if (it.getParameter<int>("disk") != disk)
              continue;
          if (it.exists("ring"))
            if (it.getParameter<int>("ring") != ring)
              continue;
          if (it.exists("blade"))
            if (it.getParameter<int>("blade") != blade)
              continue;
          if (it.exists("panel"))
            if (it.getParameter<int>("panel") != panel)
              continue;
          if (it.exists("HVgroup"))
            if (it.getParameter<int>("HVgroup") != HVgroup(panel, ring))
              continue;
          if (!found) {
            float lorentzangle = (float)it.getParameter<double>("angle");
            obj->putLorentzAngle(detid.rawId(), lorentzangle);
            edm::LogInfo("SiPixelFakeLorentzAngleESSource") << " LA= " << lorentzangle << std::endl;
            found = 2;
          } else if (found == 1) {
            edm::LogWarning("SiPixelFakeLorentzAngleESSource")
                << " detid " << rawId << " already given in ModuleParameters, skipping ..." << std::endl;
          } else
            edm::LogError("SiPixelFakeLorentzAngleESSource")
                << "ERROR!: detid" << rawId << "already exists" << std::endl;
        }  // for
      }    // if
    }      // bpix/fpix
  }        // iterate on detids

  edm::LogInfo("SiPixelFakeLorentzAngleESSource") << "Modules = " << nmodules << std::endl;

  return std::unique_ptr<SiPixelLorentzAngle>(obj);
}

void SiPixelFakeLorentzAngleESSource::setIntervalFor(const edm::eventsetup::EventSetupRecordKey&,
                                                     const edm::IOVSyncValue& iosv,
                                                     edm::ValidityInterval& oValidity) {
  edm::ValidityInterval infinity(iosv.beginOfTime(), iosv.endOfTime());
  oValidity = infinity;
}

int SiPixelFakeLorentzAngleESSource::HVgroup(int panel, int module) {
  if (1 == panel && (1 == module || 2 == module)) {
    return 1;
  } else if (1 == panel && (3 == module || 4 == module)) {
    return 2;
  } else if (2 == panel && 1 == module) {
    return 1;
  } else if (2 == panel && (2 == module || 3 == module)) {
    return 2;
  } else {
    edm::LogError("SiPixelFakeLorentzAngleESSource")
        << " ERROR! in SiPixelFakeLorentzAngleESSource::HVgroup(...), panel = " << panel << ", module = " << module
        << std::endl;
    return 0;
  }
}

void SiPixelFakeLorentzAngleESSource::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.setComment("ESSource to supply per-module SiPixelLorentzAngle payloads in the EventSetup");

  desc.add<edm::FileInPath>(
          "file", edm::FileInPath("SLHCUpgradeSimulations/Geometry/data/PhaseI/PixelSkimmedGeometry_phase1.txt"))
      ->setComment("Tracker skimmed geometry");
  desc.add<edm::FileInPath>("topologyInput",
                            edm::FileInPath("Geometry/TrackerCommonData/data/PhaseI/trackerParameters.xml"))
      ->setComment("Tracker Topology");

  desc.add<std::string>("appendToDataLabel", "")->setComment("label to which write the data");
  desc.addUntracked<double>("bPixLorentzAnglePerTesla", -9999.)->setComment("LA value for all BPix");
  desc.addUntracked<double>("fPixLorentzAnglePerTesla", -9999.)->setComment("LA value for all FPix");

  edm::ParameterSetDescription desc_BPixParameters;
  desc_BPixParameters.addOptional<int>("layer");
  desc_BPixParameters.addOptional<int>("ladder");
  desc_BPixParameters.addOptional<int>("module");
  desc_BPixParameters.addOptional<int>("side");
  desc_BPixParameters.add<double>("angle");
  std::vector<edm::ParameterSet> default_BPixParametersToAdd;
  desc.addVPSet("BPixParameters", desc_BPixParameters, default_BPixParametersToAdd)
      ->setComment("LA values for given BPix regions");

  edm::ParameterSetDescription desc_FPixParameters;
  desc_FPixParameters.addOptional<int>("side");
  desc_FPixParameters.addOptional<int>("disk");
  desc_FPixParameters.addOptional<int>("ring");
  desc_FPixParameters.addOptional<int>("blade");
  desc_FPixParameters.addOptional<int>("panel");
  desc_FPixParameters.addOptional<int>("HVgroup");
  desc_FPixParameters.add<double>("angle");
  std::vector<edm::ParameterSet> default_FPixParametersToAdd;
  desc.addVPSet("FPixParameters", desc_FPixParameters, default_FPixParametersToAdd)
      ->setComment("LA values for given FPix regions");

  edm::ParameterSetDescription desc_ModuleParameters;
  desc_ModuleParameters.add<unsigned int>("rawid");
  desc_ModuleParameters.add<double>("angle");
  std::vector<edm::ParameterSet> default_ModuleParametersToAdd;
  desc.addVPSet("ModuleParameters", desc_ModuleParameters, default_ModuleParametersToAdd)
      ->setComment("LA values for given modules");

  descriptions.addWithDefaultLabel(desc);
}
