#include <memory>
#include <string>
#include <iostream>
#include <fstream>
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelLorentzAngle.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/MeasurementError.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/MeasurementPoint.h"
#include "DataFormats/GeometrySurface/interface/GloballyPositioned.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "DataFormats/TrackerCommon/interface/PixelBarrelName.h"
#include "DataFormats/TrackerCommon/interface/PixelEndcapName.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "Geometry/CommonTopologies/interface/PixelGeomDetUnit.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"

class SiPixelLorentzAngleDBLoader : public edm::one::EDAnalyzer<edm::one::SharedResources> {
public:
  explicit SiPixelLorentzAngleDBLoader(const edm::ParameterSet& conf);

  ~SiPixelLorentzAngleDBLoader() override = default;
  void analyze(const edm::Event& e, const edm::EventSetup& c) override;

private:
  typedef std::vector<edm::ParameterSet> Parameters;
  const edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> tkGeomToken_;
  const edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> tkTopoToken_;
  const std::string recordName_;
  const bool useFile_;
  const std::string fileName_;
  const Parameters BPixParameters_;
  const Parameters FPixParameters_;
  const Parameters ModuleParameters_;
  float bPixLorentzAnglePerTesla_;
  float fPixLorentzAnglePerTesla_;

  int HVgroup(int panel, int module);

  std::vector<std::pair<uint32_t, float> > detid_la;
};

using namespace std;
using namespace edm;

// Constructor
SiPixelLorentzAngleDBLoader::SiPixelLorentzAngleDBLoader(edm::ParameterSet const& conf)
    : tkGeomToken_(esConsumes()),
      tkTopoToken_(esConsumes()),
      recordName_(conf.getUntrackedParameter<std::string>("record", "SiPixelLorentzAngleRcd")),
      useFile_(conf.getParameter<bool>("useFile")),
      fileName_(conf.getParameter<string>("fileName")),
      BPixParameters_(conf.getUntrackedParameter<Parameters>("BPixParameters")),
      FPixParameters_(conf.getUntrackedParameter<Parameters>("FPixParameters")),
      ModuleParameters_(conf.getUntrackedParameter<Parameters>("ModuleParameters")) {
  bPixLorentzAnglePerTesla_ =
      static_cast<float>(conf.getUntrackedParameter<double>("bPixLorentzAnglePerTesla", -9999.));
  fPixLorentzAnglePerTesla_ =
      static_cast<float>(conf.getUntrackedParameter<double>("fPixLorentzAnglePerTesla", -9999.));
  usesResource(cond::service::PoolDBOutputService::kSharedResource);
}

void SiPixelLorentzAngleDBLoader::analyze(const edm::Event& e, const edm::EventSetup& es) {
  static constexpr int nModules_ = 4;
  SiPixelLorentzAngle LorentzAngle;

  // Retrieve tracker geometry from geometry
  const TrackerGeometry* pDD = &es.getData(tkGeomToken_);
  // Retrieve tracker topology from geometry
  const TrackerTopology* tTopo = &es.getData(tkTopoToken_);

  for (auto& unit : pDD->detUnits()) {
    if (auto pixelUnit = dynamic_cast<PixelGeomDetUnit const*>(unit)) {
      const DetId detid = pixelUnit->geographicalId();
      auto rawId = detid.rawId();
      int found = 0;
      int side = tTopo->side(detid);  // 1:-z 2:+z for fpix, for bpix gives 0

      // fill bpix values for LA
      if (detid.subdetId() == static_cast<int>(PixelSubdetector::PixelBarrel)) {
        int layer = tTopo->pxbLayer(detid);
        // Barrel ladder id 1-20,32,44.
        int ladder = tTopo->pxbLadder(detid);
        // Barrel Z-index=1,8
        int module = tTopo->pxbModule(detid);
        if (module < nModules_ + 1) {
          side = 1;
        } else {
          side = 2;
        }

        LogPrint("SiPixelLorentzAngleDBLoader") << " pixel barrel:"
                                                << " layer=" << layer << " ladder=" << ladder << " module=" << module
                                                << "  rawId=" << rawId << " side=" << side;

        // use a commmon value (e.g. for MC)
        if (bPixLorentzAnglePerTesla_ != -9999.) {  // use common value for all
          LogPrint("SiPixelLorentzAngleDBLoader")
              << " LA=" << bPixLorentzAnglePerTesla_ << " common for all bpix" << endl;
          if (!LorentzAngle.putLorentzAngle(detid.rawId(), bPixLorentzAnglePerTesla_)) {
            LogError("SiPixelLorentzAngleDBLoader") << "ERROR!: detid already exists";
          }
          // use an external file
        } else if (useFile_) {
          LogPrint("SiPixelLorentzAngleDBLoader") << "method for reading file not implemented yet";
          // use config file
        } else {
          // first individuals are put
          for (auto& moduleParam : ModuleParameters_) {
            if (moduleParam.getParameter<unsigned int>("rawid") == detid.rawId()) {
              float lorentzangle = static_cast<float>(moduleParam.getParameter<double>("angle"));
              if (!found) {
                LorentzAngle.putLorentzAngle(detid.rawId(), lorentzangle);
                LogPrint("SiPixelLorentzAngleDBLoader")
                    << "   >> LA=" << lorentzangle << " individual value " << detid.rawId();
                found = 1;
              } else {
                LogError("SiPixelLorentzAngleDBLoader") << "ERROR!: detid already exists";
              }
            }
          }  // end on loop for ModuleParameters_

          //modules already put are automatically skipped
          for (auto& bpixParam : BPixParameters_) {
            if (bpixParam.exists("layer")) {
              if (bpixParam.getParameter<int>("layer") != layer)
                continue;
              if (bpixParam.exists("ladder"))
                if (bpixParam.getParameter<int>("ladder") != ladder)
                  continue;
              if (bpixParam.exists("module"))
                if (bpixParam.getParameter<int>("module") != module)
                  continue;
              if (bpixParam.exists("side"))
                if (bpixParam.getParameter<int>("side") != side)
                  continue;
              if (!found) {
                float lorentzangle = static_cast<float>(bpixParam.getParameter<double>("angle"));
                LorentzAngle.putLorentzAngle(detid.rawId(), lorentzangle);
                LogPrint("SiPixelLorentzAngleDBLoader") << "   >> LA=" << lorentzangle;
                found = 2;
              } else if (found == 1) {
                LogPrint("SiPixelLorentzAngleDBLoader") << "The detid already given in ModuleParameters, skipping ...";
              } else
                LogError("SiPixelLorentzAngleDBLoader") << "ERROR!: detid already exists";
            }
          }
        }  // condition to read from config

        // fill fpix values for LA (for phase2 fpix & epix)
      } else if (detid.subdetId() == static_cast<int>(PixelSubdetector::PixelEndcap)) {
        // Convert to online
        PixelEndcapName pen(detid, tTopo, true);
        int disk = pen.diskName();
        int blade = pen.bladeName();
        int panel = pen.pannelName();
        int ring = pen.ringName();

        LogPrint("SiPixelLorentzAngleDBLoader") << " pixel endcap:"
                                                << " side=" << side << " disk=" << disk << " blade=" << blade
                                                << " pannel=" << panel << " ring=" << ring << "  rawId=" << rawId;

        // use a commmon value (e.g. for MC)
        if (fPixLorentzAnglePerTesla_ != -9999.) {  // use common value for all
          LogPrint("SiPixelLorentzAngleDBLoader") << " LA =" << fPixLorentzAnglePerTesla_ << " common for all FPix";
          if (!LorentzAngle.putLorentzAngle(detid.rawId(), fPixLorentzAnglePerTesla_)) {
            LogError("SiPixelLorentzAngleDBLoader") << "detid already exists";
          }

        } else if (useFile_) {
          LogPrint("SiPixelLorentzAngleDBLoader") << "method for reading file not implemented yet";

        } else {
          //first individuals are put
          for (auto& parameter : ModuleParameters_) {
            if (parameter.getParameter<unsigned int>("rawid") == detid.rawId()) {
              float lorentzangle = static_cast<float>(parameter.getParameter<double>("angle"));
              if (!found) {
                LorentzAngle.putLorentzAngle(detid.rawId(), lorentzangle);
                LogPrint("SiPixelLorentzAngleDBLoader")
                    << " LA=" << lorentzangle << " individual value " << detid.rawId();
                found = 1;
              } else
                LogError("SiPixelLorentzAngleDBLoader") << "ERROR!: detid already exists";
            }
          }  // end loop on ModuleParameters_

          // modules already put are automatically skipped
          for (auto& fpixParam : FPixParameters_) {
            if (fpixParam.exists("side"))
              if (fpixParam.getParameter<int>("side") != side)
                continue;
            if (fpixParam.exists("disk"))
              if (fpixParam.getParameter<int>("disk") != disk)
                continue;
            if (fpixParam.exists("ring"))
              if (fpixParam.getParameter<int>("ring") != ring)
                continue;
            if (fpixParam.exists("blade"))
              if (fpixParam.getParameter<int>("blade") != blade)
                continue;
            if (fpixParam.exists("panel"))
              if (fpixParam.getParameter<int>("panel") != panel)
                continue;
            if (fpixParam.exists("HVgroup"))
              if (fpixParam.getParameter<int>("HVgroup") != HVgroup(panel, ring))
                continue;
            if (!found) {
              float lorentzangle = static_cast<float>(fpixParam.getParameter<double>("angle"));
              LorentzAngle.putLorentzAngle(detid.rawId(), lorentzangle);
              LogPrint("SiPixelLorentzAngleDBLoader") << "   >> LA=" << lorentzangle;
              found = 2;
            } else if (found == 1) {
              LogPrint("SiPixelLorentzAngleDBLoader") << "The detid already given in ModuleParameters, skipping ...";
            } else
              LogError("SiPixelLorentzAngleDBLoader") << " ERROR!: detid already exists";
          }  // end loop on FPixParameters_
        }    // condition to read from config
      }      // end on being barrel or endcap
    }
  }

  edm::Service<cond::service::PoolDBOutputService> mydbservice;
  if (mydbservice.isAvailable()) {
    try {
      if (mydbservice->isNewTagRequest(recordName_)) {
        mydbservice->createOneIOV<SiPixelLorentzAngle>(LorentzAngle, mydbservice->beginOfTime(), recordName_);
      } else {
        mydbservice->appendOneIOV<SiPixelLorentzAngle>(LorentzAngle, mydbservice->currentTime(), recordName_);
      }
    } catch (const cond::Exception& er) {
      LogPrint("SiPixelLorentzAngleDBLoader") << "SiPixelLorentzAngleDBLoader" << er.what();
    } catch (const std::exception& er) {
      LogPrint("SiPixelLorentzAngleDBLoader") << "SiPixelLorentzAngleDBLoader"
                                              << "caught std::exception " << er.what();
    }
  } else {
    LogPrint("SiPixelLorentzAngleDBLoader") << "Service is unavailable";
  }
}

int SiPixelLorentzAngleDBLoader::HVgroup(int panel, int module) {
  if (1 == panel && (1 == module || 2 == module)) {
    return 1;
  } else if (1 == panel && (3 == module || 4 == module)) {
    return 2;
  } else if (2 == panel && 1 == module) {
    return 1;
  } else if (2 == panel && (2 == module || 3 == module)) {
    return 2;
  } else {
    LogError("SiPixelLorentzAngleDBLoader")
        << " *** error *** in SiPixelLorentzAngleDBLoader::HVgroup(...), panel = " << panel << ", module = " << module
        << endl;
    return 0;
  }
}

//define this as a plug-in
DEFINE_FWK_MODULE(SiPixelLorentzAngleDBLoader);
