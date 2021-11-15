// system includes
#include <map>
#include <memory>
#include <string>
#include <iostream>
#include <fstream>

// user includes
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelLorentzAngle.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "Geometry/CommonDetUnit/interface/PixelGeomDetUnit.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

class SiPixelLorentzAngleDB : public edm::one::EDAnalyzer<> {
public:
  explicit SiPixelLorentzAngleDB(const edm::ParameterSet& conf);
  ~SiPixelLorentzAngleDB() override;
  void analyze(const edm::Event& e, const edm::EventSetup& c) override;

private:
  const edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> tkGeomToken_;
  const edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> tkTopoToken_;

  unsigned int HVgroup(unsigned int panel, unsigned int module);

  std::vector<std::pair<uint32_t, float> > detid_la;
  double magneticField_;
  std::string recordName_;

  typedef std::vector<edm::ParameterSet> Parameters;
  Parameters BPixParameters_;
  Parameters FPixParameters_;
  Parameters ModuleParameters_;

  std::string fileName_;
  bool useFile_;
};

using namespace std;
using namespace edm;

//Constructor

SiPixelLorentzAngleDB::SiPixelLorentzAngleDB(edm::ParameterSet const& conf)
    : tkGeomToken_(esConsumes()), tkTopoToken_(esConsumes()) {
  magneticField_ = conf.getParameter<double>("magneticField");
  recordName_ = conf.getUntrackedParameter<std::string>("record", "SiPixelLorentzAngleRcd");
  useFile_ = conf.getParameter<bool>("useFile");
  fileName_ = conf.getParameter<string>("fileName");

  BPixParameters_ = conf.getUntrackedParameter<Parameters>("BPixParameters");
  FPixParameters_ = conf.getUntrackedParameter<Parameters>("FPixParameters");
  ModuleParameters_ = conf.getUntrackedParameter<Parameters>("ModuleParameters");
}

// Virtual destructor needed.
SiPixelLorentzAngleDB::~SiPixelLorentzAngleDB() = default;

// Analyzer: Functions that gets called by framework every event

void SiPixelLorentzAngleDB::analyze(const edm::Event& e, const edm::EventSetup& es) {
  SiPixelLorentzAngle LorentzAngle;

  //Retrieve tracker topology from geometry
  const TrackerTopology* tTopo = &es.getData(tkTopoToken_);

  //Retrieve old style tracker geometry from geometry
  const TrackerGeometry* pDD = &es.getData(tkGeomToken_);
  edm::LogInfo("SiPixelLorentzAngle (old)")
      << " There are " << pDD->detUnits().size() << " detectors (old)" << std::endl;

  for (const auto& it : pDD->detUnits()) {
    if (dynamic_cast<PixelGeomDetUnit const*>(it) != nullptr) {
      DetId detid = it->geographicalId();
      const DetId detidc = it->geographicalId();

      // fill bpix values for LA
      if (detid.subdetId() == static_cast<int>(PixelSubdetector::PixelBarrel)) {
        edm::LogPrint("SiPixelLorentzAngleDB")
            << " pixel barrel:"
            << "  layer=" << tTopo->pxbLayer(detidc.rawId()) << "  ladder=" << tTopo->pxbLadder(detidc.rawId())
            << "  module=" << tTopo->pxbModule(detidc.rawId()) << "  rawId=" << detidc.rawId() << endl;

        if (!useFile_) {
          //first individuals are put
          for (Parameters::iterator it = ModuleParameters_.begin(); it != ModuleParameters_.end(); ++it) {
            if (it->getParameter<unsigned int>("rawid") == detidc.rawId()) {
              float lorentzangle = (float)it->getParameter<double>("angle");
              LorentzAngle.putLorentzAngle(detid.rawId(), lorentzangle);
              edm::LogPrint("SiPixelLorentzAngleDB")
                  << " individual value=" << lorentzangle << " put into rawid=" << detid.rawId() << endl;
            }
          }

          //modules already put are automatically skipped
          for (Parameters::iterator it = BPixParameters_.begin(); it != BPixParameters_.end(); ++it) {
            if (it->getParameter<unsigned int>("module") == tTopo->pxbModule(detidc.rawId()) &&
                it->getParameter<unsigned int>("layer") == tTopo->pxbLayer(detidc.rawId())) {
              float lorentzangle = (float)it->getParameter<double>("angle");
              LorentzAngle.putLorentzAngle(detid.rawId(), lorentzangle);
            }
          }

        } else {
          edm::LogError("SiPixelLorentzAngleDB")
              << "[SiPixelLorentzAngleDB::analyze] method for reading file not implemented yet" << std::endl;
        }

        // fill fpix values for LA
      } else if (detid.subdetId() == static_cast<int>(PixelSubdetector::PixelEndcap)) {
        edm::LogPrint("SiPixelLorentzAngleDB")
            << " pixel endcap:"
            << "  side=" << tTopo->pxfSide(detidc.rawId()) << "  disk=" << tTopo->pxfDisk(detidc.rawId())
            << "  blade=" << tTopo->pxfBlade(detidc.rawId()) << "  panel=" << tTopo->pxfPanel(detidc.rawId())
            << "  module=" << tTopo->pxfModule(detidc.rawId()) << "  rawId=" << detidc.rawId() << endl;

        //first individuals are put
        for (Parameters::iterator it = ModuleParameters_.begin(); it != ModuleParameters_.end(); ++it) {
          if (it->getParameter<unsigned int>("rawid") == detidc.rawId()) {
            float lorentzangle = (float)it->getParameter<double>("angle");
            LorentzAngle.putLorentzAngle(detid.rawId(), lorentzangle);
            edm::LogPrint("SiPixelLorentzAngleDB")
                << " individual value=" << lorentzangle << " put into rawid=" << detid.rawId() << endl;
          }
        }

        //modules already put are automatically skipped
        for (Parameters::iterator it = FPixParameters_.begin(); it != FPixParameters_.end(); ++it) {
          if (it->getParameter<unsigned int>("side") == tTopo->pxfSide(detidc.rawId()) &&
              it->getParameter<unsigned int>("disk") == tTopo->pxfDisk(detidc.rawId()) &&
              it->getParameter<unsigned int>("HVgroup") ==
                  HVgroup(tTopo->pxfPanel(detidc.rawId()), tTopo->pxfModule(detidc.rawId()))) {
            float lorentzangle = (float)it->getParameter<double>("angle");
            LorentzAngle.putLorentzAngle(detid.rawId(), lorentzangle);
          }
        }

      } else {
        edm::LogError("SiPixelLorentzAngleDB")
            << "[SiPixelLorentzAngleDB::analyze] detid is Pixel but neither bpix nor fpix" << std::endl;
      }
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
      edm::LogError("SiPixelLorentzAngleDB") << er.what() << std::endl;
    } catch (const std::exception& er) {
      edm::LogError("SiPixelLorentzAngleDB") << "caught std::exception " << er.what() << std::endl;
    } catch (...) {
      edm::LogError("SiPixelLorentzAngleDB") << "Funny error" << std::endl;
    }
  } else {
    edm::LogError("SiPixelLorentzAngleDB") << "Service is unavailable" << std::endl;
  }
}

unsigned int SiPixelLorentzAngleDB::HVgroup(unsigned int panel, unsigned int module) {
  if (1 == panel && (1 == module || 2 == module)) {
    return 1;
  } else if (1 == panel && (3 == module || 4 == module)) {
    return 2;
  } else if (2 == panel && 1 == module) {
    return 1;
  } else if (2 == panel && (2 == module || 3 == module)) {
    return 2;
  } else {
    edm::LogPrint("SiPixelLorentzAngleDB") << " *** error *** in SiPixelLorentzAngleDB::HVgroup(...), panel = " << panel
                                           << ", module = " << module << endl;
    return 0;
  }
}
DEFINE_FWK_MODULE(SiPixelLorentzAngleDB);
