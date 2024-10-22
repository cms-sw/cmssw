// system includes
#include <memory>
#include <string>
#include <iostream>
#include <fstream>
#include <limits>
#include <map>

// user includes
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelDynamicInefficiency.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/MeasurementError.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/MeasurementPoint.h"
#include "DataFormats/GeometrySurface/interface/GloballyPositioned.h"
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
#include "Geometry/Records/interface/TrackerTopologyRcd.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"

class SiPixelDynamicInefficiencyDB : public edm::one::EDAnalyzer<> {
public:
  explicit SiPixelDynamicInefficiencyDB(const edm::ParameterSet& conf);

  ~SiPixelDynamicInefficiencyDB() override;
  void analyze(const edm::Event& e, const edm::EventSetup& c) override;

private:
  const edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> tkTopoToken_;
  edm::ParameterSet conf_;
  std::string recordName_;

  typedef std::vector<edm::ParameterSet> Parameters;
  Parameters thePixelGeomFactors_;
  Parameters theColGeomFactors_;
  Parameters theChipGeomFactors_;
  Parameters thePUEfficiency_;
  double theInstLumiScaleFactor_;
};

using namespace std;
using namespace edm;

//Constructor

SiPixelDynamicInefficiencyDB::SiPixelDynamicInefficiencyDB(edm::ParameterSet const& conf)
    : tkTopoToken_(esConsumes()), conf_(conf) {
  recordName_ = conf_.getUntrackedParameter<std::string>("record", "SiPixelDynamicInefficiencyRcd");
  thePixelGeomFactors_ = conf_.getUntrackedParameter<Parameters>("thePixelGeomFactors");
  theColGeomFactors_ = conf_.getUntrackedParameter<Parameters>("theColGeomFactors");
  theChipGeomFactors_ = conf_.getUntrackedParameter<Parameters>("theChipGeomFactors");
  thePUEfficiency_ = conf_.getUntrackedParameter<Parameters>("thePUEfficiency");
  theInstLumiScaleFactor_ = conf_.getUntrackedParameter<double>("theInstLumiScaleFactor");
}

// Virtual destructor needed.
SiPixelDynamicInefficiencyDB::~SiPixelDynamicInefficiencyDB() = default;

// Analyzer: Functions that gets called by framework every event

void SiPixelDynamicInefficiencyDB::analyze(const edm::Event& e, const edm::EventSetup& es) {
  SiPixelDynamicInefficiency DynamicInefficiency;

  //Retrieve tracker topology from geometry
  const TrackerTopology* const tTopo = &es.getData(tkTopoToken_);

  uint32_t max = numeric_limits<uint32_t>::max();
  uint32_t mask;
  uint32_t layer, LAYER = 0;
  uint32_t ladder, LADDER = 0;
  uint32_t module, MODULE = 0;
  uint32_t side, SIDE = 0;
  uint32_t disk, DISK = 0;
  uint32_t blade, BLADE = 0;
  uint32_t panel, PANEL = 0;

  //Put BPix masks
  mask = tTopo->pxbDetId(max, LADDER, MODULE).rawId();
  DynamicInefficiency.putDetIdmask(mask);
  mask = tTopo->pxbDetId(LAYER, max, MODULE).rawId();
  DynamicInefficiency.putDetIdmask(mask);
  mask = tTopo->pxbDetId(LAYER, LADDER, max).rawId();
  DynamicInefficiency.putDetIdmask(mask);
  //Put FPix masks
  mask = tTopo->pxfDetId(max, DISK, BLADE, PANEL, MODULE).rawId();
  DynamicInefficiency.putDetIdmask(mask);
  mask = tTopo->pxfDetId(SIDE, max, BLADE, PANEL, MODULE).rawId();
  DynamicInefficiency.putDetIdmask(mask);
  mask = tTopo->pxfDetId(SIDE, DISK, max, PANEL, MODULE).rawId();
  DynamicInefficiency.putDetIdmask(mask);
  mask = tTopo->pxfDetId(SIDE, DISK, BLADE, max, MODULE).rawId();
  DynamicInefficiency.putDetIdmask(mask);
  mask = tTopo->pxfDetId(SIDE, DISK, BLADE, PANEL, max).rawId();
  DynamicInefficiency.putDetIdmask(mask);

  //Put PixelGeomFactors
  for (Parameters::iterator it = thePixelGeomFactors_.begin(); it != thePixelGeomFactors_.end(); ++it) {
    string det = it->getParameter<string>("det");
    it->exists("layer") ? layer = it->getParameter<unsigned int>("layer") : layer = LAYER;
    it->exists("ladder") ? ladder = it->getParameter<unsigned int>("ladder") : ladder = LADDER;
    it->exists("module") ? module = it->getParameter<unsigned int>("module") : module = MODULE;
    it->exists("side") ? side = it->getParameter<unsigned int>("side") : side = SIDE;
    it->exists("disk") ? disk = it->getParameter<unsigned int>("disk") : disk = DISK;
    it->exists("blade") ? blade = it->getParameter<unsigned int>("blade") : blade = BLADE;
    it->exists("panel") ? panel = it->getParameter<unsigned int>("panel") : panel = PANEL;
    double factor = it->getParameter<double>("factor");
    if (det == "bpix") {
      DetId detID = tTopo->pxbDetId(layer, ladder, module);
      edm::LogPrint("SiPixelDynamicInefficiencyDB") << "Putting Pixel geom BPix layer " << layer << " ladder " << ladder
                                                    << " module " << module << " factor " << factor << std::endl;
      DynamicInefficiency.putPixelGeomFactor(detID.rawId(), factor);
    } else if (det == "fpix") {
      DetId detID = tTopo->pxfDetId(side, disk, blade, panel, module);
      edm::LogPrint("SiPixelDynamicInefficiencyDB")
          << "Putting Pixel geom FPix side " << side << " disk " << disk << " blade " << blade << " panel " << panel
          << " module " << module << " factor " << factor << std::endl;
      DynamicInefficiency.putPixelGeomFactor(detID.rawId(), factor);
    } else
      edm::LogError("SiPixelDynamicInefficiencyDB")
          << "SiPixelDynamicInefficiencyDB input detector part is neither bpix nor fpix" << std::endl;
  }

  //Put ColumnGeomFactors
  for (Parameters::iterator it = theColGeomFactors_.begin(); it != theColGeomFactors_.end(); ++it) {
    string det = it->getParameter<string>("det");
    it->exists("layer") ? layer = it->getParameter<unsigned int>("layer") : layer = LAYER;
    it->exists("ladder") ? ladder = it->getParameter<unsigned int>("ladder") : ladder = LADDER;
    it->exists("module") ? module = it->getParameter<unsigned int>("module") : module = MODULE;
    it->exists("side") ? side = it->getParameter<unsigned int>("side") : side = SIDE;
    it->exists("disk") ? disk = it->getParameter<unsigned int>("disk") : disk = DISK;
    it->exists("blade") ? blade = it->getParameter<unsigned int>("blade") : blade = BLADE;
    it->exists("panel") ? panel = it->getParameter<unsigned int>("panel") : panel = PANEL;
    double factor = it->getParameter<double>("factor");
    if (det == "bpix") {
      DetId detID = tTopo->pxbDetId(layer, ladder, module);
      edm::LogPrint("SiPixelDynamicInefficiencyDB")
          << "Putting Column geom BPix layer " << layer << " ladder " << ladder << " module " << module << " factor "
          << factor << std::endl;
      DynamicInefficiency.putColGeomFactor(detID.rawId(), factor);
    } else if (det == "fpix") {
      DetId detID = tTopo->pxfDetId(side, disk, blade, panel, module);
      edm::LogPrint("SiPixelDynamicInefficiencyDB")
          << "Putting Column geom FPix side " << side << " disk " << disk << " blade " << blade << " panel " << panel
          << " module " << module << " factor " << factor << std::endl;
      DynamicInefficiency.putColGeomFactor(detID.rawId(), factor);
    } else
      edm::LogError("SiPixelDynamicInefficiencyDB")
          << "SiPixelDynamicInefficiencyDB input detector part is neither bpix nor fpix" << std::endl;
  }

  //Put ChipGeomFactors
  for (Parameters::iterator it = theChipGeomFactors_.begin(); it != theChipGeomFactors_.end(); ++it) {
    string det = it->getParameter<string>("det");
    it->exists("layer") ? layer = it->getParameter<unsigned int>("layer") : layer = LAYER;
    it->exists("ladder") ? ladder = it->getParameter<unsigned int>("ladder") : ladder = LADDER;
    it->exists("module") ? module = it->getParameter<unsigned int>("module") : module = MODULE;
    it->exists("side") ? side = it->getParameter<unsigned int>("side") : side = SIDE;
    it->exists("disk") ? disk = it->getParameter<unsigned int>("disk") : disk = DISK;
    it->exists("blade") ? blade = it->getParameter<unsigned int>("blade") : blade = BLADE;
    it->exists("panel") ? panel = it->getParameter<unsigned int>("panel") : panel = PANEL;
    double factor = it->getParameter<double>("factor");
    if (det == "bpix") {
      DetId detID = tTopo->pxbDetId(layer, ladder, module);
      edm::LogPrint("SiPixelDynamicInefficiencyDB") << "Putting Chip geom BPix layer " << layer << " ladder " << ladder
                                                    << " module " << module << " factor " << factor << std::endl;
      DynamicInefficiency.putChipGeomFactor(detID.rawId(), factor);
    } else if (det == "fpix") {
      DetId detID = tTopo->pxfDetId(side, disk, blade, panel, module);
      edm::LogPrint("SiPixelDynamicInefficiencyDB")
          << "Putting Chip geom FPix side " << side << " disk " << disk << " blade " << blade << " panel " << panel
          << " module " << module << " factor " << factor << std::endl;
      DynamicInefficiency.putChipGeomFactor(detID.rawId(), factor);
    } else
      edm::LogError("SiPixelDynamicInefficiencyDB")
          << "SiPixelDynamicInefficiencyDB input detector part is neither bpix nor fpix" << std::endl;
  }

  //Put PUFactors
  for (Parameters::iterator it = thePUEfficiency_.begin(); it != thePUEfficiency_.end(); ++it) {
    string det = it->getParameter<string>("det");
    it->exists("layer") ? layer = it->getParameter<unsigned int>("layer") : layer = LAYER;
    it->exists("ladder") ? ladder = it->getParameter<unsigned int>("ladder") : ladder = LADDER;
    it->exists("module") ? module = it->getParameter<unsigned int>("module") : module = MODULE;
    it->exists("side") ? side = it->getParameter<unsigned int>("side") : side = SIDE;
    it->exists("disk") ? disk = it->getParameter<unsigned int>("disk") : disk = DISK;
    it->exists("blade") ? blade = it->getParameter<unsigned int>("blade") : blade = BLADE;
    it->exists("panel") ? panel = it->getParameter<unsigned int>("panel") : panel = PANEL;
    std::vector<double> factor = it->getParameter<std::vector<double> >("factor");
    if (det == "bpix") {
      DetId detID = tTopo->pxbDetId(layer, ladder, module);
      edm::LogPrint("SiPixelDynamicInefficiencyDB")
          << "Putting PU efficiency BPix layer " << layer << " ladder " << ladder << " module " << module
          << " factor size " << factor.size() << std::endl;
      DynamicInefficiency.putPUFactor(detID.rawId(), factor);
    } else if (det == "fpix") {
      DetId detID = tTopo->pxfDetId(side, disk, blade, panel, module);
      edm::LogPrint("SiPixelDynamicInefficiencyDB")
          << "Putting PU efficiency FPix side " << side << " disk " << disk << " blade " << blade << " panel " << panel
          << " module " << module << " factor size " << factor.size() << std::endl;
      DynamicInefficiency.putPUFactor(detID.rawId(), factor);
    }
  }
  //Put theInstLumiScaleFactor
  DynamicInefficiency.puttheInstLumiScaleFactor(theInstLumiScaleFactor_);

  edm::Service<cond::service::PoolDBOutputService> mydbservice;
  if (mydbservice.isAvailable()) {
    try {
      if (mydbservice->isNewTagRequest(recordName_)) {
        mydbservice->createOneIOV<SiPixelDynamicInefficiency>(
            DynamicInefficiency, mydbservice->beginOfTime(), recordName_);
      } else {
        mydbservice->appendOneIOV<SiPixelDynamicInefficiency>(
            DynamicInefficiency, mydbservice->currentTime(), recordName_);
      }
    } catch (const cond::Exception& er) {
      edm::LogError("SiPixelDynamicInefficiencyDB") << er.what() << std::endl;
    } catch (const std::exception& er) {
      edm::LogError("SiPixelDynamicInefficiencyDB") << "caught std::exception " << er.what() << std::endl;
    } catch (...) {
      edm::LogError("SiPixelDynamicInefficiencyDB") << "Funny error" << std::endl;
    }
  } else {
    edm::LogError("SiPixelDynamicInefficiencyDB") << "Service is unavailable" << std::endl;
  }
}
DEFINE_FWK_MODULE(SiPixelDynamicInefficiencyDB);
