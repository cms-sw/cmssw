#include "CondTools/SiPixel/test/SiPixelVCalDB.h"

using namespace std;
using namespace edm;

SiPixelVCalDB::SiPixelVCalDB(edm::ParameterSet const& iConfig) {
  recordName_ = iConfig.getUntrackedParameter<std::string>("record", "SiPixelVCalRcd");
  BPixParameters_ = iConfig.getUntrackedParameter<Parameters>("BPixParameters");
  FPixParameters_ = iConfig.getUntrackedParameter<Parameters>("FPixParameters");
}

SiPixelVCalDB::~SiPixelVCalDB() {}
void SiPixelVCalDB::beginJob() {}
void SiPixelVCalDB::endJob() {}

// Analyzer: Functions that gets called by framework every event
void SiPixelVCalDB::analyze(const edm::Event& e, const edm::EventSetup& iSetup) {
  SiPixelVCal* vcal = new SiPixelVCal();
  bool phase1 = true;

  // Retrieve tracker topology from geometry
  edm::ESHandle<TrackerTopology> tTopoHandle;
  iSetup.get<TrackerTopologyRcd>().get(tTopoHandle);
  const TrackerTopology* const tTopo = tTopoHandle.product();

  // Retrieve old style tracker geometry from geometry
  edm::ESHandle<TrackerGeometry> pDD;
  iSetup.get<TrackerDigiGeometryRecord>().get(pDD);
  std::cout << " There are " << pDD->detUnits().size() << " modules" << std::endl;

  for (const auto& it : pDD->detUnits()) {
    if (dynamic_cast<PixelGeomDetUnit const*>(it) != nullptr) {
      const DetId detid = it->geographicalId();
      const unsigned int rawDetId = detid.rawId();
      int subid = detid.subdetId();

      // FILL BPIX
      if (subid == static_cast<int>(PixelSubdetector::PixelBarrel)) {
        int layer = tTopo->pxbLayer(detid);    // 1, 2, 3, 4
        int ladder = tTopo->pxbLadder(detid);  // 1-12/28/44/64
        std::cout << " pixel barrel:"
                  << " detId=" << rawDetId << ", layer=" << layer << ", ladder=" << ladder;
        for (Parameters::iterator it = BPixParameters_.begin(); it != BPixParameters_.end(); ++it) {
          if (it->getParameter<int>("layer") == layer && it->getParameter<int>("ladder") == ladder) {
            float slope = (float)it->getParameter<double>("slope");
            float offset = (float)it->getParameter<double>("offset");
            std::cout << ";  VCal slope " << slope << ", offset " << offset;
            // edm::LogInfo("SiPixelVCalDB")  << "  detId " << rawDetId << " \t
            // VCal slope " << slope << ", offset " << offset;
            vcal->putSlopeAndOffset(detid, slope, offset);
          }
        }
        std::cout << std::endl;

        // FILL FPIX
      } else if (subid == static_cast<int>(PixelSubdetector::PixelEndcap)) {
        PixelEndcapName fpix(detid, tTopo, phase1);
        int side = tTopo->pxfSide(detid);   // 1 (-z), 2 for (+z)
        int disk = fpix.diskName();         // 1, 2, 3
        int disk2 = tTopo->pxfDisk(detid);  // 1, 2, 3
        int ring = fpix.ringName();         // 1 (lower), 2 (upper)
        if (disk != disk2) {
          edm::LogError("SiPixelVCalDB::analyze")
              << "Found contradicting FPIX disk number: " << disk << " vs." << disk2 << std::endl;
        }
        std::cout << " pixel endcap:"
                  << " detId=" << rawDetId << ", side=" << side << ", disk=" << disk << ", ring=" << ring;
        for (Parameters::iterator it = FPixParameters_.begin(); it != FPixParameters_.end(); ++it) {
          if (it->getParameter<int>("side") == side && it->getParameter<int>("disk") == disk &&
              it->getParameter<int>("ring") == ring) {
            float slope = (float)it->getParameter<double>("slope");
            float offset = (float)it->getParameter<double>("offset");
            std::cout << ";  VCal slope " << slope << ", offset " << offset;
            // edm::LogInfo("SiPixelVCalDB")  << "  detId " << rawDetId << " \t
            // VCal slope " << slope << ", offset " << offset;
            vcal->putSlopeAndOffset(rawDetId, slope, offset);
          }
        }
        std::cout << std::endl;

      } else {
        edm::LogError("SiPixelVCalDB::analyze") << "detid is Pixel but neither bpix nor fpix" << std::endl;
      }
    }
  }

  // Save to DB
  edm::Service<cond::service::PoolDBOutputService> mydbservice;
  if (mydbservice.isAvailable()) {
    try {
      if (mydbservice->isNewTagRequest(recordName_)) {
        mydbservice->createNewIOV<SiPixelVCal>(vcal, mydbservice->beginOfTime(), mydbservice->endOfTime(), recordName_);
      } else {
        mydbservice->appendSinceTime<SiPixelVCal>(vcal, mydbservice->currentTime(), recordName_);
      }
    } catch (const cond::Exception& er) {
      edm::LogError("SiPixelVCalDB") << er.what() << std::endl;
    } catch (const std::exception& er) {
      edm::LogError("SiPixelVCalDB") << "caught std::exception " << er.what() << std::endl;
    } catch (...) {
      edm::LogError("SiPixelVCalDB") << "Funny error" << std::endl;
    }
  } else {
    edm::LogError("SiPixelVCalDB") << "Service is unavailable" << std::endl;
  }
}
