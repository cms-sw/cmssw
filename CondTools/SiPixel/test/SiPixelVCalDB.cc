#include "CondTools/SiPixel/test/SiPixelVCalDB.h"

using namespace std;
using namespace edm;

SiPixelVCalDB::SiPixelVCalDB(edm::ParameterSet const& conf) : conf_(conf) {
  recordName_ = conf_.getUntrackedParameter<std::string>("record", "SiPixelVCalRcd");
  BPixParameters_ = conf_.getUntrackedParameter<Parameters>("BPixParameters");
  FPixParameters_ = conf_.getUntrackedParameter<Parameters>("FPixParameters");
}

SiPixelVCalDB::~SiPixelVCalDB() {}
void SiPixelVCalDB::beginJob() {}
void SiPixelVCalDB::endJob() {}

// Analyzer: Functions that gets called by framework every event

void SiPixelVCalDB::analyze(const edm::Event& e, const edm::EventSetup& es) {
  SiPixelVCal* VCal = new SiPixelVCal();

  // Put VCals for BPIX
  std::cout  << "Put VCal slope and offsets for BPix..." << std::endl;
  //edm::LogInfo("SiPixelVCalReader")  << "Put VCal slope and offsets for BPix...";
  for (Parameters::iterator it = BPixParameters_.begin(); it != BPixParameters_.end(); ++it) {
      unsigned int layer = (unsigned int) it->getParameter<unsigned int>("layer");
      PixelId pixid = calculateBPixID(layer);
      float slope = (float) it->getParameter<double>("slope");
      float offset = (float) it->getParameter<double>("offset");
      std::cout  << "  pixid " << pixid << " \t VCal slope " << slope << ", offset " << offset << std::endl;
      //edm::LogInfo("SiPixelVCalReader")  << "  pixid " << pixid << " \t VCal slope " << slope << ", offset " << offset;
      VCal->putSlopeAndOffset(pixid,slope,offset);
  }

  // Put VCals for FPIX
  std::cout << std::endl << "Put VCal slope and offsets for FPix..." << std::endl;
  //edm::LogInfo("SiPixelVCalReader")  << "Put VCal slope and offsets for BPix...";
  for (Parameters::iterator it = FPixParameters_.begin(); it != FPixParameters_.end(); ++it) {
      unsigned int side = (unsigned int) it->getParameter<unsigned int>("side");
      unsigned int disk = (unsigned int) it->getParameter<unsigned int>("disk");
      unsigned int ring = (unsigned int) it->getParameter<unsigned int>("ring");
      PixelId pixid = calculateFPixID(side,disk,ring);
      float slope = (float) it->getParameter<double>("slope");
      float offset = (float) it->getParameter<double>("offset");
      std::cout  << "  pixid " << pixid << " \t VCal slope " << slope << ", offset " << offset << std::endl;
      //edm::LogInfo("SiPixelVCalReader")  << "  pixid " << pixid << " \t VCal slope " << slope << ", offset " << offset;
      VCal->putSlopeAndOffset(pixid,slope,offset);
  }

  // Save to DB
  edm::Service<cond::service::PoolDBOutputService> mydbservice;
  if (mydbservice.isAvailable()) {
    try {
      if (mydbservice->isNewTagRequest(recordName_)) {
        mydbservice->createNewIOV<SiPixelVCal>(
            VCal, mydbservice->beginOfTime(), mydbservice->endOfTime(), recordName_);
      } else {
        mydbservice->appendSinceTime<SiPixelVCal>(VCal, mydbservice->currentTime(), recordName_);
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


const SiPixelVCalDB::PixelId SiPixelVCalDB::calculateBPixID(const unsigned int layer){
  // BPix: 1000*(subdetId=1) + 100*(layer=1,2,3,4)
  // => L1=1100, L2=1200, L3=1300, L4=1400
  PixelId bpixLayer = static_cast<PixelId> (1000+100*layer);
  return bpixLayer;
}


const SiPixelVCalDB::PixelId SiPixelVCalDB::calculateFPixID(const unsigned int side, const unsigned int disk, const unsigned int ring){
  // FPix: 1000*(subdetId=2) + 100*(side=1,2) + 10*(disk=1,2,3) + 1*(ring=1,2)
  // => Rm1l=2111, Rm1u=2112, Rm2l=2121, Rm2u=2122, Rm3l=2131, Rm3u=2132,  (FPix minus)
  //    Rp1l=2211, Rp1u=2212, Rp2l=2221, Rp2u=2222, Rp3l=2231, Rp3u=2232,  (FPix plus)
  PixelId fpixRing = static_cast<PixelId> (2000+100*side+10*disk+ring);
  return fpixRing;
}


const int SiPixelVCalDB::getPixelSubDetector(const unsigned int pixid){ //SiPixelVCalDB::PixelId
  // subdetId: BPix=1, FPix=2
  return (pixid/1000)%10;
}


const SiPixelVCalDB::PixelId SiPixelVCalDB::detIdToPixelId(const unsigned int detid, const TrackerTopology* trackTopo, const bool phase1){
    DetId detId = DetId(detid);
    unsigned int subid = detId.subdetId();
    unsigned int pixid = 0;
    if (subid==1) { // BPix static_cast<int>(PixelSubdetector::PixelEndcap)
      PixelBarrelName bpix(detId,trackTopo,phase1);
      int layer = bpix.layerName();
      pixid = calculateBPixID(layer);
    }
    else if (subid==2) { // FPix static_cast<int>(PixelSubdetector::PixelBarrel)
      PixelEndcapName fpix(detId,trackTopo,phase1);
      int side = trackTopo->pxfSide(detId); // 1 (-z), 2 for (+z)
      int disk = fpix.diskName(); //trackTopo->pxfDisk(detId); // 1, 2, 3
      int ring = fpix.ringName(); // 1 (lower), 2 (upper)
      pixid = calculateFPixID(side,disk,ring);
    }
    PixelId pixID = static_cast<PixelId>(pixid);
    return pixID;
}


