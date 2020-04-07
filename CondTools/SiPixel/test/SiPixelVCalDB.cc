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
void SiPixelVCalDB::analyze(const edm::Event& e, const edm::EventSetup& es) {
  SiPixelVCal* vcal = new SiPixelVCal();

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
      vcal->putSlopeAndOffset(pixid,slope,offset);
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
      vcal->putSlopeAndOffset(pixid,slope,offset);
  }

  // Save to DB
  edm::Service<cond::service::PoolDBOutputService> mydbservice;
  if (mydbservice.isAvailable()) {
    try {
      if (mydbservice->isNewTagRequest(recordName_)) {
        mydbservice->createNewIOV<SiPixelVCal>(
            vcal, mydbservice->beginOfTime(), mydbservice->endOfTime(), recordName_);
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

