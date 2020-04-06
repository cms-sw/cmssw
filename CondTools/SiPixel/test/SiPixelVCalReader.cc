//#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "CondTools/SiPixel/test/SiPixelVCalReader.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"

using namespace cms;

SiPixelVCalReader::SiPixelVCalReader(const edm::ParameterSet& iConfig)
    : printdebug_(iConfig.getUntrackedParameter<bool>("printDebug", false)),
      useSimRcd_(iConfig.getParameter<bool>("useSimRcd")) {}

SiPixelVCalReader::~SiPixelVCalReader() {}

void SiPixelVCalReader::analyze(const edm::Event& e, const edm::EventSetup& iSetup) {
  edm::ESHandle<SiPixelVCal> SiPixelVCal_;
  if (useSimRcd_ == true)
    iSetup.get<SiPixelVCalSimRcd>().get(SiPixelVCal_);
  else
    iSetup.get<SiPixelVCalRcd>().get(SiPixelVCal_);
  edm::LogInfo("SiPixelVCalReader")
      << "[SiPixelVCalReader::analyze] End Reading SiPixelVCal" << std::endl;
  edm::Service<TFileService> fs;

  // Prepare tree
  TTree* tree = new TTree("tree", "tree");
  uint32_t pixid;
  double slope, offset;
  tree->Branch("pixid", &pixid, "pixid/I");
  tree->Branch("slope", &slope, "slope/D");
  tree->Branch("offset", &offset, "offset/D");

  // Prepare histograms
  slopeBPix_ = fs->make<TH1F>("VCalSlopeBarrelPixel", "VCalSlopeBarrelPixel", 150, 0, 100);
  slopeFPix_ = fs->make<TH1F>("VCalSlopeForwardPixel", "VCalSlopeForwardPixel", 150, 0, 100);
  offsetBPix_ = fs->make<TH1F>("VCalOffsetBarrelPixel", "VCalOffsetBarrelPixel", 150, -300, 100);
  offsetFPix_ = fs->make<TH1F>("VCalOffsetForwardPixel", "VCalOffsetForwardPixel", 150, -300, 100);
  std::map<unsigned int, float> slopes = SiPixelVCal_->getSlope();
  std::map<unsigned int, float> offsets = SiPixelVCal_->getOffset();
  std::map<unsigned int, float>::const_iterator it;
  
  // Fill histograms
  for (it = slopes.begin(); it != slopes.end(); it++) {
    pixid = it->first;
    slope  = it->second;
    offset = 0.;
    unsigned int subdet = SiPixelVCalDB::getPixelSubDetector(pixid);
    if (offsets.find(pixid)!=offsets.end()) // assume slopes and offsets have the same keys
      offset = offsets[pixid];
    else
      std::cerr << "Did not find offset for pixid " << pixid << "! Setting to " << offset << std::endl;
    std::cout  << "pixid " << pixid << " \t VCal slope " << slope << ", offset " << offset << std::endl;
    edm::LogInfo("SiPixelVCalReader")  << "pixid " << pixid << " \t VCal slope " << slope << ", offset " << offset;
    if (subdet==static_cast<int>(PixelSubdetector::PixelBarrel)) {
      slopeBPix_->Fill(slope);
      offsetBPix_->Fill(offset);
    } else if (subdet==static_cast<int>(PixelSubdetector::PixelEndcap)) {
      slopeFPix_->Fill(slope);
      offsetFPix_->Fill(offset);
    }
    tree->Fill();
  }

}
