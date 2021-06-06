#include "CondTools/SiPixel/test/SiPixelVCalReader.h"

using namespace cms;

SiPixelVCalReader::SiPixelVCalReader(const edm::ParameterSet& iConfig)
    : printdebug_(iConfig.getUntrackedParameter<bool>("printDebug", false)),
      useSimRcd_(iConfig.getParameter<bool>("useSimRcd")) {}

SiPixelVCalReader::~SiPixelVCalReader() {}
void SiPixelVCalReader::beginJob() {}
void SiPixelVCalReader::endJob() {}

void SiPixelVCalReader::analyze(const edm::Event& e, const edm::EventSetup& iSetup) {
  edm::ESHandle<SiPixelVCal> siPixelVCal;

  // Get record & file service
  if (useSimRcd_ == true)
    iSetup.get<SiPixelVCalSimRcd>().get(siPixelVCal);
  else
    iSetup.get<SiPixelVCalRcd>().get(siPixelVCal);
  edm::LogInfo("SiPixelVCalReader") << "[SiPixelVCalReader::analyze] End Reading SiPixelVCal" << std::endl;
  edm::Service<TFileService> fs;

  // Retrieve tracker topology from geometry
  edm::ESHandle<TrackerTopology> tTopoHandle;
  iSetup.get<TrackerTopologyRcd>().get(tTopoHandle);
  const TrackerTopology* const tTopo = tTopoHandle.product();

  // Retrieve old style tracker geometry from geometry
  edm::ESHandle<TrackerGeometry> pDD;
  iSetup.get<TrackerDigiGeometryRecord>().get(pDD);
  std::cout << " There are " << pDD->detUnits().size() << " modules" << std::endl;

  // Phase
  bool phase1 = true;

  // Prepare tree
  TTree* tree = new TTree("tree", "tree");
  uint32_t detid, subdet, layer, ladder, side, disk, ring;
  double slope, offset;
  tree->Branch("detid", &detid, "detid/I");
  tree->Branch("subdet", &subdet, "subdet/I");
  tree->Branch("layer", &layer, "layer/I");
  tree->Branch("ladder", &ladder, "ladder/I");
  tree->Branch("side", &side, "side/I");
  tree->Branch("disk", &disk, "disk/I");
  tree->Branch("ring", &ring, "ring/I");
  tree->Branch("slope", &slope, "slope/D");
  tree->Branch("offset", &offset, "offset/D");

  // Prepare histograms
  slopeBPix_ = fs->make<TH1F>("VCalSlopeBarrelPixel", "VCalSlopeBarrelPixel", 150, 0, 100);
  slopeFPix_ = fs->make<TH1F>("VCalSlopeForwardPixel", "VCalSlopeForwardPixel", 150, 0, 100);
  offsetBPix_ = fs->make<TH1F>("VCalOffsetBarrelPixel", "VCalOffsetBarrelPixel", 200, -900, 100);
  offsetFPix_ = fs->make<TH1F>("VCalOffsetForwardPixel", "VCalOffsetForwardPixel", 200, -900, 100);
  std::map<unsigned int, SiPixelVCal::VCal> vcal = siPixelVCal->getSlopeAndOffset();
  std::map<unsigned int, SiPixelVCal::VCal>::const_iterator it;

  // Fill histograms
  std::cout << std::setw(12) << "detid" << std::setw(8) << "subdet" << std::setw(8) << "layer" << std::setw(8) << "disk"
            << std::setw(14) << "VCal slope" << std::setw(8) << "offset" << std::endl;
  for (it = vcal.begin(); it != vcal.end(); it++) {
    detid = it->first;
    slope = it->second.slope;
    offset = it->second.offset;
    const DetId detIdObj(detid);
    PixelEndcapName fpix(detid, tTopo, phase1);
    subdet = detIdObj.subdetId();
    layer = tTopo->pxbLayer(detIdObj);    // 1, 2, 3, 4
    ladder = tTopo->pxbLadder(detIdObj);  // 1-12/28/44/64
    side = tTopo->pxfSide(detIdObj);      // 1, 2
    disk = tTopo->pxfDisk(detIdObj);      // 1, 2, 3
    ring = fpix.ringName();               // 1 (lower), 2 (upper)
    std::cout << std::setw(12) << detid << std::setw(8) << subdet << std::setw(8) << layer << std::setw(8) << disk
              << std::setw(14) << slope << std::setw(8) << offset << std::endl;
    // std::cout << "detid " << detid << ", subdet " << subdet << ", layer " <<
    // layer << ", disk " << disk
    //          << ", VCal slope " << slope << ", offset " << offset <<
    //          std::endl;
    // edm::LogInfo("SiPixelVCalReader") << "detid " << detid << ", subdet " <<
    // subdet << ", layer " << layer
    //                                  << ", VCal slope " << slope << ", offset
    //                                  " << offset;
    if (subdet == static_cast<int>(PixelSubdetector::PixelBarrel)) {
      slopeBPix_->Fill(slope);
      offsetBPix_->Fill(offset);
    } else if (subdet == static_cast<int>(PixelSubdetector::PixelEndcap)) {
      slopeFPix_->Fill(slope);
      offsetFPix_->Fill(offset);
    }
    tree->Fill();
  }
}
