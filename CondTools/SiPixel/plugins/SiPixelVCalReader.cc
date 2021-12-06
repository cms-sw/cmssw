// system includes
#include <cstdio>
#include <iomanip>  // std::setw
#include <iostream>
#include <sys/time.h>

// user includes
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "CondFormats/DataRecord/interface/SiPixelVCalRcd.h"
#include "CondFormats/DataRecord/interface/SiPixelVCalSimRcd.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelVCal.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "DataFormats/TrackerCommon/interface/PixelBarrelName.h"
#include "DataFormats/TrackerCommon/interface/PixelEndcapName.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"

// ROOT includes
#include "TFile.h"
#include "TH2F.h"
#include "TROOT.h"
#include "TTree.h"

class SiPixelVCalReader : public edm::one::EDAnalyzer<edm::one::SharedResources> {
public:
  explicit SiPixelVCalReader(const edm::ParameterSet&);
  ~SiPixelVCalReader() override;
  void analyze(const edm::Event&, const edm::EventSetup&) override;

private:
  // es tokens
  const edm::ESGetToken<SiPixelVCal, SiPixelVCalSimRcd> siPixelVCalSimToken_;
  const edm::ESGetToken<SiPixelVCal, SiPixelVCalRcd> siPixelVCalToken_;
  const edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> tkGeomToken_;
  const edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> tkTopoToken_;

  const bool printdebug_;
  const bool useSimRcd_;

  TH1F* slopeBPix_;
  TH1F* slopeFPix_;
  TH1F* offsetBPix_;
  TH1F* offsetFPix_;
};

using namespace cms;

SiPixelVCalReader::SiPixelVCalReader(const edm::ParameterSet& iConfig)
    : siPixelVCalSimToken_(esConsumes()),
      siPixelVCalToken_(esConsumes()),
      tkGeomToken_(esConsumes()),
      tkTopoToken_(esConsumes()),
      printdebug_(iConfig.getUntrackedParameter<bool>("printDebug", false)),
      useSimRcd_(iConfig.getParameter<bool>("useSimRcd")) {
  usesResource(TFileService::kSharedResource);
}

SiPixelVCalReader::~SiPixelVCalReader() = default;

void SiPixelVCalReader::analyze(const edm::Event& e, const edm::EventSetup& iSetup) {
  const SiPixelVCal* siPixelVCal;

  // Get record & file service
  if (useSimRcd_) {
    siPixelVCal = &iSetup.getData(siPixelVCalSimToken_);
  } else {
    siPixelVCal = &iSetup.getData(siPixelVCalToken_);
  }
  edm::LogInfo("SiPixelVCalReader") << "[SiPixelVCalReader::analyze] End Reading SiPixelVCal" << std::endl;
  edm::Service<TFileService> fs;

  // Retrieve tracker topology from geometry
  const TrackerTopology* const tTopo = &iSetup.getData(tkTopoToken_);

  // Retrieve old style tracker geometry from geometry
  const TrackerGeometry* pDD = &iSetup.getData(tkGeomToken_);
  edm::LogPrint("SiPixelVCalReader") << " There are " << pDD->detUnits().size() << " modules" << std::endl;

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
  edm::LogPrint("SiPixelVCalReader") << std::setw(12) << "detid" << std::setw(8) << "subdet" << std::setw(8) << "layer"
                                     << std::setw(8) << "disk" << std::setw(14) << "VCal slope" << std::setw(8)
                                     << "offset" << std::endl;
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
    edm::LogPrint("SiPixelVCalReader") << std::setw(12) << detid << std::setw(8) << subdet << std::setw(8) << layer
                                       << std::setw(8) << disk << std::setw(14) << slope << std::setw(8) << offset
                                       << std::endl;
    // edm::LogPrint("SiPixelVCalReader") << "detid " << detid << ", subdet " << subdet << ", layer " <<
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
DEFINE_FWK_MODULE(SiPixelVCalReader);
