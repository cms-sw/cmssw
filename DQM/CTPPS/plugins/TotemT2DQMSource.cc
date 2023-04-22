/****************************************************************************
 *
 * This is a part of TOTEM offline software.
 * Author:
 *   Laurent Forthomme
 *   Arkadiusz Cwikla
 *
 ****************************************************************************/

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/DQMStore.h"

#include "DataFormats/Common/interface/DetSetVectorNew.h"

#include "DataFormats/CTPPSDetId/interface/TotemT2DetId.h"
#include "DataFormats/TotemReco/interface/TotemT2Digi.h"
#include "DataFormats/TotemReco/interface/TotemT2RecHit.h"

#include "Geometry/Records/interface/TotemGeometryRcd.h"
#include "DQM/CTPPS/interface/TotemT2Segmentation.h"

#include <string>

class TotemT2DQMSource : public DQMEDAnalyzer {
public:
  TotemT2DQMSource(const edm::ParameterSet&);
  ~TotemT2DQMSource() override = default;

protected:
  void dqmBeginRun(const edm::Run&, const edm::EventSetup&) override;
  void bookHistograms(DQMStore::IBooker&, const edm::Run&, const edm::EventSetup&) override;
  void analyze(const edm::Event&, const edm::EventSetup&) override;

private:
  void fillActivePlanes(std::unordered_map<unsigned int, std::set<unsigned int>>&, const TotemT2DetId&);
  void fillTriggerBitset(const TotemT2DetId&);
  void clearTriggerBitset();
  bool areChannelsTriggered(const TotemT2DetId&);
  void bookErrorFlagsHistogram(DQMStore::IBooker&);
  void fillErrorFlagsHistogram(const TotemT2Digi&);
  void fillEdges(const TotemT2Digi&, const TotemT2DetId&);
  void fillToT(const TotemT2RecHit&, const TotemT2DetId&);

  const edm::ESGetToken<TotemGeometry, TotemGeometryRcd> geometryToken_;
  const edm::EDGetTokenT<edmNew::DetSetVector<TotemT2Digi>> digiToken_;
  const edm::EDGetTokenT<edmNew::DetSetVector<TotemT2RecHit>> rechitToken_;

  std::unique_ptr<TotemT2Segmentation> segm_;

  static constexpr double T2_BIN_WIDTH_NS_ = 25. / 4;
  MonitorElement* HPTDCErrorFlags_2D_ = nullptr;

  const unsigned int nbinsx_, nbinsy_;
  const unsigned int windowsNum_;

  struct SectorPlots {
    MonitorElement* activePlanes = nullptr;
    MonitorElement* activePlanesCount = nullptr;

    MonitorElement* triggerEmulator = nullptr;
    std::bitset<(TotemT2DetId::maxPlane + 1) * (TotemT2DetId::maxChannel + 1)> hitTilesArray;
    static const unsigned int MINIMAL_TRIGGER = 3;

    MonitorElement *leadingEdge = nullptr, *trailingEdge = nullptr, *timeOverTreshold = nullptr;

    SectorPlots() = default;
    SectorPlots(
        DQMStore::IBooker& ibooker, unsigned int id, unsigned int nbinsx, unsigned int nbinsy, unsigned int windowsNum);
  };

  struct PlanePlots {
    MonitorElement* digisMultiplicity = nullptr;
    MonitorElement* rechitMultiplicity = nullptr;

    PlanePlots() = default;
    PlanePlots(DQMStore::IBooker& ibooker, unsigned int id, unsigned int nbinsx, unsigned int nbinsy);
  };

  std::unordered_map<unsigned int, SectorPlots> sectorPlots_;
  std::unordered_map<unsigned int, PlanePlots> planePlots_;
};

TotemT2DQMSource::SectorPlots::SectorPlots(
    DQMStore::IBooker& ibooker, unsigned int id, unsigned int nbinsx, unsigned int nbinsy, unsigned int windowsNum) {
  std::string title, path;

  TotemT2DetId(id).armName(path, TotemT2DetId::nPath);
  ibooker.setCurrentFolder(path);

  TotemT2DetId(id).armName(title, TotemT2DetId::nFull);

  activePlanes = ibooker.book1D("active planes", title + " which planes are active;plane number", 8, -0.5, 7.5);

  activePlanesCount = ibooker.book1D(
      "number of active planes", title + " how many planes are active;number of active planes", 9, -0.5, 8.5);

  triggerEmulator = ibooker.book2DD("trigger emulator",
                                    title + " trigger emulator;arbitrary unit;arbitrary unit",
                                    nbinsx,
                                    -0.5,
                                    double(nbinsx) - 0.5,
                                    nbinsy,
                                    -0.5,
                                    double(nbinsy) - 0.5);
  leadingEdge = ibooker.book1D(
      "leading edge", title + " leading edge (DIGIs); leading edge (ns)", 25 * windowsNum, 0, 25 * windowsNum);
  trailingEdge = ibooker.book1D(
      "trailing edge", title + " trailing edge (DIGIs); trailing edge (ns)", 25 * windowsNum, 0, 25 * windowsNum);

  timeOverTreshold = ibooker.book1D(
      "time over threshold", title + " time over threshold (rechit);time over threshold (ns)", 250, -25, 100);
}

TotemT2DQMSource::PlanePlots::PlanePlots(DQMStore::IBooker& ibooker,
                                         unsigned int id,
                                         unsigned int nbinsx,
                                         unsigned int nbinsy) {
  std::string title, path;
  TotemT2DetId(id).planeName(title, TotemT2DetId::nFull);
  TotemT2DetId(id).planeName(path, TotemT2DetId::nPath);
  ibooker.setCurrentFolder(path);

  digisMultiplicity = ibooker.book2DD("digis multiplicity",
                                      title + " digis multiplicity;arbitrary unit;arbitrary unit",
                                      nbinsx,
                                      -0.5,
                                      double(nbinsx) - 0.5,
                                      nbinsy,
                                      -0.5,
                                      double(nbinsy) - 0.5);
  rechitMultiplicity = ibooker.book2DD("rechits multiplicity",
                                       title + " rechits multiplicity;x;y",
                                       nbinsx,
                                       -0.5,
                                       double(nbinsx) - 0.5,
                                       nbinsy,
                                       -0.5,
                                       double(nbinsy) - 0.5);
}

TotemT2DQMSource::TotemT2DQMSource(const edm::ParameterSet& iConfig)
    : geometryToken_(esConsumes<TotemGeometry, TotemGeometryRcd, edm::Transition::BeginRun>()),
      digiToken_(consumes<edmNew::DetSetVector<TotemT2Digi>>(iConfig.getParameter<edm::InputTag>("digisTag"))),
      rechitToken_(consumes<edmNew::DetSetVector<TotemT2RecHit>>(iConfig.getParameter<edm::InputTag>("rechitsTag"))),
      nbinsx_(iConfig.getParameter<unsigned int>("nbinsx")),
      nbinsy_(iConfig.getParameter<unsigned int>("nbinsy")),
      windowsNum_(iConfig.getParameter<unsigned int>("windowsNum")) {}

void TotemT2DQMSource::dqmBeginRun(const edm::Run&, const edm::EventSetup&) {}

void TotemT2DQMSource::bookHistograms(DQMStore::IBooker& ibooker, const edm::Run&, const edm::EventSetup& iSetup) {
  ibooker.cd();
  ibooker.setCurrentFolder("TotemT2");

  bookErrorFlagsHistogram(ibooker);

  for (unsigned int arm = 0; arm <= CTPPSDetId::maxArm; ++arm) {
    for (unsigned int pl = 0; pl <= TotemT2DetId::maxPlane; ++pl) {
      const TotemT2DetId detid(arm, pl, 0);
      const TotemT2DetId planeId(detid.planeId());
      planePlots_[planeId] = PlanePlots(ibooker, planeId, nbinsx_, nbinsy_);
    }
    const TotemT2DetId detid(arm, 0, 0);
    const TotemT2DetId secId(detid.armId());
    sectorPlots_[secId] = SectorPlots(ibooker, secId, nbinsx_, nbinsy_, windowsNum_);
  }

  // build a segmentation helper for the size of histograms previously booked
  segm_ = std::make_unique<TotemT2Segmentation>(iSetup.getData(geometryToken_), nbinsx_, nbinsy_);
}

void TotemT2DQMSource::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  // fill digis information
  for (const auto& ds_digis : iEvent.get(digiToken_)) {
    const TotemT2DetId detid(ds_digis.detId());
    const TotemT2DetId planeId(detid.planeId());
    for (const auto& digi : ds_digis) {
      segm_->fill(planePlots_[planeId].digisMultiplicity->getTH2D(), detid);
      fillTriggerBitset(detid);
      fillErrorFlagsHistogram(digi);
      fillEdges(digi, detid);
    }
  }

  // fill rechits information
  std::unordered_map<unsigned int, std::set<unsigned int>> planes;
  for (const auto& ds_rechits : iEvent.get(rechitToken_)) {
    const TotemT2DetId detid(ds_rechits.detId());
    const TotemT2DetId planeId(detid.planeId());
    for (const auto& rechit : ds_rechits) {
      segm_->fill(planePlots_[planeId].rechitMultiplicity->getTH2D(), detid);
      fillToT(rechit, detid);
      fillActivePlanes(planes, detid);
    }
  }

  for (const auto& plt : sectorPlots_)
    plt.second.activePlanesCount->Fill(planes[plt.first].size());

  for (unsigned short arm = 0; arm <= CTPPSDetId::maxArm; ++arm)
    for (unsigned short plane = 0; plane <= 1; ++plane)
      for (unsigned short id = 0; id <= TotemT2DetId::maxChannel; ++id) {
        const TotemT2DetId detid(arm, plane, id);
        if (areChannelsTriggered(detid)) {
          const TotemT2DetId secId(detid.armId());
          segm_->fill(sectorPlots_[secId].triggerEmulator->getTH2D(), detid);
        }
      }

  clearTriggerBitset();
}

void TotemT2DQMSource::fillActivePlanes(std::unordered_map<unsigned int, std::set<unsigned int>>& planes,
                                        const TotemT2DetId& detid) {
  const TotemT2DetId secId(detid.armId());
  unsigned short pl = detid.plane();

  planes[secId].insert(pl);

  sectorPlots_[secId].activePlanes->Fill(pl);
}

void TotemT2DQMSource::fillTriggerBitset(const TotemT2DetId& detid) {
  const TotemT2DetId secId(detid.armId());
  unsigned short pl = detid.plane();
  unsigned short ch = detid.channel();
  sectorPlots_[secId].hitTilesArray[4 * pl + ch] = true;
}

void TotemT2DQMSource::clearTriggerBitset() {
  for (auto& sectorPlot : sectorPlots_)
    sectorPlot.second.hitTilesArray.reset();
}

bool TotemT2DQMSource::areChannelsTriggered(const TotemT2DetId& detid) {
  unsigned int channel = detid.channel();

  // prepare mask
  std::bitset<(TotemT2DetId::maxPlane + 1) * (TotemT2DetId::maxChannel + 1)> mask;
  // check if plane is even or not
  unsigned int pl = detid.plane() % 2 == 0 ? 0 : 1;
  // set only even or only odd plane bits for this channel
  for (; pl <= TotemT2DetId::maxPlane; pl += 2)
    mask[4 * pl + channel] = true;
  const TotemT2DetId secId(detid.armId());
  // check how many masked channels were hit
  unsigned int triggeredChannelsNumber = (mask & sectorPlots_[secId].hitTilesArray).count();

  return triggeredChannelsNumber >= SectorPlots::MINIMAL_TRIGGER;
}

void TotemT2DQMSource::bookErrorFlagsHistogram(DQMStore::IBooker& ibooker) {
  HPTDCErrorFlags_2D_ = ibooker.book2D("HPTDC Errors", " HPTDC Errors?", 8, -0.5, 7.5, 2, -0.5, 1.5);
  for (unsigned short error_index = 1; error_index <= 8; ++error_index)
    HPTDCErrorFlags_2D_->setBinLabel(error_index, "Flag " + std::to_string(error_index));

  int tmpIndex = 0;
  HPTDCErrorFlags_2D_->setBinLabel(++tmpIndex, "some id 0", /* axis */ 2);
  HPTDCErrorFlags_2D_->setBinLabel(++tmpIndex, "some id 1", /* axis */ 2);
}

void TotemT2DQMSource::fillErrorFlagsHistogram(const TotemT2Digi& digi) {
  // placeholder for error hitogram filling
  (void)digi;
}

void TotemT2DQMSource::fillEdges(const TotemT2Digi& digi, const TotemT2DetId& detid) {
  const TotemT2DetId secId(detid.armId());
  sectorPlots_[secId].leadingEdge->Fill(T2_BIN_WIDTH_NS_ * digi.leadingEdge());
  sectorPlots_[secId].trailingEdge->Fill(T2_BIN_WIDTH_NS_ * digi.trailingEdge());
}

void TotemT2DQMSource::fillToT(const TotemT2RecHit& rechit, const TotemT2DetId& detid) {
  const TotemT2DetId secId(detid.armId());
  sectorPlots_[secId].timeOverTreshold->Fill(rechit.toT());
}

DEFINE_FWK_MODULE(TotemT2DQMSource);
