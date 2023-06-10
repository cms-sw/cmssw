/****************************************************************************
 *
 * This is a part of TOTEM offline software.
 * Authors:
 *   Laurent Forthomme
 *   Arkadiusz Cwikla
 *   Fredrik Oljemark
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

#include "Geometry/Records/interface/TotemGeometryRcd.h"
#include "DQM/CTPPS/interface/TotemT2Segmentation.h"

#include <string>
#include <bitset>

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
  void fillErrorFlagsHistogram(const TotemT2Digi&, const TotemT2DetId&);
  void fillEdges(const TotemT2Digi&, const TotemT2DetId&);
  void fillBX(const TotemT2Digi&, const TotemT2DetId&, const int);
  void fillToT(const TotemT2Digi&, const TotemT2DetId&);
  void fillFlags(const TotemT2Digi&, const TotemT2DetId&);

  const edm::EDGetTokenT<edmNew::DetSetVector<TotemT2Digi>> digiToken_;

  std::unique_ptr<TotemT2Segmentation> segm_;

  static constexpr double T2_BIN_WIDTH_NS_ = 25. / 4;
  MonitorElement* totemT2ErrorFlags_2D_ = nullptr;

  enum evFlag { t2TE = 0, t2LE, t2MT, t2ML };

  const unsigned int nbinsx_, nbinsy_;
  const unsigned int windowsNum_;

  struct SectorPlots {
    MonitorElement* activePlanes = nullptr;
    MonitorElement* activePlanesCount = nullptr;
    MonitorElement* activityPerBX = nullptr;

    MonitorElement* triggerEmulator = nullptr;
    std::bitset<(TotemT2DetId::maxPlane + 1) * (TotemT2DetId::maxChannel + 1)> hitTilesArray;
    static const unsigned int MINIMAL_TRIGGER = 3;

    MonitorElement *leadingEdge = nullptr, *trailingEdge = nullptr, *timeOverTreshold = nullptr, *eventFlags = nullptr;

    SectorPlots() = default;
    SectorPlots(
        DQMStore::IBooker& ibooker, unsigned int id, unsigned int nbinsx, unsigned int nbinsy, unsigned int windowsNum);
  };

  struct PlanePlots {
    MonitorElement* digisMultiplicity = nullptr;
    MonitorElement* rechitMultiplicity = nullptr;
    MonitorElement* eventFlagsPl = nullptr;

    PlanePlots() = default;
    PlanePlots(DQMStore::IBooker& ibooker, unsigned int id, unsigned int nbinsx, unsigned int nbinsy);
  };
  struct ChannelPlots {
    MonitorElement* leadingEdgeCh = nullptr;
    MonitorElement* trailingEdgeCh = nullptr;
    MonitorElement* timeOverTresholdCh = nullptr;
    MonitorElement* eventFlagsCh = nullptr;
    MonitorElement* activityPerBXCh = nullptr;

    ChannelPlots() = default;
    ChannelPlots(DQMStore::IBooker& ibooker, unsigned int id, unsigned int windowsNum);
  };

  std::unordered_map<unsigned int, SectorPlots> sectorPlots_;
  std::unordered_map<unsigned int, PlanePlots> planePlots_;
  std::unordered_map<unsigned int, ChannelPlots> channelPlots_;
};

TotemT2DQMSource::SectorPlots::SectorPlots(
    DQMStore::IBooker& ibooker, unsigned int id, unsigned int nbinsx, unsigned int nbinsy, unsigned int windowsNum) {
  std::string title, path;

  TotemT2DetId(id).armName(path, TotemT2DetId::nPath);
  ibooker.setCurrentFolder(path);

  TotemT2DetId(id).armName(title, TotemT2DetId::nFull);

  activePlanes = ibooker.book1D("active planes", title + " which planes are active (digi);plane number", 8, -0.5, 7.5);

  activePlanesCount = ibooker.book1D(
      "number of active planes (digi)", title + " how many planes are active;number of active planes", 9, -0.5, 8.5);

  activityPerBX = ibooker.book1D("activity per BX CMS", title + " Activity per BX;Event.BX", 3600, -1.5, 3598. + 0.5);

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
      "time over threshold", title + " time over threshold (digi);time over threshold (ns)", 500, -50, 200);

  eventFlags = ibooker.book1D(
      "event flags", title + " event flags (digi);Event flags (TE/LE valid, TE/LE multiple)", 4, -0.5, 3.5);

  for (unsigned short flag_index = 1; flag_index <= 4; ++flag_index)
    eventFlags->setBinLabel(flag_index, "Flag " + std::to_string(flag_index));
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

  eventFlagsPl = ibooker.book1D(
      "event flags", title + " event flags (digi);Event flags (TE/LE valid, TE/LE multiple)", 4, -0.5, 3.5);

  for (unsigned short flag_index = 1; flag_index <= 4; ++flag_index)
    eventFlagsPl->setBinLabel(flag_index, "Flag " + std::to_string(flag_index));
}

TotemT2DQMSource::ChannelPlots::ChannelPlots(DQMStore::IBooker& ibooker, unsigned int id, unsigned int windowsNum) {
  std::string title, path;
  TotemT2DetId(id).channelName(title, TotemT2DetId::nFull);
  TotemT2DetId(id).channelName(path, TotemT2DetId::nPath);
  ibooker.setCurrentFolder(path);

  leadingEdgeCh = ibooker.book1D(
      "leading edge", title + " leading edge (DIGIs); leading edge (ns)", 25 * windowsNum, 0, 25 * windowsNum);
  trailingEdgeCh = ibooker.book1D(
      "trailing edge", title + " trailing edge (DIGIs); trailing edge (ns)", 25 * windowsNum, 0, 25 * windowsNum);

  timeOverTresholdCh = ibooker.book1D(
      "time over threshold", title + " time over threshold (digi);time over threshold (ns)", 500, -50, 200);

  eventFlagsCh = ibooker.book1D(
      "event flags", title + " event flags (digi);Event flags (TE/LE valid, TE/LE multiple)", 4, -0.5, 3.5);

  activityPerBXCh = ibooker.book1D("activity per BX", title + " Activity per BX;Event.BX", 1000, -1.5, 998. + 0.5);

  for (unsigned short flag_index = 1; flag_index <= 4; ++flag_index)
    eventFlagsCh->setBinLabel(flag_index, "Flag " + std::to_string(flag_index));
}

TotemT2DQMSource::TotemT2DQMSource(const edm::ParameterSet& iConfig)
    : digiToken_(consumes<edmNew::DetSetVector<TotemT2Digi>>(iConfig.getParameter<edm::InputTag>("digisTag"))),
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
      for (unsigned int ch = 0; ch <= TotemT2DetId::maxChannel; ++ch) {
        const TotemT2DetId detidCh(arm, pl, ch);
        channelPlots_[detidCh] = ChannelPlots(ibooker, detidCh, windowsNum_);
      }
    }
    const TotemT2DetId detid(arm, 0, 0);
    const TotemT2DetId secId(detid.armId());
    sectorPlots_[secId] = SectorPlots(ibooker, secId, nbinsx_, nbinsy_, windowsNum_);
  }

  // build a segmentation helper for the size of histograms previously booked
  segm_ = std::make_unique<TotemT2Segmentation>(nbinsx_, nbinsy_);
}

void TotemT2DQMSource::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  // fill digis information
  std::unordered_map<unsigned int, std::set<unsigned int>> planes;

  for (const auto& ds_digis : iEvent.get(digiToken_)) {
    if (!ds_digis.empty()) {
      const TotemT2DetId detid(ds_digis.detId());
      const TotemT2DetId planeId(detid.planeId());
      for (const auto& digi : ds_digis) {
        segm_->fill(planePlots_[planeId].digisMultiplicity->getTH2D(), detid);
        fillTriggerBitset(detid);
        fillErrorFlagsHistogram(digi, detid);
        fillEdges(digi, detid);
        fillToT(digi, detid);
        fillFlags(digi, detid);
        int bx = iEvent.bunchCrossing();
        fillBX(digi, detid, bx);
        if (digi.hasLE() && digi.hasTE())  //good digi
          fillActivePlanes(planes, detid);
      }
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

  // fill rechits information

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
  totemT2ErrorFlags_2D_ = ibooker.book2D("nt2 readout flags", " nt2 readout flags", 4, -0.5, 3.5, 2, -0.5, 1.5);
  for (unsigned short error_index = 1; error_index <= 4; ++error_index)
    totemT2ErrorFlags_2D_->setBinLabel(error_index, "Flag " + std::to_string(error_index));

  int tmpIndex = 0;
  totemT2ErrorFlags_2D_->setBinLabel(++tmpIndex, "arm 4-5", /* axis */ 2);
  totemT2ErrorFlags_2D_->setBinLabel(++tmpIndex, "arm 5-6", /* axis */ 2);
}

void TotemT2DQMSource::fillErrorFlagsHistogram(const TotemT2Digi& digi, const TotemT2DetId& detid) {
  // readout flags histogram filling
  for (unsigned int i = 0; i < 4; i++) {
    if (digi.status() & (1 << i))
      totemT2ErrorFlags_2D_->Fill(i + 0.0, detid.arm() + 0.0);
  }
}

void TotemT2DQMSource::fillEdges(const TotemT2Digi& digi, const TotemT2DetId& detid) {
  const TotemT2DetId secId(detid.armId());
  sectorPlots_[secId].leadingEdge->Fill(T2_BIN_WIDTH_NS_ * digi.leadingEdge());
  sectorPlots_[secId].trailingEdge->Fill(T2_BIN_WIDTH_NS_ * digi.trailingEdge());
  channelPlots_[detid].leadingEdgeCh->Fill(T2_BIN_WIDTH_NS_ * digi.leadingEdge());
  channelPlots_[detid].trailingEdgeCh->Fill(T2_BIN_WIDTH_NS_ * digi.trailingEdge());
}

void TotemT2DQMSource::fillBX(const TotemT2Digi& digi, const TotemT2DetId& detid, const int bx) {
  const TotemT2DetId secId(detid.armId());
  if (digi.hasLE()) {
    sectorPlots_[secId].activityPerBX->Fill(bx);
    channelPlots_[detid].activityPerBXCh->Fill(bx);
  }
}

void TotemT2DQMSource::fillToT(const TotemT2Digi& digi, const TotemT2DetId& detid) {
  const TotemT2DetId secId(detid.armId());

  const int t_lead = digi.leadingEdge(), t_trail = digi.trailingEdge();
  // don't skip no-edge digis
  double toT = 0.;
  if (digi.hasLE() && digi.hasTE()) {
    toT = (t_trail - t_lead) * T2_BIN_WIDTH_NS_;  // in ns
  }

  sectorPlots_[secId].timeOverTreshold->Fill(toT);
  channelPlots_[detid].timeOverTresholdCh->Fill(toT);
}

void TotemT2DQMSource::fillFlags(const TotemT2Digi& digi, const TotemT2DetId& detid) {
  const TotemT2DetId secId(detid.armId());
  const TotemT2DetId planeId(detid.planeId());
  if (digi.hasTE()) {
    sectorPlots_[secId].eventFlags->Fill(t2TE + 0.0);
    planePlots_[planeId].eventFlagsPl->Fill(t2TE + 0.0);
    channelPlots_[detid].eventFlagsCh->Fill(t2TE + 0.0);
  }

  if (digi.hasLE()) {
    sectorPlots_[secId].eventFlags->Fill(t2LE + 0.0);
    planePlots_[planeId].eventFlagsPl->Fill(t2LE + 0.0);
    channelPlots_[detid].eventFlagsCh->Fill(t2LE + 0.0);
  }

  if (digi.hasManyTE()) {
    sectorPlots_[secId].eventFlags->Fill(t2MT + 0.0);
    planePlots_[planeId].eventFlagsPl->Fill(t2MT + 0.0);
    channelPlots_[detid].eventFlagsCh->Fill(t2MT + 0.0);
  }

  if (digi.hasManyLE()) {
    sectorPlots_[secId].eventFlags->Fill(t2ML + 0.0);
    planePlots_[planeId].eventFlagsPl->Fill(t2ML + 0.0);
    channelPlots_[detid].eventFlagsCh->Fill(t2ML + 0.0);
  }
}

DEFINE_FWK_MODULE(TotemT2DQMSource);
