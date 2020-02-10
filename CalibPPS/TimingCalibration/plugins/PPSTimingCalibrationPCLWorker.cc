/****************************************************************************
 *
 * This is a part of PPS offline software.
 * Authors:
 *   Edoardo Bossini
 *   Piotr Maciej Cwiklicki
 *   Laurent Forthomme
 *
 ****************************************************************************/

#include "DQMServices/Core/interface/DQMGlobalEDAnalyzer.h"
#include "DQMServices/Core/interface/DQMStore.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESWatcher.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "Geometry/VeryForwardGeometryBuilder/interface/CTPPSGeometry.h"
#include "Geometry/Records/interface/VeryForwardRealGeometryRecord.h"

#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/CTPPSDetId/interface/CTPPSDiamondDetId.h"
#include "DataFormats/CTPPSReco/interface/CTPPSDiamondRecHit.h"
#include "DataFormats/CTPPSReco/interface/CTPPSPixelLocalTrack.h"

#include "CalibPPS/TimingCalibration/interface/TimingCalibrationStruct.h"

//------------------------------------------------------------------------------

class PPSTimingCalibrationPCLWorker : public DQMGlobalEDAnalyzer<TimingCalibrationHistograms> {
public:
  explicit PPSTimingCalibrationPCLWorker(const edm::ParameterSet&);

  void dqmAnalyze(const edm::Event&, const edm::EventSetup&, const TimingCalibrationHistograms&) const override;

  static void fillDescriptions(edm::ConfigurationDescriptions&);

private:
  void bookHistograms(DQMStore::IBooker&, const edm::Run&, const edm::EventSetup&, TimingCalibrationHistograms&) const override;

  edm::ESWatcher<CTPPSGeometry> geomWatcher_;
  edm::EDGetTokenT<edm::DetSetVector<CTPPSDiamondRecHit>> diamondRecHitToken_;
  edm::EDGetTokenT<edm::DetSetVector<CTPPSPixelLocalTrack>> pixelTrackToken_;
  const std::string dqmDir_;
  struct CTPPSPixelSelection {
    int minPixTracks, maxPixTracks;
  };
  std::map<CTPPSPixelDetId,CTPPSPixelSelection> pixelPotSel_;
};

//------------------------------------------------------------------------------

PPSTimingCalibrationPCLWorker::PPSTimingCalibrationPCLWorker(const edm::ParameterSet& iConfig)
  :diamondRecHitToken_(consumes<edm::DetSetVector<CTPPSDiamondRecHit>>(iConfig.getParameter<edm::InputTag>("diamondRecHitTag"))),
   pixelTrackToken_(consumes<edm::DetSetVector<CTPPSPixelLocalTrack>>(iConfig.getParameter<edm::InputTag>("pixelTrackTag"))),
   dqmDir_(iConfig.getParameter<std::string>("dqmDir")) {
  for (const auto& pix_pot : iConfig.getParameter<std::vector<edm::ParameterSet>>("pixelPotSelection"))
    pixelPotSel_[CTPPSPixelDetId(pix_pot.getParameter<unsigned int>("potId"))] = CTPPSPixelSelection{
      pix_pot.getParameter<int>("minPixelTracks"),
      pix_pot.getParameter<int>("maxPixelTracks")};
}

//------------------------------------------------------------------------------

void PPSTimingCalibrationPCLWorker::bookHistograms(DQMStore::IBooker& iBooker, const edm::Run& iRun, const edm::EventSetup& iSetup, TimingCalibrationHistograms& iHists) const
{
  iBooker.cd();
  iBooker.setCurrentFolder(dqmDir_);
  std::string ch_name;

  edm::ESHandle<CTPPSGeometry> hGeom;
  iSetup.get<VeryForwardRealGeometryRecord>().get(hGeom);
  for (auto it = hGeom->beginSensor(); it != hGeom->endSensor(); ++it) {
    CTPPSDetId detid(it->first);
    try {
      CTPPSDiamondDetId ddetid(detid);
      std::cout << ">>> " << ddetid << std::endl;
    } catch (const cms::Exception&) { continue; }
  }
  //FIXME use geometry ESHandle
  for (unsigned short arm = 0; arm < 2; ++arm) {
    for (unsigned short st = 0; st < 2; ++st) {
      for (unsigned short pl = 0; pl < 4; ++pl) {
        for (unsigned short ch = 0; ch < 12; ++ch) {
          const CTPPSDiamondDetId detid(arm, st, 6, pl, ch); //FIXME RP?
          detid.channelName(ch_name);
          iHists.leadingTime[detid.rawId()] = iBooker.book1D("t_"+ch_name, ch_name+";t (ns);Entries", 1200, -60., 60.);
          iHists.leadingTimeVsToT[detid.rawId()] = iBooker.book2D("tvstot_"+ch_name, ch_name+";ToT (ns);t (ns)", 240, 0., 60., 450, -20., 25.);
        } // loop over channels
      } // loop over arms
    } // loop over stations
  } // loop over arms
}

//------------------------------------------------------------------------------

void PPSTimingCalibrationPCLWorker::dqmAnalyze(const edm::Event& iEvent, const edm::EventSetup& iSetup, const TimingCalibrationHistograms& iHists) const
{
  // first unpack the pixel tracks information to ensure event quality
  edm::Handle<edm::DetSetVector<CTPPSPixelLocalTrack>> dsv_pixtrks;
  iEvent.getByToken(pixelTrackToken_, dsv_pixtrks);
  // require at least pixel tracks in the event before further analysis
  if (dsv_pixtrks->empty()) {
    edm::LogWarning("PPSTimingCalibrationPCLWorker:dqmAnalyze") << "No pixel tracks retrieved from the event content.";
    return;
  }
  // check the per-pot tracks multiplicity
  std::map<CTPPSPixelDetId,unsigned short> m_pixtrks_mult;
  for (const auto& ds_pixtrks : *dsv_pixtrks) {
    const CTPPSPixelDetId detid(ds_pixtrks.detId());
    if (pixelPotSel_.count(detid) == 0)
      continue; // no selection defined, discarding this pot
    for (const auto& track : ds_pixtrks)
      if (track.isValid())
        m_pixtrks_mult[detid]++;
  }
  std::array<bool,2> pass_pix_sel{{true, true}}; // enough but not too much tracks were found for this pot
  for (const auto& mult_vs_pot : m_pixtrks_mult) {
    const auto& pix_sel = pixelPotSel_.at(mult_vs_pot.first);
    if (pix_sel.minPixTracks < 0 || mult_vs_pot.second >= pix_sel.minPixTracks)
      pass_pix_sel[mult_vs_pot.first.arm()] = false;
    if (pix_sel.maxPixTracks < 0 || mult_vs_pot.second <= pix_sel.maxPixTracks)
      pass_pix_sel[mult_vs_pot.first.arm()] = false;
  }
  if (!pass_pix_sel[0] || !pass_pix_sel[1]) {
    LogDebug("PPSTimingCalibrationPCLWorker:dqmAnalyze") << "Event not passing pixel selection";
    return;
  }

  // then extract the rechits information for later processing
  edm::Handle<edm::DetSetVector<CTPPSDiamondRecHit>> dsv_rechits;
  iEvent.getByToken(diamondRecHitToken_, dsv_rechits);
  // ensure timing detectors rechits are found in the event content
  if (dsv_rechits->empty()) {
    edm::LogWarning("PPSTimingCalibrationPCLWorker:dqmAnalyze") << "No rechits retrieved from the event content.";
    return;
  }
  for (const auto& ds_rechits : *dsv_rechits) {
    const CTPPSDiamondDetId detid(ds_rechits.detId());
    if (!pass_pix_sel.at(detid.arm()))
      continue;
    if (iHists.leadingTimeVsToT.count(detid.rawId()) == 0) {
      edm::LogWarning("PPSTimingCalibrationPCLWorker:dqmAnalyze") << "Pad with detId=" << detid << " is not set to be monitored.";
      continue;
    }
    for (const auto& rechit : ds_rechits) {
      // skip invalid rechits
      if (rechit.time() == 0. || rechit.toT() < 0.)
        continue;
      iHists.leadingTime.at(detid.rawId())->Fill(rechit.time());
      iHists.leadingTimeVsToT.at(detid.rawId())->Fill(rechit.toT(), rechit.time());
    }
  }
}

//------------------------------------------------------------------------------

void PPSTimingCalibrationPCLWorker::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("diamondRecHitTag", edm::InputTag("ctppsDiamondRecHits"))
    ->setComment("input tag for the PPS diamond detectors rechits");
  desc.add<edm::InputTag>("pixelTrackTag", edm::InputTag("ctppsPixelLocalTracks"))
    ->setComment("input tag for the PPS pixel tracks");
  desc.add<std::string>("dqmDir", "AlCaReco/PPSTimingCalibrationPCL")
    ->setComment("output path for the various DQM plots");

  edm::ParameterSetDescription desc_pixpotcuts;
  desc_pixpotcuts.add<unsigned int>("potId", 0);
  desc_pixpotcuts.add<int>("minPixelTracks", -1)
    ->setComment("minimal pixel tracks multiplicity");
  desc_pixpotcuts.add<int>("maxPixelTracks", -1)
    ->setComment("maximal pixel tracks multiplicity for shower rejection");

  std::vector<edm::ParameterSet> potsDefaults;
  std::vector<CTPPSPixelDetId> pots_ids = {
    CTPPSPixelDetId(1, 0), CTPPSPixelDetId(1, 1)
  };
  for (const auto& pot : pots_ids) {
    edm::ParameterSet pot_ps;
    pot_ps.addParameter<unsigned int>("potId", pot.rawId());
    pot_ps.addParameter<int>("minPixelTracks", 1);
    pot_ps.addParameter<int>("maxPixelTracks", 6);
    potsDefaults.emplace_back(pot_ps);
  }
  desc.addVPSet("pixelPotSelection", desc_pixpotcuts, potsDefaults);

  descriptions.addWithDefaultLabel(desc);
}

DEFINE_FWK_MODULE(PPSTimingCalibrationPCLWorker);
