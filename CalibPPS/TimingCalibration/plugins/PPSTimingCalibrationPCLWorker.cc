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
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/CTPPSDetId/interface/CTPPSDiamondDetId.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/CTPPSReco/interface/CTPPSDiamondRecHit.h"

#include "CalibPPS/TimingCalibration/interface/TimingCalibrationStruct.h"

//------------------------------------------------------------------------------

class PPSTimingCalibrationPCLWorker : public DQMGlobalEDAnalyzer<TimingCalibrationHistograms> {
public:
  explicit PPSTimingCalibrationPCLWorker(const edm::ParameterSet&);

  void dqmAnalyze(const edm::Event&, const edm::EventSetup&, const TimingCalibrationHistograms&) const override;

  static void fillDescriptions(edm::ConfigurationDescriptions&);

private:
  void bookHistograms(DQMStore::IBooker&, const edm::Run&, const edm::EventSetup&, TimingCalibrationHistograms&) const override;

  edm::EDGetTokenT<edm::DetSetVector<CTPPSDiamondRecHit>> diamondRecHitToken_;
  const std::string dqmDir_;
};

//------------------------------------------------------------------------------

PPSTimingCalibrationPCLWorker::PPSTimingCalibrationPCLWorker(const edm::ParameterSet& iConfig)
  :diamondRecHitToken_(consumes<edm::DetSetVector<CTPPSDiamondRecHit>>(iConfig.getParameter<edm::InputTag>("diamondRecHitTag"))),
   dqmDir_(iConfig.getParameter<std::string>("dqmDir")) {
}

//------------------------------------------------------------------------------

void PPSTimingCalibrationPCLWorker::bookHistograms(DQMStore::IBooker& iBooker, const edm::Run& iRun, const edm::EventSetup& iSetup, TimingCalibrationHistograms& iHists) const
{
  iBooker.cd();
  iBooker.setCurrentFolder(dqmDir_);
  std::string ch_name;
  //FIXME use geometry ESHandle
  for (unsigned short arm = 0; arm < 2; ++arm) {
    for (unsigned short st = 0; st < 2; ++st) {
      for (unsigned short pl = 0; pl < 4; ++pl) {
        for (unsigned short ch = 0; ch < 12; ++ch) {
          const CTPPSDiamondDetId detid(arm, st, 0, pl, ch); //FIXME RP?
          detid.channelName(ch_name);
          iHists.leadingTime[detid.rawId()] = iBooker.book1D(Form("t_%s", ch_name.c_str()), Form("%s;t (ns);Entries", ch_name.c_str()), 1200, -60., 60.);
          iHists.leadingTimeVsToT[detid.rawId()] = iBooker.book2D(Form("tvstot_%s", ch_name.c_str()), Form("%s;ToT (ns);t (ns)", ch_name.c_str()), 240, 0., 60., 450, -20., 25.);
        } // loop over channels
      } // loop over arms
    } // loop over stations
  } // loop over arms
}

//------------------------------------------------------------------------------

void PPSTimingCalibrationPCLWorker::dqmAnalyze(const edm::Event& iEvent, const edm::EventSetup& iSetup, const TimingCalibrationHistograms& iHists) const
{
  edm::Handle<edm::DetSetVector<CTPPSDiamondRecHit>> dsv_rechits;
  iEvent.getByToken(diamondRecHitToken_, dsv_rechits);
  if (dsv_rechits->empty()) {
    edm::LogWarning("PPSTimingCalibrationPCLWorker:dqmAnalyze") << "No rechits retrieved from the event content.";
    return;
  }
  for (const auto& ds_rechits : *dsv_rechits) {
    const CTPPSDiamondDetId detid(ds_rechits.detId());
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
  desc.add<std::string>("dqmDir", "AlCaReco/PPSTimingCalibrationPCL")
    ->setComment("output path for the various DQM plots");
  descriptions.addWithDefaultLabel(desc);
}

DEFINE_FWK_MODULE(PPSTimingCalibrationPCLWorker);
