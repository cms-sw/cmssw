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
#include "DataFormats/CTPPSDigi/interface/CTPPSDiamondDigi.h"

#include "CalibPPS/TimingCalibration/interface/TimingCalibrationStruct.h"

//------------------------------------------------------------------------------

class PPSTimingCalibrationPCLWorker : public DQMGlobalEDAnalyzer<TimingCalibrationHistograms> {
public:
  explicit PPSTimingCalibrationPCLWorker(const edm::ParameterSet&);

  void dqmAnalyze(const edm::Event&, const edm::EventSetup&, const TimingCalibrationHistograms&) const override;

  static void fillDescriptions(edm::ConfigurationDescriptions&);

private:
  void bookHistograms(DQMStore::IBooker&, const edm::Run&, const edm::EventSetup&, TimingCalibrationHistograms&) const override;

  edm::EDGetTokenT<edm::DetSetVector<CTPPSDiamondDigi>> digiToken_;
  const std::string dqmDir_;
  const double ts_to_ns_;
};

//------------------------------------------------------------------------------

PPSTimingCalibrationPCLWorker::PPSTimingCalibrationPCLWorker(const edm::ParameterSet& iConfig)
  :digiToken_(consumes<edm::DetSetVector<CTPPSDiamondDigi>>(iConfig.getParameter<edm::InputTag>("digiTag"))),
   dqmDir_(iConfig.getParameter<std::string>("dqmDir")),
   ts_to_ns_(iConfig.getParameter<double>("timeSliceNs")) {
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
  edm::Handle<edm::DetSetVector<CTPPSDiamondDigi>> dsv_digis;
  iEvent.getByToken(digiToken_, dsv_digis);
  if (dsv_digis->empty()) {
    edm::LogWarning("PPSTimingCalibrationPCLWorker:dqmAnalyze") << "No digis retrieved from the event content.";
    return;
  }
  for (const auto& ds_digis : *dsv_digis) {
    const CTPPSDiamondDetId detid(ds_digis.detId());
    if (iHists.leadingTimeVsToT.count(detid.rawId()) == 0) {
      edm::LogWarning("PPSTimingCalibrationPCLWorker:dqmAnalyze") << "Pad with detId=" << detid << " is not set to be monitored.";
      continue;
    }
    for (const auto& digi : ds_digis) {
      const int t_lead = digi.leadingEdge(), t_trail = digi.trailingEdge();
      // skip invalid digis
      if (t_lead == 0 && t_trail == 0)
        continue;
      double tot = -1., ch_t = 0.;
      if (t_lead != 0 && t_trail != 0) { // skip digis with invalid ToT
        tot = (t_trail - t_lead) * ts_to_ns_;
        ch_t = (t_lead % 1024) * ts_to_ns_;
        iHists.leadingTime.at(detid.rawId())->Fill(ch_t); //FIXME
        iHists.leadingTimeVsToT.at(detid.rawId())->Fill(tot, ch_t);
      }
    }
  }
}

//------------------------------------------------------------------------------

void PPSTimingCalibrationPCLWorker::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("digiTag", edm::InputTag("ctppsDiamondRawToDigi", "TimingDiamond"))
    ->setComment("input tag for the PPS diamond detectors digis");
  desc.add<std::string>("dqmDir", "AlCaReco/PPSTimingCalibrationPCL")
    ->setComment("output path for the various DQM plots");
  desc.add<double>("timeSliceNs", 25./*ns*/ / 1024./*bins*/)
    ->setComment("conversion constant between HPTDC timing bin size and nanoseconds");
  descriptions.addWithDefaultLabel(desc);
}

DEFINE_FWK_MODULE(PPSTimingCalibrationPCLWorker);
