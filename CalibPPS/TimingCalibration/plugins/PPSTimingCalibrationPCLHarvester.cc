/****************************************************************************
 *
 * This is a part of PPS offline software.
 * Authors:
 *   Edoardo Bossini
 *   Piotr Maciej Cwiklicki
 *   Laurent Forthomme
 *
 ****************************************************************************/

#include "DQMServices/Core/interface/DQMEDHarvester.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "Geometry/VeryForwardGeometryBuilder/interface/CTPPSGeometry.h"
#include "Geometry/Records/interface/VeryForwardRealGeometryRecord.h"

#include "CalibPPS/TimingCalibration/interface/TimingCalibrationStruct.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"

#include "DataFormats/CTPPSDetId/interface/CTPPSDiamondDetId.h"
#include "CondFormats/PPSObjects/interface/PPSTimingCalibration.h"
#include "TFitResult.h"
//------------------------------------------------------------------------------

class PPSTimingCalibrationPCLHarvester : public DQMEDHarvester {
public:
  PPSTimingCalibrationPCLHarvester(const edm::ParameterSet&);
  void beginRun(const edm::Run&, const edm::EventSetup&) override;

  static void fillDescriptions(edm::ConfigurationDescriptions&);

private:
  void dqmEndJob(DQMStore::IBooker&, DQMStore::IGetter&) override;
  edm::ESGetToken<CTPPSGeometry, VeryForwardRealGeometryRecord> geomEsToken_;
  std::vector<CTPPSDiamondDetId> detids_;
  const std::string dqmDir_;
  const std::string formula_;
  const unsigned int min_entries_;
  const double threshold_fraction_of_max_;
  static constexpr double upper_limit_max_search_ = 20;
  static constexpr double upper_limit_range_search_ = 20;
  static constexpr double lower_limit_range_search_ = 8;
  static constexpr double resolution_ = 0.1;
  static constexpr double offset_ = 0.;
  TF1 interp_;
};

//------------------------------------------------------------------------------

PPSTimingCalibrationPCLHarvester::PPSTimingCalibrationPCLHarvester(const edm::ParameterSet& iConfig)
    : geomEsToken_(esConsumes<edm::Transition::BeginRun>()),
      dqmDir_(iConfig.getParameter<std::string>("dqmDir")),
      formula_(iConfig.getParameter<std::string>("formula")),
      min_entries_(iConfig.getParameter<unsigned int>("minEntries")),
      threshold_fraction_of_max_(iConfig.getParameter<double>("thresholdFractionOfMax")),
      interp_("interp", formula_.c_str(), 10.5, 25.) {
  // first ensure DB output service is available
  edm::Service<cond::service::PoolDBOutputService> poolDbService;
  if (!poolDbService.isAvailable())
    throw cms::Exception("PPSTimingCalibrationPCLHarvester") << "PoolDBService required";

  // constrain the min/max fit values
  interp_.SetParLimits(1, 9., 15.);
  interp_.SetParLimits(2, 0.2, 2.5);
}

//------------------------------------------------------------------------------

void PPSTimingCalibrationPCLHarvester::beginRun(const edm::Run& iRun, const edm::EventSetup& iSetup) {
  const auto& geom = iSetup.getData(geomEsToken_);
  for (auto it = geom.beginSensor(); it != geom.endSensor(); ++it) {
    if (!CTPPSDiamondDetId::check(it->first))
      continue;
    const CTPPSDiamondDetId detid(it->first);
    detids_.emplace_back(detid);
  }
}

//------------------------------------------------------------------------------

void PPSTimingCalibrationPCLHarvester::dqmEndJob(DQMStore::IBooker& iBooker, DQMStore::IGetter& iGetter) {
  // book the parameters containers
  PPSTimingCalibration::ParametersMap calib_params;
  PPSTimingCalibration::TimingMap calib_time;

  iGetter.cd();
  iGetter.setCurrentFolder(dqmDir_);

  // compute the fit parameters for all monitored channels
  TimingCalibrationHistograms hists;
  std::string ch_name;
  for (const auto& detid : detids_) {
    detid.channelName(ch_name);
    const auto chid = detid.rawId();
    const PPSTimingCalibration::Key key{
        (int)detid.arm(), (int)detid.station(), (int)detid.plane(), (int)detid.channel()};

    calib_params[key] = {0, 0, 0, 0};
    calib_time[key] = std::make_pair(offset_, resolution_);

    hists.leadingTime[chid] = iGetter.get(dqmDir_ + "/t_" + ch_name);
    if (hists.leadingTime[chid] == nullptr) {
      edm::LogInfo("PPSTimingCalibrationPCLHarvester:dqmEndJob")
          << "Failed to retrieve leading time monitor for channel (" << detid << ").";
      continue;
    }
    hists.toT[chid] = iGetter.get(dqmDir_ + "/tot_" + ch_name);
    if (hists.toT[chid] == nullptr) {
      edm::LogInfo("PPSTimingCalibrationPCLHarvester:dqmEndJob")
          << "Failed to retrieve time over threshold monitor for channel (" << detid << ").";
      continue;
    }
    hists.leadingTimeVsToT[chid] = iGetter.get(dqmDir_ + "/tvstot_" + ch_name);
    if (hists.leadingTimeVsToT[chid] == nullptr) {
      edm::LogInfo("PPSTimingCalibrationPCLHarvester:dqmEndJob")
          << "Failed to retrieve leading time vs. time over threshold monitor for channel (" << detid << ").";
      continue;
    }
    if (min_entries_ > 0 && hists.leadingTimeVsToT[chid]->getEntries() < min_entries_) {
      edm::LogWarning("PPSTimingCalibrationPCLHarvester:dqmEndJob")
          << "Not enough entries for channel (" << detid << "): " << hists.leadingTimeVsToT[chid]->getEntries() << " < "
          << min_entries_ << ". Skipping calibration.";
      continue;
    }

    //find max
    int max_bin_pos = 1;
    for (int i = 0; i < hists.toT[chid]->getNbinsX(); i++) {
      double bin_value = hists.toT[chid]->getBinContent(i);
      int bin_x_pos = hists.toT[chid]->getTH1()->GetXaxis()->GetBinCenter(i);
      if (bin_x_pos > upper_limit_max_search_)
        break;
      if (bin_value > hists.toT[chid]->getBinContent(max_bin_pos))
        max_bin_pos = i;
    }
    //find ranges
    int upper_limit_pos = max_bin_pos;
    int lower_limit_pos = max_bin_pos;
    double threshold = threshold_fraction_of_max_ * hists.toT[chid]->getBinContent(max_bin_pos);
    while (hists.toT[chid]->getTH1()->GetXaxis()->GetBinCenter(upper_limit_pos) < upper_limit_range_search_) {
      upper_limit_pos++;
      if (hists.toT[chid]->getBinContent(upper_limit_pos) < threshold)
        break;
    }
    while (hists.toT[chid]->getTH1()->GetXaxis()->GetBinCenter(lower_limit_pos) > lower_limit_range_search_) {
      lower_limit_pos--;
      if (hists.toT[chid]->getBinContent(lower_limit_pos) < threshold)
        break;
    }
    double upper_tot_range = hists.toT[chid]->getTH1()->GetXaxis()->GetBinCenter(upper_limit_pos);
    double lower_tot_range = hists.toT[chid]->getTH1()->GetXaxis()->GetBinCenter(lower_limit_pos);

    {  // scope for x-profile

      std::string ch_name;
      detid.channelName(ch_name);
      auto profile = iBooker.bookProfile(ch_name + "_prof_x", ch_name + "_prof_x", 240, 0., 60., 450, -20., 25.);

      std::unique_ptr<TProfile> prof(hists.leadingTimeVsToT[chid]->getTH2F()->ProfileX("_prof_x", 1, -1));
      *(profile->getTProfile()) = *((TProfile*)prof->Clone());
      profile->getTProfile()->SetTitle(ch_name.c_str());
      profile->getTProfile()->SetName(ch_name.c_str());

      interp_.SetParameters(hists.leadingTime[chid]->getRMS(),
                            hists.toT[chid]->getMean(),
                            0.8,
                            hists.leadingTime[chid]->getMean() - hists.leadingTime[chid]->getRMS());
      const auto& res = profile->getTProfile()->Fit(&interp_, "B+", "", lower_tot_range, upper_tot_range);
      if (!(bool)res) {
        calib_params[key] = {
            interp_.GetParameter(0), interp_.GetParameter(1), interp_.GetParameter(2), interp_.GetParameter(3)};
        calib_time[key] =
            std::make_pair(offset_, resolution_);  // hardcoded offset/resolution placeholder for the time being
        // can possibly do something with interp_.GetChiSquare() in the near future

      } else
        edm::LogWarning("PPSTimingCalibrationPCLHarvester:dqmEndJob")
            << "Fit did not converge for channel (" << detid << ").";
    }
  }

  // fill the DB object record
  PPSTimingCalibration calib(formula_, calib_params, calib_time);

  // write the object
  edm::Service<cond::service::PoolDBOutputService> poolDbService;
  poolDbService->writeOneIOV(calib, poolDbService->currentTime(), "PPSTimingCalibrationRcd_HPTDC");
}

//------------------------------------------------------------------------------

void PPSTimingCalibrationPCLHarvester::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<std::string>("dqmDir", "AlCaReco/PPSTimingCalibrationPCL")
      ->setComment("input path for the various DQM plots");
  desc.add<std::string>("formula", "[0]/(exp((x-[1])/[2])+1)+[3]")
      ->setComment("interpolation formula for the time walk component");
  desc.add<unsigned int>("minEntries", 100)->setComment("minimal number of hits to extract calibration");
  desc.add<double>("thresholdFractionOfMax", 0.05)
      ->setComment("threshold for TOT fit, defined as percent of max count in 1D TOT");
  descriptions.addWithDefaultLabel(desc);
}

DEFINE_FWK_MODULE(PPSTimingCalibrationPCLHarvester);
