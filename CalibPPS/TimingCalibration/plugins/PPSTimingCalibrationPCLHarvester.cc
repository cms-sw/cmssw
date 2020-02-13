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
#include "CondFormats/CTPPSReadoutObjects/interface/PPSTimingCalibration.h"

#include "TProfile.h"
#include "TF1.h"

//------------------------------------------------------------------------------

class PPSTimingCalibrationPCLHarvester : public DQMEDHarvester {
public:
  PPSTimingCalibrationPCLHarvester(const edm::ParameterSet&);
  void beginRun(const edm::Run&, const edm::EventSetup&) override;

  static void fillDescriptions(edm::ConfigurationDescriptions&);

private:
  void dqmEndJob(DQMStore::IBooker&, DQMStore::IGetter&) override;
  std::vector<CTPPSDiamondDetId> detids_;
  const std::string dqmDir_;
  const std::string formula_;
  const unsigned int min_entries_;
  TF1 interp_;
};

//------------------------------------------------------------------------------

PPSTimingCalibrationPCLHarvester::PPSTimingCalibrationPCLHarvester(const edm::ParameterSet& iConfig)
    : dqmDir_(iConfig.getParameter<std::string>("dqmDir")),
      formula_(iConfig.getParameter<std::string>("formula")),
      min_entries_(iConfig.getParameter<unsigned int>("minEntries")),
      interp_("interp", formula_.c_str(), 10.5, 25.) {
  interp_.SetParLimits(1, 9., 15.);
  interp_.SetParLimits(2, 0.2, 2.5);
}

//------------------------------------------------------------------------------

void PPSTimingCalibrationPCLHarvester::beginRun(const edm::Run& iRun, const edm::EventSetup& iSetup) {
  edm::ESHandle<CTPPSGeometry> hGeom;
  iSetup.get<VeryForwardRealGeometryRecord>().get(hGeom);
  for (auto it = hGeom->beginSensor(); it != hGeom->endSensor(); ++it) {
    CTPPSDetId base_detid(it->first);
    try {
      CTPPSDiamondDetId detid(base_detid);
      if (detid.station() == 1)
        detids_.emplace_back(detid);
    } catch (const cms::Exception&) {
      continue;
    }
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
    //std::cout << "1:" << detid << std::endl;
    hists.leadingTime[chid] = iGetter.get("t_" + ch_name);
    if (hists.leadingTime[chid] == nullptr) {
      edm::LogInfo("PPSTimingCalibrationPCLHarvester:dqmEndJob")
          << "Failed to retrieve leading time monitor for channel (" << detid << ").";
      continue;
    }
    //std::cout << "2:" << detid << std::endl;
    hists.toT[chid] = iGetter.get("tot_" + ch_name);
    if (hists.toT[chid] == nullptr) {
      edm::LogInfo("PPSTimingCalibrationPCLHarvester:dqmEndJob")
          << "Failed to retrieve time over threshold monitor for channel (" << detid << ").";
      continue;
    }
    //std::cout << "3:" << detid << std::endl;
    hists.leadingTimeVsToT[chid] = iGetter.get("tvstot_" + ch_name);
    if (hists.leadingTimeVsToT[chid] == nullptr) {
      edm::LogInfo("PPSTimingCalibrationPCLHarvester:dqmEndJob")
          << "Failed to retrieve leading time vs. time over threshold monitor for channel (" << detid << ").";
      continue;
    }
    //std::cout << "4:" << detid << std::endl;
    if (min_entries_ > 0 && hists.leadingTimeVsToT[chid]->getEntries() < min_entries_) {
      edm::LogWarning("PPSTimingCalibrationPCLHarvester:dqmEndJob")
          << "Not enough entries for channel (" << detid << "): " << hists.leadingTimeVsToT[chid]->getEntries() << " < "
          << min_entries_ << ". Skipping calibration.";
      continue;
    }
    const double upper_tot_range = hists.toT[chid]->getMean() + 2.5;
    auto prof = hists.leadingTimeVsToT[chid]->getTH2D()->ProfileX("_prof_x", 1, -1);
    interp_.SetParameters(hists.leadingTime[chid]->getRMS(),
                          hists.toT[chid]->getMean(),
                          0.8,
                          hists.leadingTime[chid]->getMean() - hists.leadingTime[chid]->getRMS());
    //std::cout << detid << ": " << hists.leadingTime[chid]->getMean() << std::endl;
    const auto& res = prof->Fit(&interp_, "B+", "", 10.4, upper_tot_range);
    if ((bool)res) {
      calib_params[key] = {
          interp_.GetParameter(0), interp_.GetParameter(1), interp_.GetParameter(2), interp_.GetParameter(3)};
      calib_time[key] = std::make_pair(0.1, 0.);  //FIXME hardcoded resolution/offset placeholder
      // do something with interp_.GetChiSquare()...
    } else
      edm::LogWarning("PPSTimingCalibrationPCLHarvester:dqmEndJob")
          << "Fit did not converge for channel (" << detid << ").";
    delete prof;
  }

  // fill the DB object record
  PPSTimingCalibration calib(formula_, calib_params, calib_time);

  // write the object
  edm::Service<cond::service::PoolDBOutputService> poolDbService;
  if (!poolDbService.isAvailable())
    throw cms::Exception("PPSTimingCalibrationPCLHarvester:dqmEndJob") << "PoolDBService required";
  poolDbService->writeOne(&calib, poolDbService->currentTime(), "PPSTimingCalibrationRcd");
}

//------------------------------------------------------------------------------

void PPSTimingCalibrationPCLHarvester::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<std::string>("dqmDir", "AlCaReco/PPSTimingCalibrationPCL")
      ->setComment("input path for the various DQM plots");
  desc.add<std::string>("formula", "[0]/(exp((x-[1])/[2])+1)+[3]")
      ->setComment("interpolation formula for the time walk component");
  desc.add<unsigned int>("minEntries", 100)->setComment("minimal number of hits to extract calibration");
  descriptions.addWithDefaultLabel(desc);
}

DEFINE_FWK_MODULE(PPSTimingCalibrationPCLHarvester);
