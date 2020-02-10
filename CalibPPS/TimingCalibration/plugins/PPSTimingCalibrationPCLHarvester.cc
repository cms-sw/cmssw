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

#include "CommonTools/Utils/interface/FormulaEvaluator.h"

#include "Geometry/VeryForwardGeometryBuilder/interface/CTPPSGeometry.h"
#include "Geometry/Records/interface/VeryForwardRealGeometryRecord.h"

#include "CalibPPS/TimingCalibration/interface/TimingCalibrationStruct.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"

#include "DataFormats/CTPPSDetId/interface/CTPPSDiamondDetId.h"
#include "CondFormats/CTPPSReadoutObjects/interface/PPSTimingCalibration.h"

//------------------------------------------------------------------------------

class PPSTimingCalibrationPCLHarvester : public DQMEDHarvester {
public:
  PPSTimingCalibrationPCLHarvester(const edm::ParameterSet&);
  void endRun(const edm::Run&, const edm::EventSetup&) override;

  static void fillDescriptions(edm::ConfigurationDescriptions&);

private:
  void dqmEndJob(DQMStore::IBooker&, DQMStore::IGetter&) override;
  std::vector<CTPPSDiamondDetId> detids_;
  const std::string formula_;
  const reco::FormulaEvaluator interp_;
};

//------------------------------------------------------------------------------

PPSTimingCalibrationPCLHarvester::PPSTimingCalibrationPCLHarvester(const edm::ParameterSet& iConfig)
  :formula_(iConfig.getParameter<std::string>("formula")),
   interp_(formula_)
{}

//------------------------------------------------------------------------------

void PPSTimingCalibrationPCLHarvester::endRun(const edm::Run& iRun, const edm::EventSetup& iSetup)
{
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

void PPSTimingCalibrationPCLHarvester::dqmEndJob(DQMStore::IBooker& iBooker, DQMStore::IGetter& iGetter)
{
  // book the parameters containers
  PPSTimingCalibration::ParametersMap calib_params;
  PPSTimingCalibration::TimingMap calib_time;

  // compute the fit parameters for all monitored channels
  TimingCalibrationHistograms hists;
  std::string ch_name;
  for (const auto& detid : detids_) {
    detid.channelName(ch_name);
    const PPSTimingCalibration::Key key{ (int)detid.arm(), (int)detid.station(), (int)detid.plane(), (int)detid.channel() };
    hists.leadingTime[detid.rawId()] = iGetter.get("t_"+ch_name);
    hists.leadingTimeVsToT[detid.rawId()] = iGetter.get("tvstot_"+ch_name);
    std::cout << detid << ": " << hists.leadingTime[detid.rawId()]->getMean() << std::endl;
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
  descriptions.addWithDefaultLabel(desc);
}

DEFINE_FWK_MODULE(PPSTimingCalibrationPCLHarvester);
