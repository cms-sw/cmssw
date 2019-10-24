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

#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"
#include "CondFormats/CTPPSReadoutObjects/interface/PPSTimingCalibration.h"

//------------------------------------------------------------------------------

class PPSTimingCalibrationPCLHarvester : public DQMEDHarvester {
public:
  PPSTimingCalibrationPCLHarvester(const edm::ParameterSet&);
  void endRun(const edm::Run&, const edm::EventSetup&) override;

  static void fillDescriptions(edm::ConfigurationDescriptions&);

private:
  void dqmEndJob(DQMStore::IBooker&, DQMStore::IGetter&) override;
};

//------------------------------------------------------------------------------

PPSTimingCalibrationPCLHarvester::PPSTimingCalibrationPCLHarvester(const edm::ParameterSet& iConfig)
{
}

//------------------------------------------------------------------------------

void PPSTimingCalibrationPCLHarvester::endRun(const edm::Run& iRun, const edm::EventSetup& iSetup)
{
}

//------------------------------------------------------------------------------

void PPSTimingCalibrationPCLHarvester::dqmEndJob(DQMStore::IBooker& iBooker, DQMStore::IGetter& iGetter)
{
  // fill the DB object record
  PPSTimingCalibration calib;

  // write the object
  edm::Service<cond::service::PoolDBOutputService> poolDbService;
  if (!poolDbService.isAvailable())
    throw cms::Exception("PPSTimingCalibrationPCLHarvester:dqmEndJob") << "PoolDBService required";
  poolDbService->writeOne(&calib, poolDbService->currentTime(), "PPSTimingCalibrationRcd");
}

//------------------------------------------------------------------------------

void PPSTimingCalibrationPCLHarvester::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  /*desc.add<edm::FileInPath>("calibrationFile", edm::FileInPath())
      ->setComment("file with SAMPIC calibrations, ADC and INL; if empty or corrupted, no calibration will be applied");
  desc.add<unsigned int>("subDetector", (unsigned int)PPSTimingCalibrationESSource::DetectorType::INVALID)
      ->setComment("type of sub-detector for which the calibrations are provided");*/

  descriptions.add("ppsTimingCalibrationPCLHarvester", desc);
}

DEFINE_FWK_MODULE(PPSTimingCalibrationPCLHarvester);
