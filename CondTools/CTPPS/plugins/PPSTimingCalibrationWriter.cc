/****************************************************************************
 *
 * This is a part of TOTEM/PPS offline software.
 * Author:
 *   Laurent Forthomme (laurent.forthomme@cern.ch)
 *
 ****************************************************************************/

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"

#include "CondFormats/CTPPSReadoutObjects/interface/PPSTimingCalibration.h"
#include "CondFormats/DataRecord/interface/PPSTimingCalibrationRcd.h"

#include <memory>

class PPSTimingCalibrationWriter : public edm::one::EDAnalyzer<> {
public:
  explicit PPSTimingCalibrationWriter(const edm::ParameterSet&) {}

private:
  void beginJob() override {}
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void endJob() override {}
};

void PPSTimingCalibrationWriter::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  // get timing calibration parameters
  edm::ESHandle<PPSTimingCalibration> hTimingCalib;
  iSetup.get<PPSTimingCalibrationRcd>().get(hTimingCalib);

  // store the calibration into a DB object
  edm::Service<cond::service::PoolDBOutputService> poolDbService;
  if (poolDbService.isAvailable())
    poolDbService->writeOne(hTimingCalib.product(), poolDbService->currentTime(), "PPSTimingCalibrationRcd");
  else
    throw cms::Exception("PPSTimingCalibrationWriter") << "PoolDBService required.";
}

DEFINE_FWK_MODULE(PPSTimingCalibrationWriter);
