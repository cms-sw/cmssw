/****************************************************************************
 *
 * This is a part of TOTEM/PPS offline software.
 * Author:
 *   Laurent Forthomme (laurent.forthomme@cern.ch)
 *   Christopher Misan
 *
 ****************************************************************************/

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"

#include "CondFormats/PPSObjects/interface/PPSTimingCalibrationLUT.h"
#include "CondFormats/DataRecord/interface/PPSTimingCalibrationLUTRcd.h"

#include <memory>

class PPSTimingCalibrationLUTWriter : public edm::one::EDAnalyzer<> {
public:
  explicit PPSTimingCalibrationLUTWriter(const edm::ParameterSet&)
      : tokenCalibration_(esConsumes<PPSTimingCalibrationLUT, PPSTimingCalibrationLUTRcd>()) {}

private:
  void beginJob() override {}
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void endJob() override {}

  edm::ESGetToken<PPSTimingCalibrationLUT, PPSTimingCalibrationLUTRcd> tokenCalibration_;
};

void PPSTimingCalibrationLUTWriter::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  // get timing calibration parameters
  const auto& hTimingCalib = iSetup.getData(tokenCalibration_);
  // store the calibration into a DB object
  edm::Service<cond::service::PoolDBOutputService> poolDbService;
  if (poolDbService.isAvailable())
    poolDbService->writeOneIOV(hTimingCalib, poolDbService->currentTime(), "PPSTimingCalibrationLUTRcd");
  else
    throw cms::Exception("PPSTimingCalibrationLUTWriter") << "PoolDBService required.";
}

DEFINE_FWK_MODULE(PPSTimingCalibrationLUTWriter);