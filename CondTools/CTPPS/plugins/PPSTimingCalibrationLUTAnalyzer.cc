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
#include "FWCore/Framework/interface/ESWatcher.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"

#include "CondFormats/PPSObjects/interface/PPSTimingCalibrationLUT.h"
#include "CondFormats/DataRecord/interface/PPSTimingCalibrationLUTRcd.h"

#include <memory>

class PPSTimingCalibrationLUTAnalyzer : public edm::one::EDAnalyzer<> {
public:
  explicit PPSTimingCalibrationLUTAnalyzer(const edm::ParameterSet&)
      : tokenCalibration_(esConsumes<PPSTimingCalibrationLUT, PPSTimingCalibrationLUTRcd>()) {}

private:
  void beginJob() override {}
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void endJob() override {}

  edm::ESWatcher<PPSTimingCalibrationLUTRcd> calibWatcher_;

  edm::ESGetToken<PPSTimingCalibrationLUT, PPSTimingCalibrationLUTRcd> tokenCalibration_;
};

void PPSTimingCalibrationLUTAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  // get timing calibration parameters
  if (calibWatcher_.check(iSetup)) {
    edm::LogInfo("PPSTimingCalibrationLUTAnalyzer") << "Calibrations retrieved:\n" << iSetup.getData(tokenCalibration_);
  }
}

DEFINE_FWK_MODULE(PPSTimingCalibrationLUTAnalyzer);
