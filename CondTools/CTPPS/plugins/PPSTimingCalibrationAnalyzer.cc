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
#include "FWCore/Framework/interface/ESWatcher.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"

#include "CondFormats/CTPPSReadoutObjects/interface/PPSTimingCalibration.h"
#include "CondFormats/DataRecord/interface/PPSTimingCalibrationRcd.h"

#include <memory>

class PPSTimingCalibrationAnalyzer : public edm::one::EDAnalyzer<> {
public:
  explicit PPSTimingCalibrationAnalyzer(const edm::ParameterSet&) {}

private:
  void beginJob() override {}
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void endJob() override {}

  edm::ESWatcher<PPSTimingCalibrationRcd> calibWatcher_;
};

void PPSTimingCalibrationAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  // get timing calibration parameters
  edm::ESHandle<PPSTimingCalibration> hTimingCalib;
  if (calibWatcher_.check(iSetup)) {
    iSetup.get<PPSTimingCalibrationRcd>().get(hTimingCalib);

    edm::LogInfo("PPSTimingCalibrationAnalyzer") << "Calibrations retrieved:\n" << *hTimingCalib;
  }
}

DEFINE_FWK_MODULE(PPSTimingCalibrationAnalyzer);
