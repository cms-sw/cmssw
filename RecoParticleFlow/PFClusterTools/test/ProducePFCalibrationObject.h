#ifndef ProducePFCalibrationObject_H
#define ProducePFCalibrationObject_H

/** \class ProducePFCalibrationObject
 *  
 *  
 *  This is used by ProducePFCalibration.py config to 
 *  generate payload for offline and HLT PF hadron calibration.
 *  
 *  \Original author G. Cerminara - CERN
 */

#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "CondFormats/PhysicsToolsObjects/interface/PerformancePayloadFromTFormula.h"
#include "CondFormats/DataRecord/interface/PFCalibrationRcd.h"

#include <vector>
#include <string>

class ProducePFCalibrationObject : public edm::one::EDAnalyzer<edm::one::WatchRuns> {
public:
  /// Constructor
  ProducePFCalibrationObject(const edm::ParameterSet&);

  /// Destructor
  ~ProducePFCalibrationObject() override;

  // Operations
  void beginRun(const edm::Run& run, const edm::EventSetup& eSetup) override;
  void endRun(const edm::Run& run, const edm::EventSetup& eSetup) override {}

  void analyze(const edm::Event&, const edm::EventSetup&) override {}

protected:
private:
  bool read;
  bool write;

  std::vector<edm::ParameterSet> fToWrite;
  std::vector<std::string> fToRead;

  const edm::ESGetToken<PerformancePayload, PFCalibrationRcd> perfToken;
};
#endif
