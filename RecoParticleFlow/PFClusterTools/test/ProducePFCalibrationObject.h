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

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <vector>
#include <string>

class ProducePFCalibrationObject : public edm::EDAnalyzer {
public:
  /// Constructor
  ProducePFCalibrationObject(const edm::ParameterSet&);

  /// Destructor
  ~ProducePFCalibrationObject() override;

  // Operations
  //   virtual void beginJob();
  void beginRun(const edm::Run& run, const edm::EventSetup& eSetup) override;

  void analyze(const edm::Event&, const edm::EventSetup&) override {}
  void endJob() override {}

protected:
private:
  bool read;
  bool write;

  std::vector<edm::ParameterSet> fToWrite;
  std::vector<std::string> fToRead;
};
#endif
