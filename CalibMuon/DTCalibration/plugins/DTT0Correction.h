#ifndef CalibMuon_DTCalibration_DTT0Correction_h
#define CalibMuon_DTCalibration_DTT0Correction_h

/** \class DTT0Correction
 *  Class that reads and corrects t0 DB
 */

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include <string>

class DTT0;
class DTGeometry;
namespace dtCalibration {
  class DTT0BaseCorrection;
}

class DTT0Correction : public edm::EDAnalyzer {
public:
  /// Constructor
  DTT0Correction(const edm::ParameterSet& pset);

  /// Destructor
  ~DTT0Correction() override;

  // Operations

  void beginJob() override {}
  void beginRun(const edm::Run& run, const edm::EventSetup& setup) override;
  void analyze(const edm::Event& event, const edm::EventSetup& setup) override {}
  void endJob() override;

protected:
private:
  const DTT0* t0Map_;
  edm::ESHandle<DTGeometry> muonGeom_;

  std::unique_ptr<dtCalibration::DTT0BaseCorrection> correctionAlgo_;
};
#endif
