#ifndef CalibMuon_DTCalibration_DTT0Correction_h
#define CalibMuon_DTCalibration_DTT0Correction_h

/** \class DTT0Correction
 *  Class that reads and corrects t0 DB
 *
 *  $Date: 2012/03/02 19:47:32 $
 *  $Revision: 1.1 $
 *  \author A. Vilela Pereira
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
  virtual ~DTT0Correction();

  // Operations

  virtual void beginJob() {}
  virtual void beginRun( const edm::Run& run, const edm::EventSetup& setup );
  virtual void analyze(const edm::Event& event, const edm::EventSetup& setup){}
  virtual void endJob();

protected:

private:

  const DTT0* t0Map_;
  edm::ESHandle<DTGeometry> muonGeom_;

  dtCalibration::DTT0BaseCorrection* correctionAlgo_;
};
#endif
