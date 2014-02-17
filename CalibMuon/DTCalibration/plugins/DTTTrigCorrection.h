#ifndef CalibMuon_DTCalibration_DTTTrigCorrection_h
#define CalibMuon_DTCalibration_DTTTrigCorrection_h

/** \class DTTTrigCorrection
 *  Class which read a ttrig DB and correct it with
 *  the near SL (or the global average)
 *
 *  $Date: 2012/03/02 19:47:32 $
 *  $Revision: 1.9 $
 *  \author S. Maselli - INFN Torino
 */

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include <string>

class DTTtrig;
class DTGeometry;
namespace dtCalibration {
  class DTTTrigBaseCorrection;
}

class DTTTrigCorrection : public edm::EDAnalyzer {
public:
  /// Constructor
  DTTTrigCorrection(const edm::ParameterSet& pset);

  /// Destructor
  virtual ~DTTTrigCorrection();

  // Operations

  virtual void beginJob() {}
  virtual void beginRun( const edm::Run& run, const edm::EventSetup& setup );
  virtual void analyze(const edm::Event& event, const edm::EventSetup& setup){}
  virtual void endJob();

protected:

private:
  std::string dbLabel_;

  const DTTtrig* tTrigMap_;
  edm::ESHandle<DTGeometry> muonGeom_;

  dtCalibration::DTTTrigBaseCorrection* correctionAlgo_;
};
#endif

