#ifndef DTTTrigCorrection_H
#define DTTTrigCorrection_H

/** \class DTTTrigCorrection
 *  Class which read a ttrig DB and correct it with
 *  the near SL (or the global average)
 *
 *  $Date: 2008/08/13 12:42:09 $
 *  $Revision: 1.2 $
 *  \author S. Maselli - INFN Torino
 */

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include <string>

class DTTtrig;
class DTGeometry;
class DTTTrigBaseCorrection;

class DTTTrigCorrection : public edm::EDAnalyzer {
public:
  /// Constructor
  DTTTrigCorrection(const edm::ParameterSet& pset);

  /// Destructor
  virtual ~DTTTrigCorrection();

  // Operations

  virtual void beginJob(const edm::EventSetup& setup);
  virtual void beginRun( const edm::Run& run, const edm::EventSetup& setup );
  virtual void analyze(const edm::Event& event, const edm::EventSetup& setup){}
  virtual void endJob();

protected:

private:
  const DTTtrig* tTrigMap_;
  edm::ESHandle<DTGeometry> muonGeom_;

  DTTTrigBaseCorrection* correctionAlgo_;
};
#endif

