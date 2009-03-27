#ifndef DTTTrigCorrectionFirst_H
#define DTTTrigCorrectionFirst_H

/** \class DTTTrigCorrection
 *  Class which read a ttrig DB and correct it with
 *  the near SL (or the global average)
 *
 *  $Date: 2008/11/20 14:05:57 $
 *  $Revision: 1.3 $
 *  \author S. Maselli - INFN Torino
 */

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include <FWCore/Framework/interface/ESHandle.h>

#include <string>

class DTTtrig;
class DTGeometry;

class DTTTrigCorrectionFirst : public edm::EDAnalyzer {
public:
  /// Constructor
  DTTTrigCorrectionFirst(const edm::ParameterSet& pset);

  /// Destructor
  virtual ~DTTTrigCorrectionFirst();

  // Operations

  virtual void beginJob(const edm::EventSetup& setup);
  virtual void beginRun( const edm::Run& run, const edm::EventSetup& setup );
  virtual void analyze(const edm::Event& event, const edm::EventSetup& setup){}

  virtual void endJob();

protected:

private:
  const DTTtrig *tTrigMap;
  edm::ESHandle<DTGeometry> muonGeom;

  bool debug;
  double ttrigMin,ttrigMax,rmsLimit; 
};
#endif

