#ifndef DTTTrigCorrection_H
#define DTTTrigCorrection_H

/** \class DTTTrigCorrection
 *  Class which read a ttrig DB and correct it with
 *  the near SL (or the global average)
 *
 *  $Date: 2007/11/09 17:55:38 $
 *  $Revision: 1.1 $
 *  \author S. Maselli - INFN Torino
 */

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include <FWCore/Framework/interface/ESHandle.h>

#include <string>

class DTTtrig;
class DTGeometry;

class DTTTrigCorrection : public edm::EDAnalyzer {
public:
  /// Constructor
  DTTTrigCorrection(const edm::ParameterSet& pset);

  /// Destructor
  virtual ~DTTTrigCorrection();

  // Operations

  virtual void beginJob(const edm::EventSetup& setup);

  virtual void analyze(const edm::Event& event, const edm::EventSetup& setup){}

  virtual void endJob();

protected:

private:
  const DTTtrig *tTrigMap;
  edm::ESHandle<DTGeometry> muonGeom;

  bool debug;
  double ttrigMin,ttrigMax;

};
#endif

