#ifndef FakeTTrig_H
#define FakeTTrig_H

/** \class FakeTTrigDB
 *
 *  Class which produce fake DB of ttrig with the correction of :
 *    --- 500 ns of delay
 *    --- time of wire propagation
 *    --- time of fly
 *
 *  $Date: 2010/02/15 16:45:47 $
 *  $Revision: 1.5 $
 *  \author Giorgia Mila - INFN Torino
 */

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include <FWCore/Framework/interface/ESHandle.h>
#include "FWCore/ParameterSet/interface/ParameterSet.h"

namespace CLHEP {
  class RandGaussQ;
}

#include <string>
class DTGeometry;
class DTSuperLayer;
class DTTtrig;

class FakeTTrig : public edm::EDAnalyzer {
public:
  /// Constructor
  FakeTTrig(const edm::ParameterSet& pset);

  /// Destructor
  virtual ~FakeTTrig();

  // Operations
  virtual void beginRun(const edm::Run& run, const edm::EventSetup& setup );
  virtual void analyze(const edm::Event& event, const edm::EventSetup& setup){}
  virtual void endJob();

  // TOF computation
  double tofComputation(const DTSuperLayer* superlayer);
  // wire propagation delay
  double wirePropComputation(const DTSuperLayer* superlayer);

protected:

private:
  edm::ESHandle<DTGeometry> muonGeom;
  edm::ParameterSet ps;

  double smearing;

  std::string dbLabel;

  /// tTrig from the DB
  float tTrigRef;
  float tTrigRMSRef;
  float kFactorRef;

  // Get the tTrigMap
  edm::ESHandle<DTTtrig> tTrigMapRef;

  // the random generator
  CLHEP::RandGaussQ* theGaussianDistribution;

};
#endif
