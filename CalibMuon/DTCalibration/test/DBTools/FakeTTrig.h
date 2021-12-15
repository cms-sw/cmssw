#ifndef FakeTTrig_H
#define FakeTTrig_H

/** \class FakeTTrigDB
 *
 *  Class which produce fake DB of ttrig with the correction of :
 *    --- 500 ns of delay
 *    --- time of wire propagation
 *    --- time of fly
 *
 *  \author Giorgia Mila - INFN Torino
 */

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "Geometry/DTGeometry/interface/DTGeometry.h"
#include "CondFormats/DataRecord/interface/DTTtrigRcd.h"
#include <string>
class DTGeometry;
class DTSuperLayer;
class DTTtrig;

class FakeTTrig : public edm::one::EDAnalyzer<edm::one::WatchRuns, edm::one::WatchLuminosityBlocks> {
public:
  /// Constructor
  FakeTTrig(const edm::ParameterSet& pset);

  /// Destructor
  virtual ~FakeTTrig();

  // Operations
  virtual void beginRun(const edm::Run& run, const edm::EventSetup& setup) override;
  virtual void beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override;
  virtual void analyze(const edm::Event& event, const edm::EventSetup& setup) override {}
  virtual void endRun(const edm::Run& run, const edm::EventSetup& setup) override{};
  virtual void endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override{};
  virtual void endJob() override;

  // TOF computation
  double tofComputation(const DTSuperLayer* superlayer);
  // wire propagation delay
  double wirePropComputation(const DTSuperLayer* superlayer);

protected:
private:
  edm::ESHandle<DTGeometry> muonGeom;
  edm::ParameterSet ps;

  double smearing;

  /// tTrig from the DB
  float tTrigRef;
  float tTrigRMSRef;
  float kFactorRef;

  // Get the tTrigMap
  edm::ESHandle<DTTtrig> tTrigMapRef;

  bool dataBaseWriteWasDone;

  edm::ESGetToken<DTTtrig, DTTtrigRcd> ttrigToken_;
  edm::ESGetToken<DTGeometry, MuonGeometryRecord> dtGeomToken_;
};
#endif
