#ifndef DTTTrigCorrectionFirst_H
#define DTTTrigCorrectionFirst_H

/** \class DTTTrigCorrection
 *  Class which read a ttrig DB and correct it with
 *  the near SL (or the global average)
 *
 *  \author S. Maselli - INFN Torino
 */

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "CondFormats/DataRecord/interface/DTTtrigRcd.h"

#include <string>

class DTTtrig;
class DTGeometry;

class DTTTrigCorrectionFirst : public edm::one::EDAnalyzer<edm::one::WatchRuns> {
public:
  /// Constructor
  DTTTrigCorrectionFirst(const edm::ParameterSet& pset);

  /// Destructor
  ~DTTTrigCorrectionFirst() override;

  // Operations

  void beginJob() override {}
  void beginRun(const edm::Run& run, const edm::EventSetup& setup) override;
  void analyze(const edm::Event& event, const edm::EventSetup& setup) override {}

  void endRun(const edm::Run& run, const edm::EventSetup& setup) override{};
  void endJob() override;

protected:
private:
  edm::ESHandle<DTGeometry> muonGeom;
  const edm::ESGetToken<DTGeometry, MuonGeometryRecord> dtGeomToken_;

  const DTTtrig* tTrigMap;
  const edm::ESGetToken<DTTtrig, DTTtrigRcd> ttrigToken_;

  bool debug;
  double ttrigMin, ttrigMax, rmsLimit;
};
#endif
