#ifndef ShiftTTrigDB_H
#define ShiftTTrigDB_H

/** \class ShiftTTrigDB
 *  Class which read a ttrig DB and modifies it introducing
 *  shifts with chamber granularity
 *
 *  \author S. Bolognesi - INFN Torino
 */

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "Geometry/DTGeometry/interface/DTGeometry.h"
#include "CondFormats/DataRecord/interface/DTTtrigRcd.h"

#include <map>
#include <string>
#include <vector>

class DTTtrig;
class DTGeometry;

class ShiftTTrigDB : public edm::one::EDAnalyzer<> {
public:
  /// Constructor
  ShiftTTrigDB(const edm::ParameterSet& pset);

  /// Destructor
  virtual ~ShiftTTrigDB();

  // Operations

  virtual void beginRun(const edm::Run& run, const edm::EventSetup& setup);

  virtual void analyze(const edm::Event& event, const edm::EventSetup& setup) {}

  virtual void endJob();

protected:
private:
  const DTTtrig* tTrigMap;
  edm::ESHandle<DTGeometry> muonGeom;

  std::vector<std::vector<int> > chambers;
  std::vector<double> shifts;
  std::map<std::vector<int>, double> mapShiftsByChamber;
  bool debug;

  edm::ESGetToken<DTTtrig, DTTtrigRcd> ttrigToken_;
  edm::ESGetToken<DTGeometry, MuonGeometryRecord> dtGeomToken_;
};
#endif
