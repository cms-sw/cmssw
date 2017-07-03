#ifndef ShiftTTrigDB_H
#define ShiftTTrigDB_H

/** \class ShiftTTrigDB
 *  Class which read a ttrig DB and modifies it introducing
 *  shifts with chamber granularity
 *
 *  \author S. Bolognesi - INFN Torino
 */

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include <map>
#include <string>
#include <vector>

class DTTtrig;
class DTGeometry;

class ShiftTTrigDB : public edm::EDAnalyzer {
public:
  /// Constructor
  ShiftTTrigDB(const edm::ParameterSet& pset);

  /// Destructor
  ~ShiftTTrigDB() override;

  // Operations

  void beginRun(const edm::Run& run, const edm::EventSetup& setup ) override;

  void analyze(const edm::Event& event, const edm::EventSetup& setup) override{}

  void endJob() override;

protected:

private:
  const DTTtrig *tTrigMap;
  edm::ESHandle<DTGeometry> muonGeom;

  std::string dbLabel;

  std::vector<std::vector<int> > chambers;
  std::vector<double> shifts;
  std::map<std::vector <int>, double> mapShiftsByChamber;
  bool debug;

};
#endif

