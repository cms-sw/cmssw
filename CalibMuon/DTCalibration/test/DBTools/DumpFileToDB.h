#ifndef DumpFileToDB_H
#define DumpFileToDB_H

/** \class DumpFileToDB
 *  Dump the content of a txt file with the format
 *  of ORCA MuBarDigiParameters (see DTCalibrationMap for details)
 *  into a DB. At the moment only the ttrig info is handled.
 *
 *  \author G. Cerminara - INFN Torino
 */

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "CondFormats/DataRecord/interface/DTTtrigRcd.h"

#include <string>
#include <fstream>
#include <vector>

class DTCalibrationMap;
class DTTtrig;

class DumpFileToDB : public edm::one::EDAnalyzer<edm::one::WatchRuns> {
public:
  /// Constructor
  DumpFileToDB(const edm::ParameterSet& pset);

  /// Destructor
  ~DumpFileToDB() override;

  // Operations
  void beginRun(const edm::Run& run, const edm::EventSetup& setup) override;

  void endRun(const edm::Run& run, const edm::EventSetup& setup) override {}

  void analyze(const edm::Event& event, const edm::EventSetup& setup) override {}

  void endJob() override;

protected:
private:
  std::vector<int> readChannelsMap(std::stringstream& linestr);

  const DTCalibrationMap* theCalibFile;
  std::string mapFileName;

  std::string dbToDump;
  std::string format;

  // sum the correction in the txt file (for the mean value) to what is input DB
  bool diffMode;
  const DTTtrig* tTrigMapOrig;

  edm::ESGetToken<DTTtrig, DTTtrigRcd> ttrigToken_;
};
#endif
