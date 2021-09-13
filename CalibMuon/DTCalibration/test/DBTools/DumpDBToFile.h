#ifndef DumpDBToFile_H
#define DumpDBToFile_H

/** \class DumpDBToFile
 *  Class which dump the ttrig written in a DB into
 *  a txt file of the same format of ORCA MuBarDigiParameters
 *  (see DTCalibrationMap for details)
 *
 *  \author G. Cerminara - INFN Torino
 */

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"

#include <string>

class DTMtime;
class DTTtrig;
class DTT0;
class DTStatusFlag;
class DTDeadFlag;
class DTCalibrationMap;
class DTReadOutMapping;
class DTRecoConditions;

class DumpDBToFile : public edm::one::EDAnalyzer<edm::one::WatchRuns> {
public:
  /// Constructor
  DumpDBToFile(const edm::ParameterSet &pset);

  /// Destructor
  ~DumpDBToFile() override;

  // Operations
  void beginRun(const edm::Run &run, const edm::EventSetup &setup) override;

  void endRun(const edm::Run &run, const edm::EventSetup &setup) override {}

  void analyze(const edm::Event &event, const edm::EventSetup &setup) override {}

  void endJob() override;

protected:
private:
  const DTMtime *mTimeMap;
  const DTTtrig *tTrigMap;
  const DTT0 *tZeroMap;
  const DTStatusFlag *statusMap;
  const DTDeadFlag *deadMap;
  const DTReadOutMapping *channelsMap;
  const DTRecoConditions *rconds;

  DTCalibrationMap *theCalibFile;

  std::string theOutputFileName;

  std::string dbToDump;
  std::string dbLabel;
  std::string format;
};
#endif
