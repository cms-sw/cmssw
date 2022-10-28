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

#include "CondFormats/DTObjects/interface/DTMtime.h"
#include "CondFormats/DataRecord/interface/DTMtimeRcd.h"
#include "CondFormats/DTObjects/interface/DTTtrig.h"
#include "CondFormats/DataRecord/interface/DTTtrigRcd.h"
#include "CondFormats/DTObjects/interface/DTT0.h"
#include "CondFormats/DataRecord/interface/DTT0Rcd.h"
#include "CondFormats/DataRecord/interface/DTStatusFlagRcd.h"
#include "CondFormats/DTObjects/interface/DTStatusFlag.h"
#include "CondFormats/DataRecord/interface/DTDeadFlagRcd.h"
#include "CondFormats/DTObjects/interface/DTDeadFlag.h"
#include "CondFormats/DataRecord/interface/DTReadOutMappingRcd.h"
#include "CondFormats/DTObjects/interface/DTReadOutMapping.h"
#include "CondFormats/DTObjects/interface/DTRecoConditions.h"
#include "CondFormats/DataRecord/interface/DTRecoConditionsTtrigRcd.h"
#include "CondFormats/DataRecord/interface/DTRecoConditionsVdriftRcd.h"
#include "CondFormats/DataRecord/interface/DTRecoConditionsUncertRcd.h"

#include <string>

class DTCalibrationMap;

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
  edm::ESGetToken<DTMtime, DTMtimeRcd> mTimeMapToken_;
  edm::ESGetToken<DTTtrig, DTTtrigRcd> tTrigMapToken_;
  edm::ESGetToken<DTT0, DTT0Rcd> t0MapToken_;
  edm::ESGetToken<DTStatusFlag, DTStatusFlagRcd> statusMapToken_;
  edm::ESGetToken<DTDeadFlag, DTDeadFlagRcd> deadMapToken_;
  edm::ESGetToken<DTReadOutMapping, DTReadOutMappingRcd> readOutMapToken_;
  edm::ESGetToken<DTRecoConditions, DTRecoConditionsTtrigRcd> tTrigToken_;
  edm::ESGetToken<DTRecoConditions, DTRecoConditionsVdriftRcd> vDriftToken_;
  edm::ESGetToken<DTRecoConditions, DTRecoConditionsUncertRcd> uncertToken_;

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
