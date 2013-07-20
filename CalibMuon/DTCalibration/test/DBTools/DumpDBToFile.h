#ifndef DumpDBToFile_H
#define DumpDBToFile_H

/** \class DumpDBToFile
 *  Class which dump the ttrig written in a DB into
 *  a txt file of the same format of ORCA MuBarDigiParameters
 *  (see DTCalibrationMap for details)
 *
 *  $Date: 2010/02/15 16:45:47 $
 *  $Revision: 1.8 $
 *  \author G. Cerminara - INFN Torino
 */

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include <string>

class DTMtime;
class DTTtrig;
class DTT0;
class DTStatusFlag;
class DTDeadFlag;
class DTCalibrationMap;
class DTReadOutMapping;

class DumpDBToFile : public edm::EDAnalyzer {
public:
  /// Constructor
  DumpDBToFile(const edm::ParameterSet& pset);

  /// Destructor
  virtual ~DumpDBToFile();

  // Operations
  virtual void beginRun(const edm::Run& run, const edm::EventSetup& setup );

  virtual void analyze(const edm::Event& event, const edm::EventSetup& setup){}

  virtual void endJob();

protected:

private:
  const DTMtime *mTimeMap;
  const DTTtrig *tTrigMap;
  const DTT0 *tZeroMap;
  const DTStatusFlag *statusMap;
  const DTDeadFlag *deadMap;
  const DTReadOutMapping *channelsMap;

  DTCalibrationMap *theCalibFile;

  std::string theOutputFileName;

  std::string dbToDump;
  std::string dbLabel;

};
#endif

