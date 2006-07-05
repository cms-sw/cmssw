#ifndef DumpDBToFile_H
#define DumpDBToFile_H

/** \class DumpDBToFile
 *  Class which dump the ttrig written in a DB into
 *  a txt file of the same format of ORCA MuBarDigiParameters
 *  (see DTCalibrationMap for details)
 *
 *  $Date: 2006/07/03 15:09:40 $
 *  $Revision: 1.2 $
 *  \author G. Cerminara - INFN Torino
 */

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include <string>

class DTTtrig;
class DTT0;
class DTStatusFlag;
class DTCalibrationMap;

class DumpDBToFile : public edm::EDAnalyzer {
public:
  /// Constructor
  DumpDBToFile(const edm::ParameterSet& pset);

  /// Destructor
  virtual ~DumpDBToFile();

  // Operations

  virtual void beginJob(const edm::EventSetup& setup);

  virtual void analyze(const edm::Event& event, const edm::EventSetup& setup){}

  virtual void endJob();

protected:

private:
  const DTTtrig *tTrigMap;
  const DTT0 *tZeroMap;
  const DTStatusFlag *statusMap;

  DTCalibrationMap *theCalibFile;

  std::string theOutputFileName;

  std::string dbToDump;

};
#endif

