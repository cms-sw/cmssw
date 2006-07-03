#ifndef DumpDBToFile_H
#define DumpDBToFile_H

/** \class DumpDBToFile
 *  Class which dump the ttrig written in a DB into
 *  a txt file of the same format of ORCA MuBarDigiParameters
 *  (see DTCalibrationMap for details)
 *
 *  $Date: 2006/06/16 12:22:38 $
 *  $Revision: 1.1 $
 *  \author G. Cerminara - INFN Torino
 */

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include <string>

class DTTtrig;
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

  DTCalibrationMap *theCalibFile;

  std::string theOutputFileName;
};
#endif

