#ifndef DumpFileToDB_H
#define DumpFileToDB_H

/** \class DumpFileToDB
 *  Dump the content of a txt file with the format
 *  of ORCA MuBarDigiParameters (see DTCalibrationMap for details)
 *  into a DB. At the moment only the ttrig info is handled.
 *
 *  $Date: 2006/07/03 15:09:40 $
 *  $Revision: 1.2 $
 *  \author G. Cerminara - INFN Torino
 */

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include <string>


class DTCalibrationMap;

class DumpFileToDB : public edm::EDAnalyzer {
public:
  /// Constructor
  DumpFileToDB(const edm::ParameterSet& pset);

  /// Destructor
  virtual ~DumpFileToDB();

  // Operations

  virtual void analyze(const edm::Event& event, const edm::EventSetup& setup){}

  virtual void endJob();

protected:

private:
  const DTCalibrationMap *theCalibFile;

  std::string dbToDump;
};
#endif

