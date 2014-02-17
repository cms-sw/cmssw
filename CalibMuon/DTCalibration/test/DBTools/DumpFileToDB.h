#ifndef DumpFileToDB_H
#define DumpFileToDB_H

/** \class DumpFileToDB
 *  Dump the content of a txt file with the format
 *  of ORCA MuBarDigiParameters (see DTCalibrationMap for details)
 *  into a DB. At the moment only the ttrig info is handled.
 *
 *  $Date: 2010/05/25 18:34:19 $
 *  $Revision: 1.5 $
 *  \author G. Cerminara - INFN Torino
 */

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include <string>
#include <fstream>
#include <vector>

class DTCalibrationMap;
class DTTtrig;

class DumpFileToDB : public edm::EDAnalyzer {
public:
  /// Constructor
  DumpFileToDB(const edm::ParameterSet& pset);

  /// Destructor
  virtual ~DumpFileToDB();

  // Operations
 // Operations
  virtual void beginRun(const edm::Run& run, const edm::EventSetup& setup );

  virtual void analyze(const edm::Event& event, const edm::EventSetup& setup){}
 
  virtual void endJob();

protected:

private:
  std::vector<int> readChannelsMap (std::stringstream &linestr);

  const DTCalibrationMap *theCalibFile;
  std::string mapFileName;

  std::string dbToDump;
  // sum the correction in the txt file (for the mean value) to what is input DB 
  bool diffMode;
  const DTTtrig *tTrigMapOrig;


};
#endif

