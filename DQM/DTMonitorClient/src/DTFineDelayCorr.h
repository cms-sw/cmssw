#ifndef DTFineDelayCorr_H
#define DTFineDelayCorr_H

/** \class DTFineDelayCorr
 * *
 *  Class to calculate corrections to the fine delays loaded in DT MiniCrates
 *
 *  $Date: 2010/01/07 10:51:14 $
 *  $Revision: 1.2 $
 *  \author  M. Giunta, C. Battilana
 *   
 */

#include "DQM/DTMonitorClient/src/DTLocalTriggerBaseTest.h"
#include "FWCore/Framework/interface/ESHandle.h"
// Geometry
#include "Geometry/DTGeometry/interface/DTGeometry.h"
// L1Trigger
#include "L1TriggerConfig/DTTPGConfig/interface/DTConfigManager.h"
#include "L1TriggerConfig/DTTPGConfig/interface/DTConfigManagerRcd.h"
// DB & Calib
#include <CondFormats/DTObjects/interface/DTTPGParameters.h>
#include <CondFormats/DataRecord/interface/DTTPGParametersRcd.h>

#include <map>
#include <vector>

class DTFineDelayCorr: public DTLocalTriggerBaseTest{

public:

  /// Constructor
  DTFineDelayCorr(const edm::ParameterSet& ps);
  
  /// Destructor
  virtual ~DTFineDelayCorr();

protected:
  /// Begin Job
  void beginJob();

  /// Begin Run
  void beginRun(const edm::Run& run, const edm::EventSetup& evSU);

  /// End Job
  void endJob();
 
  /// DQM Client Diagnostic
  void runClientDiagnostic();
  
  // Write the new delays to a file 
  void writeConsts(const std::string& outputFileName) const;
  

 private:
  std::string t0MeanHistoTag;
  std::string outputFileName;
  std::string oldDelaysInputFile;
  std::string trSource;
  bool readOldFromDb;
  bool writeDB;
  bool gaussMean;
  int minEntries;
  int nEvents;
  edm::ESHandle< DTConfigManager > dtConfig;
  edm::ESHandle< DTTPGParameters > worstPhaseMap;

// The map between the Chamber and the old delays
  std::map< DTChamberId, std::pair<int,float> > oldDelayMap;

// The map between the Chamber and the new delays
  std::map< DTChamberId, std::vector<float> > delayMap;

};

#endif
