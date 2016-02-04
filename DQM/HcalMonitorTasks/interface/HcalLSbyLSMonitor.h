#ifndef DQM_HCALMONITORTASKS_HCALLSBYLSMONITOR_H
#define DQM_HCALMONITORTASKS_HCALLSBYLSMONITOR_H

#include "DQM/HcalMonitorTasks/interface/HcalBaseDQMonitor.h"
#include <cmath>


/** \class HcalLSbyLSMonitor
  *
  * $Date: 2010/05/07 17:38:16 $
  * $Revision: 1.2 $
  * \author J. Temple - Univ. of Maryland
  */

struct hotNeighborParams{
  int DeltaIphi;
  int DeltaIeta;
  int DeltaDepth;
  double minCellEnergy; // cells below this threshold can never be considered "hot" by this algorithm
  double minNeighborEnergy; //neighbors must have some amount of energy to be counted
  double maxEnergy; //  a cell above this energy will always be considered hot
  double HotEnergyFrac; // a cell will be considered hot if neighbor energy/ cell energy is less than this value
};

class HcalLSbyLSMonitor: public HcalBaseDQMonitor {

 public:
  HcalLSbyLSMonitor(const edm::ParameterSet& ps);

  ~HcalLSbyLSMonitor();

  void setup();
  void beginRun(const edm::Run& run, const edm::EventSetup& c);
  void endRun(const edm::Run& run, const edm::EventSetup& c){};
  
  void done();
  void cleanup(void);
  void reset();
  void endJob();

  // analyze function
  //void analyze(edm::Event const&e, edm::EventSetup const&s){};

  // Begin LumiBlock
  void beginLuminosityBlock(const edm::LuminosityBlock& lumiSeg,
                            const edm::EventSetup& c) ;

  // End LumiBlock
  void endLuminosityBlock(const edm::LuminosityBlock& lumiSeg,
                          const edm::EventSetup& c);

  void periodicReset(){};


 private:
  int minEvents_;
  std::vector<std::string> TaskList_;
};

#endif
