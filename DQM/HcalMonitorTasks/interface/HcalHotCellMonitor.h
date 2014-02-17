#ifndef DQM_HCALMONITORTASKS_HCALHOTCELLMONITOR_H
#define DQM_HCALMONITORTASKS_HCALHOTCELLMONITOR_H

#include "DQM/HcalMonitorTasks/interface/HcalBaseDQMonitor.h"
#include "CalibFormats/HcalObjects/interface/HcalDbService.h"
// collection info
#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"

#include <cmath>


/** \class HcalHotCellMonitor
  *
  * $Date: 2012/06/27 13:20:29 $
  * $Revision: 1.45 $
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

class HcalHotCellMonitor: public HcalBaseDQMonitor {

 public:
  HcalHotCellMonitor(const edm::ParameterSet& ps);

  ~HcalHotCellMonitor();

  void setup();
  void beginRun(const edm::Run& run, const edm::EventSetup& c);
  void endRun(const edm::Run& run, const edm::EventSetup& c);
  
  void done();
  void cleanup(void);
  void reset();
  void endJob();

  // analyze function
  void analyze(edm::Event const&e, edm::EventSetup const&s);

  // Begin LumiBlock
  void beginLuminosityBlock(const edm::LuminosityBlock& lumiSeg,
                            const edm::EventSetup& c) ;

  // End LumiBlock
  void endLuminosityBlock(const edm::LuminosityBlock& lumiSeg,
                          const edm::EventSetup& c);

  void processEvent(const HBHERecHitCollection& hbHits,
                    const HORecHitCollection& hoHits,
                    const HFRecHitCollection& hfHits
		    );

  void processEvent_rechitenergy( const HBHERecHitCollection& hbheHits,
                                  const HORecHitCollection& hoHits,
                                  const HFRecHitCollection& hfHits);

  template <class R, class C> void processHit_rechitNeighbors(R& rechititer,
							      C& collection,
							      hotNeighborParams& params);

  void periodicReset();


 private:
  void fillNevents_neighbor();
  void fillNevents_energy();
  void fillNevents_persistentenergy();
  
  void fillNevents_problemCells();
  void zeroCounters();

  int minEvents_; // minimum # of events in a lumi block before persistent test will be checked
  // Booleans to control which of the three hot cell checking routines are used
  bool test_neighbor_;
  bool test_energy_;
  bool test_et_;
  bool test_persistent_;
  bool setupDone_;
  
  double energyThreshold_, HBenergyThreshold_, HEenergyThreshold_, HOenergyThreshold_, HFenergyThreshold_;
  double ETThreshold_, HBETThreshold_, HEETThreshold_, HOETThreshold_, HFETThreshold_;
  double persistentThreshold_, HBpersistentThreshold_, HEpersistentThreshold_, HOpersistentThreshold_, HFpersistentThreshold_;
  double persistentETThreshold_, HBpersistentETThreshold_, HEpersistentETThreshold_, HOpersistentETThreshold_, HFpersistentETThreshold_;

  double HFfarfwdScale_;

  double minErrorFlag_; // minimum error rate needed to dump out bad bin info 
  
  double nsigma_;
  double HBnsigma_, HEnsigma_, HOnsigma_, HFnsigma_;
  EtaPhiHists   AboveNeighborsHotCellsByDepth;
  EtaPhiHists   AboveEnergyThresholdCellsByDepth;
  EtaPhiHists   AboveETThresholdCellsByDepth;
  EtaPhiHists   AbovePersistentThresholdCellsByDepth; 
  EtaPhiHists   AbovePersistentETThresholdCellsByDepth;

  double SiPMscale_;
  int aboveneighbors[85][72][4];
  int aboveenergy[85][72][4]; // when rechit is above threshold energy
  int aboveet[85][72][4]; // when rechit is above threshold et
  int abovepersistent[85][72][4]; // when rechit is consistently above some threshold
  int abovepersistentET[85][72][4];
  int rechit_occupancy_sum[85][72][4];

  bool excludeHORing2_;


  // Diagnostic plots
  MonitorElement* d_HBenergyVsNeighbor;
  MonitorElement* d_HEenergyVsNeighbor;
  MonitorElement* d_HOenergyVsNeighbor;
  MonitorElement* d_HFenergyVsNeighbor;

  int hbVsNeighbor[500];
  int heVsNeighbor[500];
  int hoVsNeighbor[500];
  int hfVsNeighbor[500];

  hotNeighborParams HBHENeighborParams_, HONeighborParams_, HFNeighborParams_;

  edm::InputTag hbheRechitLabel_, hoRechitLabel_, hfRechitLabel_;
};

#endif
