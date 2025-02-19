#ifndef DQM_HCALMONITORTASKS_HCALCOARSEPEDESTALDQMONITOR_H
#define DQM_HCALMONITORTASKS_HCALCOARSEPEDESTALDQMONITOR_H

#include "DQM/HcalMonitorTasks/interface/HcalBaseDQMonitor.h"
#include "EventFilter/HcalRawToDigi/interface/HcalUnpacker.h"
#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"
#include "CalibFormats/HcalObjects/interface/HcalDbService.h"
#include "CalibFormats/HcalObjects/interface/HcalDbRecord.h"

/** \class HcalCoarsePedestalMonitor
  *  
  * $Date: 2010/11/17 19:17:30 $
  * $Revision: 1.5 $
  * \author J. Temple - Univ. of Maryland
  */


class HcalCoarsePedestalMonitor: public HcalBaseDQMonitor {
public:
  HcalCoarsePedestalMonitor(const edm::ParameterSet& ps); 
  ~HcalCoarsePedestalMonitor(); 

  void setup();
  void beginRun(const edm::Run& run, const edm::EventSetup& c);
  void cleanup();

  void analyze(const edm::Event& e, const edm::EventSetup& c);

  void processEvent(const HBHEDigiCollection& hbhe,
		    const HODigiCollection& ho,
		    const HFDigiCollection& hf,
		    const HcalUnpackerReport& report);


  // Begin LumiBlock
  void beginLuminosityBlock(const edm::LuminosityBlock& lumiSeg,
                            const edm::EventSetup& c) ;

  // End LumiBlock
  void endLuminosityBlock(const edm::LuminosityBlock& lumiSeg,
                          const edm::EventSetup& c);


  void endRun(const edm::Run& run, const edm::EventSetup& c);
  void endJob();
  void reset();

private:  ///Methods, variables accessible only within class code
 
  void fill_Nevents();
  void zeroCounters();

  // Store sum of pedestal values over all events
  EtaPhiHists CoarsePedestalsSumByDepth;
  EtaPhiHists CoarsePedestalsOccByDepth;

  double pedestalsum_[85][72][4]; // sum of pedestal values over all events
  int pedestalocc_[85][72][4];

  double ADCDiffThresh_; // store difference value that causes channel to be considered in error

  edm::InputTag digiLabel_;
  int minEvents_;
  bool excludeHORing2_;

};

#endif
