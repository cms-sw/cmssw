#ifndef GUARD_DQM_HCALMONITORTASKS_HCALEXPERTMONITOR_H
#define GUARD_DQM_HCALMONITORTASKS_HCALEXPERTMONITOR_H

#include "DQM/HcalMonitorTasks/interface/HcalBaseMonitor.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "EventFilter/HcalRawToDigi/interface/HcalUnpacker.h"
// The following are needed for using pedestals in fC:
#include "CondFormats/HcalObjects/interface/HcalPedestal.h"
#include "CondFormats/HcalObjects/interface/HcalPedestalWidth.h"

// Raw data stuff
#include "DQM/HcalMonitorTasks/interface/HcalBaseMonitor.h"
#include "EventFilter/HcalRawToDigi/interface/HcalUnpacker.h"
#include "EventFilter/HcalRawToDigi/interface/HcalHTRData.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
#include "DataFormats/FEDRawData/interface/FEDTrailer.h"

// Use for stringstream
#include <iostream>
#include <iomanip>
#include <cmath>

/** \class HcalExpertMonitor
  *
  * $Date: 2008/09/24 21:10:49 $
  * $Revision: 1.1 $
  * \author J. Temple - Univ. of Maryland
  */

class HcalExpertMonitor:  public HcalBaseMonitor {
 public:
  HcalExpertMonitor();
  ~HcalExpertMonitor();

  void unpack(const FEDRawData& raw, const HcalElectronicsMap& emap);
  void setup(const edm::ParameterSet& ps, DQMStore* dbe);
  void reset();
  void clearME();

  // processEvent routine -- specifies what inputs are looked at each event
  void processEvent(const  HBHERecHitCollection& hbheHits,
		    const HORecHitCollection& hoHits, 
		    const HFRecHitCollection& hfHits,
		    const HBHEDigiCollection& hbheDigis,
		    const HODigiCollection& hoDigis,
		    const HFDigiCollection& hfDigis,
		    const HcalTrigPrimDigiCollection& tpDigis,
		    const FEDRawDataCollection& rawraw,
		    const HcalUnpackerReport& report,
		    const HcalElectronicsMap& emap
		    //const ZDCRecHitCollection& zdcHits
		    );
  // Check RecHits each event
  void processEvent_RecHit(const HBHERecHitCollection& hbheHits, 
			   const HORecHitCollection& hoHits, 
			   const HFRecHitCollection& hfHits);
  // Check Digis each event
  void processEvent_Digi(const HBHEDigiCollection & hbheDigis,
			 const HODigiCollection& hoDigis, 
			 const HFDigiCollection& hfDigis,
			 const HcalTrigPrimDigiCollection& tpDigis,
			 const HcalElectronicsMap& emap);
  // Check Raw Data each event
  void processEvent_RawData(const FEDRawDataCollection& rawraw,
			    const HcalUnpackerReport& report,
			    const HcalElectronicsMap& emap);

 private:
  
  int ievt_;
  MonitorElement* meEVT_;

  std::vector <int> fedUnpackList_;
  int firstFED_;
  // Add in histograms here (MonitorElements can handle TH1F, TH2F, TProfile plots)
  MonitorElement* SampleHist;
  MonitorElement* SampleHist2;

}; // class HcalExpertMonitor

#endif  
