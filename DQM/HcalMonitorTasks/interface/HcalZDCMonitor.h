#ifndef DQM_HCALMONITORTASKS_HCALZDCMONITOR_H
#define DQM_HCALMONITORTASKS_HCALZDCMONITOR_H

#include "DQM/HcalMonitorTasks/interface/HcalBaseMonitor.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "EventFilter/HcalRawToDigi/interface/HcalUnpacker.h"
#include <cmath>

/** \class HcalZDCMonitor
 *
 * $DATE: 2009/05/01 16:37:00 %
 * $Revision: 1.1.2.4 $
 * \author J. Temple -- Univ. of Maryland
 */

class HcalZDCMonitor: public HcalBaseMonitor
{
 public:
  HcalZDCMonitor();
  ~HcalZDCMonitor();
  void setup(const edm::ParameterSet& ps, DQMStore* dbe);
  void processEvent(const ZDCDigiCollection& digi,
                    const ZDCRecHitCollection& rechit);
  void reset();
  void done();
 private:
  void zeroCounters();
  void fillHistos();
  void setZDClabels(MonitorElement* h);

  // Should ievt_, meEVT_ be part of base class?
  int ievt_;
  int zdc_checkNevents_;
  MonitorElement* meEVT_;

  double deadthresh_;
  MonitorElement* ProblemZDC_;
  MonitorElement* avgoccZDC_;
  std::vector<MonitorElement*> timeZDC_;
  MonitorElement* avgtimeZDC_;
  std::vector<MonitorElement*> energyZDC_;
  MonitorElement* avgenergyZDC_;
  MonitorElement* avgXplus_;
  MonitorElement* avgXminus_;
};

#endif
