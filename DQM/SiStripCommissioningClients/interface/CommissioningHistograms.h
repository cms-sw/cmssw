#ifndef DQM_SiStripCommissioningClients_CommissioningHistograms_H
#define DQM_SiStripCommissioningClients_CommissioningHistograms_H

#include "DataFormats/SiStripDetId/interface/SiStripControlKey.h"
#include "DataFormats/SiStripDetId/interface/SiStripReadoutKey.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMServices/Core/interface/MonitorUserInterface.h"
#include "DQM/SiStripCommon/interface/SiStripHistoNamingScheme.h"
#include "DQM/SiStripCommon/interface/ExtractTObject.h"
#include "DQM/SiStripCommissioningSummary/interface/CommissioningSummary.h"
#include <boost/cstdint.hpp>
#include "TProfile.h"
#include "TH1F.h"
#include <iostream>
#include <string>
#include <vector>
#include <map>

class MonitorUserInterface;

class CommissioningHistograms {

 public:

  /** */
  CommissioningHistograms( MonitorUserInterface* );
  /** */
  virtual ~CommissioningHistograms();

  // ----- "Actions" -----
  
  /** */
  virtual void createSummaryHistos();
  /** */
  virtual void createTrackerMap();
  /** */
  virtual void uploadToConfigDb();
  
 protected:
  
  /** */
  inline MonitorUserInterface* const mui() const;
  
 private:
  
  /** */
  MonitorUserInterface* mui_;
  
};

// ----- inline methods -----

MonitorUserInterface* const CommissioningHistograms::mui() const { return mui_; }

#endif // DQM_SiStripCommissioningClients_CommissioningHistograms_H




