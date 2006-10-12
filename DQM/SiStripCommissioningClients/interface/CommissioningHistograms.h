#ifndef DQM_SiStripCommissioningClients_CommissioningHistograms_H
#define DQM_SiStripCommissioningClients_CommissioningHistograms_H

#include "DataFormats/SiStripDetId/interface/SiStripControlKey.h"
#include "DataFormats/SiStripDetId/interface/SiStripReadoutKey.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMServices/Core/interface/MonitorUserInterface.h"
#include "DQM/SiStripCommon/interface/SiStripHistoNamingScheme.h"
#include "DQM/SiStripCommon/interface/ExtractTObject.h"
#include "DQM/SiStripCommissioningSummary/interface/CommissioningSummary.h"
#include "DQM/SiStripCommissioningSummary/interface/SummaryFactory.h"
#include <boost/cstdint.hpp>
#include "TProfile.h"
#include "TH1F.h"
#include <iostream>
#include <string>
#include <vector>
#include <map>

class MonitorUserInterface;
class SiStripSummary;

class CommissioningHistograms {

 public:

  /** */
  CommissioningHistograms( MonitorUserInterface* );
  /** */
  virtual ~CommissioningHistograms();

  /** */
  void createCollations( const std::vector<std::string>& added_contents );
  /** */
  virtual void histoAnalysis();
  /** */
  virtual void createSummaryHistos( const std::vector<SummaryFactory::Histo>&, 
				    const std::string& directory );
  /** */
  virtual void createTrackerMap();
  /** */
  virtual void uploadToConfigDb();
  
 protected:
  
  /** */
  inline MonitorUserInterface* const mui() const;
  /** */
  inline const std::vector<std::string>& collations() const;
  
 private:
  
  /** */
  MonitorUserInterface* mui_;
  /** Record of collation histos that have been created. */
  std::vector<std::string> collations_;
  
};

// ----- inline methods -----

MonitorUserInterface* const CommissioningHistograms::mui() const { return mui_; }
const std::vector<std::string>& CommissioningHistograms::collations() const { return collations_; }

#endif // DQM_SiStripCommissioningClients_CommissioningHistograms_H




