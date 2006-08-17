#ifndef DQM_SiStripCommissioningClients_CommissioningHistograms_H
#define DQM_SiStripCommissioningClients_CommissioningHistograms_H

#include "DataFormats/SiStripDetId/interface/SiStripControlKey.h"
#include "DataFormats/SiStripDetId/interface/SiStripReadoutKey.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMServices/Core/interface/MonitorUserInterface.h"
//#include "DQM/SiStripCommon/interface/SummaryHistogramFactory.h"
#include "DQM/SiStripCommon/interface/SiStripHistoNamingScheme.h"
#include "DQM/SiStripCommon/interface/SiStripEnumeratedTypes.h"
#include "DQM/SiStripCommon/interface/ExtractTObject.h"
//#include "DQM/SiStripCommissioningSummary/src/CommissioningSummaryFactory.cc"
//#include "DQM/SiStripCommissioningSummary/interface/SummaryHistogram.h"
//#include "DQM/SiStripCommissioningAnalysis/interface/CommissioningAnalysis.h"
#include <boost/cstdint.hpp>
#include "TProfile.h"
#include "TH1.h"
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
  
  /** */
  void subscribeNew();
  /** */
  void createCollations( const std::vector<std::string>& contents );
  /** */
  virtual void histoAnalysis();
  
  /** */
  virtual void saveHistos( std::string filename );
  /** */
  virtual void createSummaryHisto( const sistrip::SummaryHisto&, 
				   const sistrip::SummaryType&, 
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

  /** */
  sistrip::Action action_;
  
};

// ----- inline methods -----

MonitorUserInterface* const CommissioningHistograms::mui() const { return mui_; }
const std::vector<std::string>& CommissioningHistograms::collations() const { return collations_; }

#endif // DQM_SiStripCommissioningClients_CommissioningHistograms_H




