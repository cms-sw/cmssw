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
  void createCollations( const std::vector<std::string>& added_contents );
  /** */
  virtual void histoAnalysis();
  /** */
  virtual void createSummaryHistos( const std::vector<sistrip::SummaryHisto>&, 
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
  /** */
  //inline SummaryHistogramFactory<CommissioningAnalysis::Monitorables>* const factory() const;
  /** */
  //inline void factory( CommissioningSummaryFactory* );
  /** */
  //inline std::map<uint32_t,CommissioningAnalysis::Monitorables*>& data();

 private:
  
  /** */
  MonitorUserInterface* mui_;

  /** Record of collation histos that have been created. */
  std::vector<std::string> collations_;

  /** */
  //SummaryHistogramFactory<CommissioningAnalysis::Monitorables>* factory_;

  /** */
  //std::map<uint32_t,CommissioningAnalysis::Monitorables*> data_;

};

// ----- inline methods -----

MonitorUserInterface* const CommissioningHistograms::mui() const { return mui_; }
const std::vector<std::string>& CommissioningHistograms::collations() const { return collations_; }
//SummaryHistogramFactory<CommissioningAnalysis::Monitorables>* const CommissioningHistograms::factory() const { return factory_; } 
//void CommissioningHistograms::factory( CommissioningSummaryFactory* factory ) { factory_ = factory; } 
//std::map<uint32_t,CommissioningAnalysis::Monitorables*>& CommissioningHistograms::data() { return data_; }

#endif // DQM_SiStripCommissioningClients_CommissioningHistograms_H




