#ifndef DQM_SiStripCommissioningClients_CommissioningHistograms_H
#define DQM_SiStripCommissioningClients_CommissioningHistograms_H

#include "DataFormats/SiStripCommon/interface/SiStripEnumeratedTypes.h"
#include "DataFormats/SiStripCommon/interface/SiStripFecKey.h"
#include "DataFormats/SiStripCommon/interface/SiStripFedKey.h"
#include "DataFormats/SiStripCommon/interface/SiStripHistoNamingScheme.h"
#include "DQM/SiStripCommon/interface/ExtractTObject.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMServices/Core/interface/CollateMonitorElement.h"
#include "DQMServices/Core/interface/MonitorUserInterface.h"
#include <boost/cstdint.hpp>
#include "TProfile.h"
#include "TH1.h"
#include <iostream>
#include <string>
#include <vector>
#include <map>

class MonitorUserInterface;
class CollateMonitorElement;

class CommissioningHistograms {

 public:
  
  typedef std::pair<std::string,CollateMonitorElement*> Collation;
  typedef std::vector<Collation> Collations;
  typedef std::map<uint32_t,Collations> CollationsMap;
  typedef std::map<uint32_t,uint32_t> FedToFecMap;
  
  /** */
  CommissioningHistograms( MonitorUserInterface*,
			   const sistrip::Task& );
  /** */
  virtual ~CommissioningHistograms();

  // ---------- "Actions" on MonitorElements ----------

  /** */
  void createCollations( const std::vector<std::string>& contents );
  /** */
  virtual void histoAnalysis( bool debug );
  /** */
  virtual void createSummaryHisto( const sistrip::Monitorable&, 
				   const sistrip::Presentation&, 
				   const std::string& top_level_dir,
				   const sistrip::Granularity& );
  /** */
  virtual void uploadToConfigDb();
  
  /** Wraps virtual createSummaryHisto() method for Seal::Callback. */
  void createSummaryHisto( std::pair<sistrip::Monitorable,
			   sistrip::Presentation>, 
			   std::pair<std::string,
			   sistrip::Granularity> ); 
  
 protected:
  
  /** */
  inline MonitorUserInterface* const mui() const;
  /** */
  inline const CollationsMap& collations() const;
  /** */
  inline const FedToFecMap& mapping() const;
  /** */
  inline const sistrip::Task& task() const;
  
  TH1* histogram( const sistrip::Monitorable&, 
		  const sistrip::Presentation&, 
		  const sistrip::View&,
		  const std::string& directory,
		  const uint32_t& xbins );

 private:
  
  CommissioningHistograms();
  
  /** */
  MonitorUserInterface* mui_;

  /** Record of collation histos that have been created. */
  CollationsMap collations_;
  
  /** Mapping between FED and FEC keys. */
  FedToFecMap mapping_;
  
  /** */
  sistrip::Action action_;

  sistrip::Task task_;
  
};

// ----- inline methods -----

MonitorUserInterface* const CommissioningHistograms::mui() const { return mui_; }
const CommissioningHistograms::CollationsMap& CommissioningHistograms::collations() const { return collations_; }
const CommissioningHistograms::FedToFecMap& CommissioningHistograms::mapping() const { return mapping_; }
const sistrip::Task& CommissioningHistograms::task() const { return task_; }

#endif // DQM_SiStripCommissioningClients_CommissioningHistograms_H



