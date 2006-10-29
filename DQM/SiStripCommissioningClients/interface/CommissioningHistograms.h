#ifndef DQM_SiStripCommissioningClients_CommissioningHistograms_H
#define DQM_SiStripCommissioningClients_CommissioningHistograms_H

#include "DataFormats/SiStripDetId/interface/SiStripControlKey.h"
#include "DataFormats/SiStripDetId/interface/SiStripReadoutKey.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMServices/Core/interface/MonitorUserInterface.h"
#include "DQM/SiStripCommon/interface/SiStripHistoNamingScheme.h"
#include "DQM/SiStripCommon/interface/SiStripEnumeratedTypes.h"
#include "DQM/SiStripCommon/interface/ExtractTObject.h"
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
  
  typedef std::vector<std::string> Collations;
  typedef std::map<uint32_t,Collations> CollationsMap;
  typedef std::map<uint32_t,uint32_t> FedToFecMap;
  
  /** */
  CommissioningHistograms( MonitorUserInterface* );
  /** */
  virtual ~CommissioningHistograms();

  // ---------- "Actions" on MonitorElements ----------

  /** */
  void createCollations( const std::vector<std::string>& contents );
  /** */
  virtual void histoAnalysis( bool debug );
  /** */
  virtual void createSummaryHisto( const sistrip::SummaryHisto&, 
				   const sistrip::SummaryType&, 
				   const std::string& top_level_dir,
				   const sistrip::Granularity& );
  /** */
  virtual void uploadToConfigDb();
  
  /** Wraps virtual createSummaryHisto() method for Seal::Callback. */
  void createSummaryHisto( std::pair<sistrip::SummaryHisto,
			   sistrip::SummaryType>, 
			   std::pair<std::string,
			   sistrip::Granularity> ); 
  
 protected:
  
  /** */
  inline MonitorUserInterface* const mui() const;
  /** */
  inline const CollationsMap& collations() const;
  /** */
  inline const FedToFecMap mapping() const;
  
  TH1* histogram( const sistrip::SummaryHisto&, 
		  const sistrip::SummaryType&, 
		  const sistrip::View&,
		  const string& directory,
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
  
};

// ----- inline methods -----

MonitorUserInterface* const CommissioningHistograms::mui() const { return mui_; }
const CommissioningHistograms::CollationsMap& CommissioningHistograms::collations() const { return collations_; }
const CommissioningHistograms::FedToFecMap CommissioningHistograms::mapping() const { return mapping_; }

#endif // DQM_SiStripCommissioningClients_CommissioningHistograms_H



