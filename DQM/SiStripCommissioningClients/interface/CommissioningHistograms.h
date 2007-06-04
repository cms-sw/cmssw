#ifndef DQM_SiStripCommissioningClients_CommissioningHistograms_H
#define DQM_SiStripCommissioningClients_CommissioningHistograms_H

#include "DataFormats/SiStripCommon/interface/SiStripConstants.h"
#include "DataFormats/SiStripCommon/interface/SiStripFecKey.h"
#include "DataFormats/SiStripCommon/interface/SiStripFedKey.h"
#include "DataFormats/SiStripCommon/interface/SiStripHistoTitle.h"
#include "DataFormats/SiStripCommon/interface/SiStripEnumsAndStrings.h"
#include "DQM/SiStripCommon/interface/ExtractTObject.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMServices/Core/interface/CollateMonitorElement.h"
#include "DQMServices/Core/interface/MonitorUserInterface.h"
#include "DQMServices/Core/interface/DaqMonitorBEInterface.h"
#include <boost/cstdint.hpp>
#include "TProfile.h"
#include "TH1.h"
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <map>

class MonitorElement;

class CommissioningHistograms {

 public:

  // ---------- Constructors, destructors ----------
  
  /** */
  CommissioningHistograms( MonitorUserInterface* const,
			   const sistrip::RunType& );
  
  /** */
  CommissioningHistograms( DaqMonitorBEInterface* const,
			   const sistrip::RunType& );
  
  /** */
  virtual ~CommissioningHistograms();

  // ---------- Structs, typedefs ----------
  
  /** Simple container class for histograms. */
  class Histo {
  public:
    Histo( const std::string& title, 
	   MonitorElement* const me,
	   CollateMonitorElement* const cme ) 
      : title_(title), me_(me), cme_(cme) {;}
    Histo() : title_(""), me_(0), cme_(0) {;}
    void print( std::stringstream& ) const;
    std::string title_;
    MonitorElement* me_;
    CollateMonitorElement* cme_;
  };
  
  typedef std::vector<Histo*> Histos;
  typedef std::map<uint32_t,Histos> HistosMap;
  typedef std::map<uint32_t,uint32_t> FedToFecMap;

  // ---------- Generic static methods for clients ----------
  
  /** Extracts run number from list of MonitorElements. */
  static uint32_t runNumber( DaqMonitorBEInterface* const,
			     const std::vector<std::string>& );
  
  /** Extracts run type from list of MonitorElements. */
  static sistrip::RunType runType( DaqMonitorBEInterface* const,
				   const std::vector<std::string>& );
  
  /** Retrieves list of histograms in form of strings. */
  static void getContents( DaqMonitorBEInterface* const,
			   std::vector<std::string>& );
  
  // ---------- Client "actions" on histograms ----------
  
  /** */
  void extractHistograms( const std::vector<std::string>& );
  /** */
  void createCollations( const std::vector<std::string>& );
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
  
  /** */
  void remove( std::string pattern = "" ); 
  
  /** */
  void save( std::string& filename,
	     uint32_t run_number = 0 ); 
  
 protected:

  // ---------- Protected methods ----------

  /** */
  inline MonitorUserInterface* const mui() const;
  /** */
  inline DaqMonitorBEInterface* const bei() const;
  /** */
  inline const HistosMap& histos() const;
  /** */
  inline const FedToFecMap& mapping() const;
  /** */
  inline const sistrip::RunType& task() const;
  
  TH1* histogram( const sistrip::Monitorable&, 
		  const sistrip::Presentation&, 
		  const sistrip::View&,
		  const std::string& directory,
		  const uint32_t& xbins );
  
  /** */
  void clearHistosMap();
  
  /** */
  void printHistosMap();

 private:

  // ---------- Protected data ----------
  
  /** */
  CommissioningHistograms();
  
  /** */
  MonitorUserInterface* mui_;

  /** */
  DaqMonitorBEInterface* bei_;
  
  /** Record of collation histos that have been created. */
  HistosMap histos_;
  
  /** Map b/w FED and FEC keys from histo dirs/names. */
  FedToFecMap mapping_;
  
  sistrip::RunType task_;
  
};

// ---------- inline methods ----------

MonitorUserInterface* const CommissioningHistograms::mui() const { return mui_; }
DaqMonitorBEInterface* const CommissioningHistograms::bei() const { return bei_; }
const CommissioningHistograms::HistosMap& CommissioningHistograms::histos() const { return histos_; }
const CommissioningHistograms::FedToFecMap& CommissioningHistograms::mapping() const { return mapping_; }
const sistrip::RunType& CommissioningHistograms::task() const { return task_; }

#endif // DQM_SiStripCommissioningClients_CommissioningHistograms_H
