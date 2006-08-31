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
  
  typedef std::vector<std::string> Collations;
  typedef std::map<uint32_t,Collations> CollationsMap;
  typedef std::map<uint32_t,uint32_t> FedToFecMap;
  
  /** */
  CommissioningHistograms( MonitorUserInterface* );
  /** */
  virtual ~CommissioningHistograms();

  // ---------- General "actions" ----------

/*   /\** *\/ */
/*   static void subscribe( MonitorUserInterface*, */
/* 			 std::string match_pattern ); */
/*   /\** *\/ */
/*   static void unsubscribe( MonitorUserInterface*, */
/* 			   std::string match_pattern ); */
/*   /\** *\/ */
/*   static void saveHistos( MonitorUserInterface*, */
/* 			  std::string filename ); */
  
  // ---------- "Actions" on MonitorElements ----------

  /** */
  void createCollations( const std::vector<std::string>& contents );
  /** */
  virtual void histoAnalysis();
  /** */
  virtual void createSummaryHisto( const sistrip::SummaryHisto&, 
				   const sistrip::SummaryType&, 
				   const std::string& directory );
  /** */
  virtual void uploadToConfigDb();
  
  /** Wraps virtual createSummaryHisto() method for Seal::Callback. */
  void createSummaryHisto( std::pair<sistrip::SummaryHisto,
			   sistrip::SummaryType> summ, 
			   std::string directory ); 
  
 protected:
  
  /** */
  inline MonitorUserInterface* const mui() const;
  /** */
  inline const CollationsMap& collations() const;
  /** */
  inline const FedToFecMap mapping() const;
  
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



/* /\** Simple container class to hold summary histo criteria. *\/ */
/* class Summary { */
/*   sistrip::SummaryHisto histo_; */
/*   sistrip::SummaryType type_; */
/*   std::string dir_; */
/*  public: */
/*   Summary( sistrip::SummaryHisto histo, */
/* 	   sistrip::SummaryType type, */
/* 	   std::string dir ) :  */
/*     histo_(histo), type_(type), dir_(dir) {;} */
/*   Summary( const Summary& s ) {  */
/*     histo_ = s.histo_;  */
/*     type_ = s.type_;  */
/*     dir_ = s.dir_; */
/*   }  */
/*   Summary() :  */
/*     histo_(sistrip::UNKNOWN_SUMMARY_HISTO), */
/*     type_(sistrip::UNKNOWN_SUMMARY_TYPE), */
/*     dir_("") {;} */
/*   Summary& operator= ( const Summary& s ) { return *this; } */
/* }; */

/* void createSummaryHisto( CommissioningHistograms::Summary ) {;}  */



