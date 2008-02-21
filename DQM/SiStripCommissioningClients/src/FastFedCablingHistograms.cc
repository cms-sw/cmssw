#include "DQM/SiStripCommissioningClients/interface/FastFedCablingHistograms.h"
#include "CondFormats/SiStripObjects/interface/FastFedCablingAnalysis.h"
#include "DataFormats/SiStripCommon/interface/SiStripConstants.h"
#include "DQM/SiStripCommissioningSummary/interface/FastFedCablingSummaryFactory.h"
#include "DQM/SiStripCommon/interface/ExtractTObject.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <iostream>
#include <sstream>
#include <iomanip>
#include "TProfile.h"


using namespace std;
using namespace sistrip;

// -----------------------------------------------------------------------------
/** */
FastFedCablingHistograms::FastFedCablingHistograms( MonitorUserInterface* mui ) 
  : CommissioningHistograms( mui, sistrip::FAST_CABLING )
{
  factory_ = auto_ptr<FastFedCablingSummaryFactory>( new FastFedCablingSummaryFactory );
  LogTrace(mlDqmClient_) 
    << "[FastFedCablingHistograms::" << __func__ << "]"
    << " Constructing object...";
}

// -----------------------------------------------------------------------------
/** */
FastFedCablingHistograms::FastFedCablingHistograms( DaqMonitorBEInterface* bei ) 
  : CommissioningHistograms( bei, sistrip::FAST_CABLING )
{
  LogTrace(mlDqmClient_) 
    << "[FastFedCablingHistograms::" << __func__ << "]"
    << " Constructing object...";
}

// -----------------------------------------------------------------------------
/** */
FastFedCablingHistograms::~FastFedCablingHistograms() {
  LogTrace(mlDqmClient_) 
    << "[FastFedCablingHistograms::" << __func__ << "]"
    << " Destructing object...";
}

// -----------------------------------------------------------------------------	 
/** */	 
void FastFedCablingHistograms::histoAnalysis( bool debug ) {
  LogTrace(mlDqmClient_)
    << "[FastFedCablingHistograms::" << __func__ << "]";

  // Some initialisation
  uint16_t valid = 0;
  HistosMap::const_iterator iter;
  Analyses::iterator ianal;
  std::map<std::string,uint16_t> errors;
  
  // Clear map holding analysis objects
  for ( ianal = data().begin(); ianal != data().end(); ianal++ ) { 
    if ( ianal->second ) { delete ianal->second; }
  } 
  data().clear();
  
  // Iterate through map containing histograms
  for ( iter = histos().begin(); 
	iter != histos().end(); iter++ ) {
    
    // Check vector of histos is not empty
    if ( iter->second.empty() ) {
      edm::LogWarning(mlDqmClient_)
	<< "[FastFedCablingHistograms::" << __func__ << "]"
	<< " Zero histograms found!";
      continue;
    }
    
    // Retrieve pointers to histos
    std::vector<TH1*> profs;
    Histos::const_iterator ihis = iter->second.begin(); 
    for ( ; ihis != iter->second.end(); ihis++ ) {
      TProfile* prof = ExtractTObject<TProfile>().extract( (*ihis)->me_ );
      if ( prof ) { profs.push_back(prof); }
    } 
    
    // Perform histo analysis
    FastFedCablingAnalysis* anal = new FastFedCablingAnalysis( iter->first );
    FedToFecMap::const_iterator ifed = mapping().find( iter->first );
    if ( ifed != mapping().end() ) { anal->fecKey( ifed->second ); }
    anal->analysis( profs );
    data()[iter->first] = anal; 
    if ( anal->isValid() ) { valid++; }
    if ( !anal->getErrorCodes().empty() ) { 
      errors[anal->getErrorCodes()[0]]++;
    }

  }
  
  if ( !histos().empty() ) {
    edm::LogVerbatim(mlDqmClient_) 
      << "[FastFedCablingHistograms::" << __func__ << "]"
      << " Analyzed histograms for " << histos().size() 
      << " FED channels, of which " << valid 
      << " (" << 100 * valid / histos().size()
      << "%) are valid.";
    if ( !errors.empty() ) {
      uint16_t count = 0;
      std::stringstream ss;
      ss << std::endl;
      std::map<std::string,uint16_t>::const_iterator ii;
      for ( ii = errors.begin(); ii != errors.end(); ++ii ) { 
	ss << " " << ii->first << ": " << ii->second << std::endl;
	count += ii->second;
      }
      edm::LogWarning(mlDqmClient_) 
	<< "[FastFedCablingHistograms::" << __func__ << "]"
	<< " Found " << count << " error strings: "
	<< ss.str();
    }
  } else {
    edm::LogWarning(mlDqmClient_) 
      << "[FastFedCablingHistograms::" << __func__ << "]"
      << " No histograms to analyze!";
  }
  
}
