#include "DQM/SiStripCommissioningClients/interface/OptoScanHistograms.h"
#include "CondFormats/SiStripObjects/interface/OptoScanAnalysis.h"
#include "DataFormats/SiStripCommon/interface/SiStripConstants.h"
#include "DQM/SiStripCommissioningSummary/interface/OptoScanSummaryFactory.h"
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
OptoScanHistograms::OptoScanHistograms( DQMOldReceiver* mui ) 
  : CommissioningHistograms( mui, sistrip::OPTO_SCAN )
{
  factory_ = auto_ptr<OptoScanSummaryFactory>( new OptoScanSummaryFactory );
  LogTrace(mlDqmClient_) 
    << "[OptoScanHistograms::" << __func__ << "]"
    << " Constructing object...";
}

// -----------------------------------------------------------------------------
/** */
OptoScanHistograms::OptoScanHistograms( DQMStore* bei ) 
  : CommissioningHistograms( bei, sistrip::OPTO_SCAN )
{
  LogTrace(mlDqmClient_) 
    << "[OptoScanHistograms::" << __func__ << "]"
    << " Constructing object...";
}

// -----------------------------------------------------------------------------
/** */
OptoScanHistograms::~OptoScanHistograms() {
  LogTrace(mlDqmClient_) 
    << "[OptoScanHistograms::" << __func__ << "]"
    << " Denstructing object...";
}

// -----------------------------------------------------------------------------	 
/** */	 
void OptoScanHistograms::histoAnalysis( bool debug ) {
  LogTrace(mlDqmClient_)
    << "[OptoScanHistograms::" << __func__ << "]";

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
	<< "[OptoScanHistograms::" << __func__ << "]"
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
    OptoScanAnalysis* anal = new OptoScanAnalysis( iter->first );
    anal->analysis( profs );
    data()[iter->first] = anal; 
    if ( anal->isValid() ) { valid++; }
    if ( !anal->getErrorCodes().empty() ) { 
      errors[anal->getErrorCodes()[0]]++;
    }
    
  }
  
  if ( !histos().empty() ) {
    edm::LogVerbatim(mlDqmClient_) 
      << "[OptoScanHistograms::" << __func__ << "]"
      << " Analyzed histograms for " << histos().size() 
      << " FED channels, of which " << valid 
      << " (" << 100 * valid / histos().size()
      << "%) are valid.";
  } else {
    edm::LogWarning(mlDqmClient_) 
      << "[OptoScanHistograms::" << __func__ << "]"
      << " No histograms to analyze!";
  }

  if ( !histos().empty() ) {
    edm::LogVerbatim(mlDqmClient_) 
      << "[OptoScanHistograms::" << __func__ << "]"
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
	<< "[OptoScanHistograms::" << __func__ << "]"
	<< " Found " << count << " errors ("
	<< 100 * count / histos().size() << "%): " 
	<< ss.str();
    }
  } else {
    edm::LogWarning(mlDqmClient_) 
      << "[OptoScanHistograms::" << __func__ << "]"
      << " No histograms to analyze!";
  }
  
}
