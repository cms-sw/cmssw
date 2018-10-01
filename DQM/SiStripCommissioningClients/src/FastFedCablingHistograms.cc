#include "DQM/SiStripCommissioningClients/interface/FastFedCablingHistograms.h"
#include "CondFormats/SiStripObjects/interface/FastFedCablingAnalysis.h"
#include "DataFormats/SiStripCommon/interface/SiStripConstants.h"
#include "DQM/SiStripCommissioningAnalysis/interface/FastFedCablingAlgorithm.h"
#include "DQM/SiStripCommissioningSummary/interface/FastFedCablingSummaryFactory.h"
#include "DQM/SiStripCommon/interface/ExtractTObject.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "TProfile.h"
#include <iostream>
#include <sstream>
#include <iomanip>

using namespace std;
using namespace sistrip;

// -----------------------------------------------------------------------------
/** */
FastFedCablingHistograms::FastFedCablingHistograms( const edm::ParameterSet& pset,
                                                    DQMStore* bei )
  : CommissioningHistograms( pset.getParameter<edm::ParameterSet>("FastFedCablingParameters"),
                             bei,
                             sistrip::FAST_CABLING )
{
  factory_ = unique_ptr<FastFedCablingSummaryFactory>( new FastFedCablingSummaryFactory );
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
    FastFedCablingAlgorithm algo( this->pset(), anal );
    FedToFecMap::const_iterator ifed = mapping().find( iter->first );
    if ( ifed != mapping().end() ) { anal->fecKey( ifed->second ); }
    algo.analysis( profs );
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

// -----------------------------------------------------------------------------
/** */
void FastFedCablingHistograms::printAnalyses() {
  Analyses::iterator ianal = data().begin();
  Analyses::iterator janal = data().end();
  for ( ; ianal != janal; ++ianal ) { 

    FastFedCablingAnalysis* anal = dynamic_cast<FastFedCablingAnalysis*>( ianal->second );
    if ( !anal ) { 
      edm::LogError(mlDqmClient_)
	<< "[FastFedCablingHistograms::" << __func__ << "]"
	<< " NULL pointer to analysis object!";
      continue; 
    }

    std::stringstream ss;
    anal->print( ss ); 
    if ( anal->isValid() &&
	 !(anal->isDirty()) && 
	 !(anal->badTrimDac()) ) { LogTrace(mlDqmClient_) << ss.str(); 
    } else { edm::LogWarning(mlDqmClient_) << ss.str(); }

  }

}

// -----------------------------------------------------------------------------
/** */
void FastFedCablingHistograms::printSummary() {

  std::stringstream good;
  std::stringstream bad;
  
  Analyses::iterator ianal = data().begin();
  Analyses::iterator janal = data().end();
  for ( ; ianal != janal; ++ianal ) { 

    FastFedCablingAnalysis* anal = dynamic_cast<FastFedCablingAnalysis*>( ianal->second );
    if ( !anal ) { 
      edm::LogError(mlDqmClient_)
	<< "[FastFedCablingHistograms::" << __func__ << "]"
	<< " NULL pointer to analysis object!";
      continue; 
    }

    if ( anal->isValid() &&
	 !(anal->isDirty()) && 
	 !(anal->badTrimDac()) ) { 
      anal->summary( good ); 
    } else { anal->summary( bad ); }

  }

  if ( good.str().empty() ) { good << "None found!"; }
  LogTrace(mlDqmClient_) 
    << "[FastFedCablingHistograms::" << __func__ << "]"
    << " Printing summary of good analyses:" << "\n"
    << good.str();
  
  if ( bad.str().empty() ) { return; } //@@ bad << "None found!"; }
  LogTrace(mlDqmClient_) 
    << "[FastFedCablingHistograms::" << __func__ << "]"
    << " Printing summary of bad analyses:" << "\n"
    << bad.str();
  
}
