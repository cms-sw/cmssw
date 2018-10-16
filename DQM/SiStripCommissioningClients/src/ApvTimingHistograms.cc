#include "DQM/SiStripCommissioningClients/interface/ApvTimingHistograms.h"
#include "CondFormats/SiStripObjects/interface/ApvTimingAnalysis.h"
#include "DataFormats/SiStripCommon/interface/SiStripConstants.h"
#include "DQM/SiStripCommissioningAnalysis/interface/ApvTimingAlgorithm.h"
#include "DQM/SiStripCommissioningSummary/interface/ApvTimingSummaryFactory.h"
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
ApvTimingHistograms::ApvTimingHistograms( const edm::ParameterSet& pset,
                                          DQMStore* bei ) 
  : CommissioningHistograms( pset.getParameter<edm::ParameterSet>("ApvTimingParameters"),
                             bei,
                             sistrip::APV_TIMING )
{
  factory_ = unique_ptr<ApvTimingSummaryFactory>( new ApvTimingSummaryFactory );
  LogTrace(mlDqmClient_) 
    << "[ApvTimingHistograms::" << __func__ << "]"
    << " Constructing object...";
}

// -----------------------------------------------------------------------------
/** */
ApvTimingHistograms::~ApvTimingHistograms() {
  LogTrace(mlDqmClient_) 
    << "[ApvTimingHistograms::" << __func__ << "]"
    << " Destructing object...";
}

// -----------------------------------------------------------------------------	 
/** */	 
void ApvTimingHistograms::histoAnalysis( bool debug ) {
  LogTrace(mlDqmClient_)
    << "[ApvTimingHistograms::" << __func__ << "]";

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
  
  // Reset minimum / maximum delays
  float time_min =  1. * sistrip::invalid_;
  float time_max = -1. * sistrip::invalid_;
  uint32_t device_min = sistrip::invalid_;
  uint32_t device_max = sistrip::invalid_;
  
  // Iterate through map containing histograms
  for ( iter = histos().begin();
	iter != histos().end(); iter++ ) {
    
    // Check vector of histos is not empty
    if ( iter->second.empty() ) {
      edm::LogWarning(mlDqmClient_) 
	<< "[ApvTimingHistograms::" << __func__ << "]"
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
    ApvTimingAnalysis* anal = new ApvTimingAnalysis( iter->first );
    ApvTimingAlgorithm algo( this->pset(), anal );
    algo.analysis( profs );
    data()[iter->first] = anal; 

    // Check if tick mark found
    if ( !anal->foundTickMark() ) { continue; }
    
    // Find maximum time
    if ( anal->time() > time_max ) { 
      time_max = anal->time(); 
      device_max = iter->first;
    }
    
    // Find minimum time
    if ( anal->time() < time_min ) { 
      time_min = anal->time(); 
      device_min = iter->first;
    }
    
  }
  
  // Adjust maximum (and minimum) delay(s) to find optimum sampling point(s)
  if ( time_max > sistrip::valid_ ||
       time_max < -1.*sistrip::valid_ ) { 

    edm::LogWarning(mlDqmClient_)
      << "[ApvTimingHistograms::" << __func__ << "]"
      << " Unable to set maximum time! Found unexpected value: "
      << time_max;
    
  } else {
    
    SiStripFecKey min( device_min );
    edm::LogVerbatim(mlDqmClient_)
      << "[ApvTimingHistograms::" << __func__ << "]"
      << " Crate/FEC/Ring/CCU/module/channel: " 
      << min.fecCrate() << "/" 
      << min.fecSlot() << "/" 
      << min.fecRing() << "/" 
      << min.ccuAddr() << "/"
      << min.ccuChan() << "/"
      << min.lldChan() 
      << " has minimum time for tick mark rising edge [ns]: " << time_min;
    
    SiStripFecKey max( device_max );
    edm::LogVerbatim(mlDqmClient_)
      << "[ApvTimingHistograms::" << __func__ << "]"
      << " Crate/FEC/Ring/CCU/module/channel: " 
      << max.fecCrate() << "/" 
      << max.fecSlot() << "/" 
      << max.fecRing() << "/" 
      << max.ccuAddr() << "/"
      << max.ccuChan() << "/"
      << max.lldChan() 
      << " has maximum time for tick mark rising edge [ns]: " << time_max;

    edm::LogVerbatim(mlDqmClient_)
      << "[ApvTimingHistograms::" << __func__ << "]"
      << " Difference b/w minimum and maximum times"
      << " for tick mark rising edges [ns] is: " << ( time_max - time_min );

  }
  
  // Set reference time for all analysis objects
  for ( ianal = data().begin(); ianal != data().end(); ianal++ ) { 
    ApvTimingAnalysis* anal = dynamic_cast<ApvTimingAnalysis*>(ianal->second);
    if ( !anal ) { continue; }
    anal->refTime( time_max, this->pset().getParameter<int>("TargetDelay") );
    if ( anal->isValid() ) { valid++; }
    if ( !anal->getErrorCodes().empty() ) { 
      errors[anal->getErrorCodes()[0]]++;
    }
  }

  if ( !histos().empty() ) {
    edm::LogVerbatim(mlDqmClient_) 
      << "[ApvTimingHistograms::" << __func__ << "]"
      << " Analyzed histograms for " << histos().size() 
      << " FED channels, of which " << valid 
      << " (" << 100 * valid / histos().size()
      << "%) are valid.";
  } else {
    edm::LogWarning(mlDqmClient_) 
      << "[ApvTimingHistograms::" << __func__ << "]"
      << " No histograms to analyze!";
  }

  if ( !histos().empty() ) {
    edm::LogVerbatim(mlDqmClient_) 
      << "[ApvTimingHistograms::" << __func__ << "]"
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
	<< "[ApvTimingHistograms::" << __func__ << "]"
	<< " Found " << count << " errors ("
	<< 100 * count / histos().size() << "%): " 
	<< ss.str();
    }
  } else {
    edm::LogWarning(mlDqmClient_) 
      << "[ApvTimingHistograms::" << __func__ << "]"
      << " No histograms to analyze!";
  }
  
}
