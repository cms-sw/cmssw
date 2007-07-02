#include "DQM/SiStripCommissioningClients/interface/ApvTimingHistograms.h"
#include "DQM/SiStripCommissioningSummary/interface/SummaryGenerator.h"
#include "DataFormats/SiStripCommon/interface/SiStripConstants.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <iostream>
#include <sstream>
#include <iomanip>
 
using namespace std;
using namespace sistrip;

// -----------------------------------------------------------------------------
/** */
ApvTimingHistograms::ApvTimingHistograms( MonitorUserInterface* mui ) 
  : CommissioningHistograms( mui, sistrip::APV_TIMING ),
    factory_( new Factory )
{
  LogTrace(mlDqmClient_) 
    << "[ApvTimingHistograms::" << __func__ << "]"
    << " Constructing object...";
}

// -----------------------------------------------------------------------------
/** */
ApvTimingHistograms::ApvTimingHistograms( DaqMonitorBEInterface* bei ) 
  : CommissioningHistograms( bei, sistrip::APV_TIMING ),
    factory_( new Factory )
{
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
  HistosMap::const_iterator iter = 0;
  Analyses::iterator ianal = 0;
  std::map<std::string,uint16_t> errors;
  
  // Clear map holding analysis objects
  for ( ianal = data_.begin(); ianal != data_.end(); ianal++ ) { 
    if ( ianal->second ) { delete ianal->second; }
  } 
  data_.clear();
  
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
    anal->analysis( profs );
    data_[iter->first] = anal; 
    
    // Check tick height is valid
    if ( anal->height() < ApvTimingAnalysis::tickMarkHeightThreshold_ ) { 
      anal->addErrorCode(sistrip::tickMarkBelowThresh_);      
      continue; 
    }

    // Check time of rising edge
    if ( anal->time() > sistrip::valid_ ) { continue; }
    
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
      << "[SiStripCommissioningOffline::" << __func__ << "]"
      << " Unable to set maximum time! Found unexpected value: "
      << time_max;
    
  } else {
    
    SiStripFecKey min( device_min );
    edm::LogVerbatim(mlDqmClient_)
      << "[SiStripCommissioningOffline::" << __func__ << "]"
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
      << "[SiStripCommissioningOffline::" << __func__ << "]"
      << " Crate/FEC/Ring/CCU/module/channel: " 
      << max.fecCrate() << "/" 
      << max.fecSlot() << "/" 
      << max.fecRing() << "/" 
      << max.ccuAddr() << "/"
      << max.ccuChan() << "/"
      << max.lldChan() 
      << " has maximum time for tick mark rising edge [ns]: " << time_max;

    edm::LogVerbatim(mlDqmClient_)
      << "[SiStripCommissioningOffline::" << __func__ << "]"
      << " Difference b/w minimum and maximum times"
      << " for tick mark rising edges [ns] is: " << ( time_max - time_min );

  }
  
  // Set reference time for all analysis objects
  for ( ianal = data_.begin(); ianal != data_.end(); ianal++ ) { 
    ianal->second->refTime( time_max ); 
    if ( ianal->second->isValid() ) { valid++; }
    if ( debug ) {
      std::stringstream ss;
      ianal->second->print( ss ); 
      if ( ianal->second->isValid() ) { LogTrace(mlDqmClient_) << ss.str(); }
      else { edm::LogWarning(mlDqmClient_) << ss.str(); }
      if ( !ianal->second->getErrorCodes().empty() ) { 
	errors[ianal->second->getErrorCodes()[0]]++;
      }
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
      edm::LogVerbatim(mlDqmClient_) 
	<< "[FastFedCablingHistograms::" << __func__ << "]"
	<< " Found " << count << " errors ("
	<< 100 * count / histos().size() << "%): " 
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
void ApvTimingHistograms::createSummaryHisto( const sistrip::Monitorable& mon, 
					      const sistrip::Presentation& pres, 
					      const std::string& dir,
					      const sistrip::Granularity& gran ) {
  LogTrace(mlDqmClient_)
    << "[ApvTimingHistograms::" << __func__ << "]";
  
  // Check view 
  sistrip::View view = SiStripEnumsAndStrings::view(dir);
  if ( view == sistrip::UNKNOWN_VIEW ) { return; }
  
  // Analyze histograms
  if ( data_.empty() ) { histoAnalysis( false ); }
  
  // Extract data to be histogrammed
  uint32_t xbins = factory_->init( mon, pres, view, dir, gran, data_ );
  
  // Create summary histogram (if it doesn't already exist)
  TH1* summary = 0;
  if ( pres != sistrip::HISTO_1D ) { summary = histogram( mon, pres, view, dir, xbins ); }
  else { summary = histogram( mon, pres, view, dir, sistrip::FED_ADC_RANGE, 0., sistrip::FED_ADC_RANGE*1. ); }
  
  // Fill histogram with data
  factory_->fill( *summary );
  
}
