#include "DQM/SiStripCommissioningClients/interface/PedestalsHistograms.h"
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
PedestalsHistograms::PedestalsHistograms( MonitorUserInterface* mui ) 
  : CommissioningHistograms( mui, sistrip::PEDESTALS ),
    factory_( new Factory )
{
  LogTrace(mlDqmClient_) 
    << "[PedestalsHistograms::" << __func__ << "]"
    << " Constructing object...";
}

// -----------------------------------------------------------------------------
/** */
PedestalsHistograms::PedestalsHistograms( DaqMonitorBEInterface* bei ) 
  : CommissioningHistograms( bei, sistrip::PEDESTALS ),
    factory_( new Factory )
{
  LogTrace(mlDqmClient_) 
    << "[PedestalsHistograms::" << __func__ << "]"
    << " Constructing object...";
}

// -----------------------------------------------------------------------------
/** */
PedestalsHistograms::~PedestalsHistograms() {
  LogTrace(mlDqmClient_) 
    << "[PedestalsHistograms::" << __func__ << "]"
    << " Destructing object...";
}

// -----------------------------------------------------------------------------	 
/** */	 
void PedestalsHistograms::histoAnalysis( bool debug ) {

  uint16_t valid = 0;
  
  // Clear map holding analysis objects
  data_.clear();
  
  // Iterate through map containing histograms
  HistosMap::const_iterator iter = histos().begin();
  for ( ; iter != histos().end(); iter++ ) {
    
    // Check vector of histos is not empty
    if ( iter->second.empty() ) {
      edm::LogWarning(mlDqmClient_) 
	<< "[PedestalsHistograms::" << __func__ << "]"
	<< " Zero histograms found!";
      continue;
    }
    
    // Retrieve pointers to profile histos
    std::vector<TH1*> profs;
    Histos::const_iterator ihis = iter->second.begin(); 
    for ( ; ihis != iter->second.end(); ihis++ ) {
      TProfile* prof = ExtractTObject<TProfile>().extract( (*ihis)->me_ );
      if ( prof ) { profs.push_back(prof); }
    }
    
    // Perform histo analysis
    PedestalsAnalysis anal( iter->first );
    anal.analysis( profs );
    data_[iter->first] = anal; 
    if ( anal.isValid() ) { valid++; }
    if ( debug ) {
      std::stringstream ss;
      anal.print( ss, 1 ); 
      anal.print( ss, 2 ); 
      if ( anal.isValid() ) { LogTrace(mlDqmClient_) << ss.str(); }
      else { edm::LogWarning(mlDqmClient_) << ss.str(); }
    }
    
  }
  
  if ( !histos().empty() ) {
    edm::LogVerbatim(mlDqmClient_) 
      << "[PedestalsHistograms::" << __func__ << "]"
      << " Analyzed histograms for " << histos().size() 
      << " FED channels, of which " << valid 
      << " (" << 100 * valid / histos().size()
      << "%) are valid.";
  } else {
    edm::LogWarning(mlDqmClient_) 
      << "[PedestalsHistograms::" << __func__ << "]"
      << " No histograms to analyze!";
  }
  
}

// -----------------------------------------------------------------------------
/** */
void PedestalsHistograms::createSummaryHisto( const sistrip::Monitorable& histo, 
					      const sistrip::Presentation& type, 
					      const std::string& directory,
					      const sistrip::Granularity& gran ) {
  LogTrace(mlDqmClient_)
    << "[PedestalsHistograms::" << __func__ << "]";
  
  // Check view 
  sistrip::View view = SiStripEnumsAndStrings::view(directory);
  if ( view == sistrip::UNKNOWN_VIEW ) { return; }

  // Analyze histograms
  histoAnalysis( false );

  // Extract data to be histogrammed
  factory_->init( histo, type, view, directory, gran );
  uint32_t xbins = factory_->extract( data_ );

  // Create summary histogram (if it doesn't already exist)
  TH1* summary = histogram( histo, type, view, directory, xbins );

  // Fill histogram with data
  factory_->fill( *summary );
  
}
