#include "DQM/SiStripCommissioningClients/interface/FedCablingHistograms.h"
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
FedCablingHistograms::FedCablingHistograms( MonitorUserInterface* mui ) 
  : CommissioningHistograms( mui, sistrip::FED_CABLING ),
    factory_( new Factory )
{
  LogTrace(mlDqmClient_) 
       << "[FedCablingHistograms::" << __func__ << "]"
       << " Constructing object...";
}

// -----------------------------------------------------------------------------
/** */
FedCablingHistograms::FedCablingHistograms( DaqMonitorBEInterface* bei ) 
  : CommissioningHistograms( bei, sistrip::APV_TIMING ),
    factory_( new Factory )
{
  LogTrace(mlDqmClient_) 
    << "[FedCablingHistograms::" << __func__ << "]"
    << " Constructing object...";
}

// -----------------------------------------------------------------------------
/** */
FedCablingHistograms::~FedCablingHistograms() {
  LogTrace(mlDqmClient_) 
    << "[FedCablingHistograms::" << __func__ << "]"
    << " Destructing object...";
}

// -----------------------------------------------------------------------------	 
/** */	 
void FedCablingHistograms::histoAnalysis( bool debug ) {
  LogTrace(mlDqmClient_)
    << "[FedCablingHistograms::" << __func__ << "]";

  uint16_t valid = 0;
  HistosMap::const_iterator iter = 0;
  Analyses::iterator ianal = 0;
  
  // Clear map holding analysis objects
  for ( ianal = data_.begin(); ianal != data_.end(); ianal++ ) { 
    if ( ianal->second ) { delete ianal->second; }
  } 
  data_.clear();
  
  // Iterate through map containing histograms
  for ( iter = histos().begin(); 
	iter != histos().end(); iter++ ) {
    
    // Check vector of histos is not empty
    if ( iter->second.empty() ) {
      edm::LogWarning(mlDqmClient_)
	<< "[FedCablingHistograms::" << __func__ << "]"
	<< " Zero histograms found!";
      continue;
    }
    
    // Retrieve pointers to profile histos for this FED channel 
    std::vector<TH1*> profs;
    Histos::const_iterator ihis = iter->second.begin(); 
    for ( ; ihis != iter->second.end(); ihis++ ) {
      TProfile* prof = ExtractTObject<TProfile>().extract( (*ihis)->me_ );
      if ( prof ) { profs.push_back(prof); }
    } 
    
    // Perform histo analysis
    FedCablingAnalysis* anal = new FedCablingAnalysis( iter->first );
    anal->analysis( profs );
    data_[iter->first] = anal; 
    if ( anal->isValid() ) { valid++; }
    if ( debug ) {
      std::stringstream ss;
      anal->print( ss ); 
      if ( anal->isValid() ) { LogTrace(mlDqmClient_) << ss.str(); }
      else { edm::LogWarning(mlDqmClient_) << ss.str(); }
    }
    
  }
  
  if ( !histos().empty() ) {
    edm::LogVerbatim(mlDqmClient_) 
      << "[FedCablingHistograms::" << __func__ << "]"
      << " Analyzed histograms for " << histos().size() 
      << " FED channels, of which " << valid 
      << " (" << 100 * valid / histos().size()
      << "%) are valid.";
  } else {
    edm::LogWarning(mlDqmClient_) 
      << "[FedCablingHistograms::" << __func__ << "]"
      << " No histograms to analyze!";
  }
  
}

// -----------------------------------------------------------------------------
/** */
void FedCablingHistograms::createSummaryHisto( const sistrip::Monitorable& histo, 
					       const sistrip::Presentation& type, 
					       const std::string& dir,
					       const sistrip::Granularity& gran ) {
  LogTrace(mlDqmClient_)
    << "[FedCablingHistograms::" << __func__ << "]";
  
  // Check view 
  sistrip::View view = SiStripEnumsAndStrings::view(dir);
  if ( view == sistrip::UNKNOWN_VIEW ) { return; }
  
  // Analyze histograms if not done already
  if ( data_.empty() ) { histoAnalysis( false ); }
  
  // Extract data to be histogrammed
  uint32_t xbins = factory_->init( histo, type, view, dir, gran, data_ );
  
  // Create summary histogram (if it doesn't already exist)
  TH1* summary = histogram( histo, type, view, dir, xbins );

  // Fill histogram with data
  factory_->fill( *summary );
  
}




