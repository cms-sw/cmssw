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
PedestalsHistograms::~PedestalsHistograms() {
  LogTrace(mlDqmClient_) 
    << "[PedestalsHistograms::" << __func__ << "]"
    << " Destructing object...";
}

// -----------------------------------------------------------------------------	 
/** */	 
void PedestalsHistograms::histoAnalysis( bool debug ) {
  
  // Clear map holding analysis objects
  data_.clear();

  // Iterate through map containing vectors of profile histograms
  CollationsMap::const_iterator iter = collations().begin();
  for ( ; iter != collations().end(); iter++ ) {
    
    // Check vector of histos is not empty (should be 4 histos)
    if ( iter->second.empty() ) {
      edm::LogWarning(mlDqmClient_) 
	<< "[PedestalsHistograms::" << __func__ << "]"
	<< " Zero collation histograms found!" << endl;
      continue;
    }
    
    // Retrieve pointers to profile histos for this FED channel 
    vector<TH1*> profs;
    Collations::const_iterator ihis = iter->second.begin(); 
    for ( ; ihis != iter->second.end(); ihis++ ) {
      TProfile* prof = ExtractTObject<TProfile>().extract( mui()->get( ihis->first ) );
      if ( prof ) { profs.push_back(prof); }
    } 
    
    // Perform histo analysis
    PedestalsAnalysis anal( iter->first );
    anal.analysis( profs );
    data_[iter->first] = anal; 
    if ( debug ) {
      static uint16_t cntr = 0;
      stringstream ss;
      anal.print( ss, 0 ); 
      anal.print( ss, 1 ); 
      cout << ss.str() << endl;
      cntr++;
    }
    
  }

  LogTrace(mlDqmClient_) 
    << "[PedestalsHistograms::" << __func__ << "]"
    << " Analyzed histograms for " 
    << collations().size() 
    << " FED channels" << endl;
  
}

// -----------------------------------------------------------------------------
/** */
void PedestalsHistograms::createSummaryHisto( const sistrip::SummaryHisto& histo, 
					      const sistrip::SummaryType& type, 
					      const string& directory,
					      const sistrip::Granularity& gran ) {
  LogTrace(mlDqmClient_) << "[PedestalsHistograms::" << __func__ << "]";
  
  // Check view 
  sistrip::View view = SiStripHistoNamingScheme::view(directory);
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
