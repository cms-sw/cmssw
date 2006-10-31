#include "DQM/SiStripCommissioningClients/interface/OptoScanHistograms.h"
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
OptoScanHistograms::OptoScanHistograms( MonitorUserInterface* mui ) 
  : CommissioningHistograms( mui, sistrip::OPTO_SCAN ),
    factory_( new Factory )
{
  cout << endl // LogTrace(mlDqmClient_) 
       << "[OptoScanHistograms::" << __func__ << "]"
       << " Constructing object...";
}

// -----------------------------------------------------------------------------
/** */
OptoScanHistograms::~OptoScanHistograms() {
  cout << endl // LogTrace(mlDqmClient_) 
       << "[OptoScanHistograms::" << __func__ << "]"
       << " Denstructing object...";
}

// -----------------------------------------------------------------------------	 
/** */	 
void OptoScanHistograms::histoAnalysis( bool debug ) {
  
  // Clear map holding analysis objects
  data_.clear();

  // Iterate through map containing vectors of profile histograms
  CollationsMap::const_iterator iter = collations().begin();
  for ( ; iter != collations().end(); iter++ ) {
    
    // Check vector of histos is not empty (should be 2 histos)
    if ( iter->second.empty() ) {
      cerr << endl // edm::LogWarning(mlDqmClient_) 
	   << "[OptoScanHistograms::" << __func__ << "]"
	   << " Zero collation histograms found!";
      continue;
    }
    
    // Retrieve pointers to profile histos for this FED channel 
    vector<TProfile*> profs;
    Collations::const_iterator ihis = iter->second.begin(); 
    for ( ; ihis != iter->second.end(); ihis++ ) {
      TProfile* prof = ExtractTObject<TProfile>().extract( mui()->get( *ihis ) );
      if ( prof ) { profs.push_back(prof); }
    } 
    
    // Perform histo analysis
    OptoScanAnalysis anal( iter->first );
    anal.analysis( profs );
    data_[iter->first] = anal; 
    if ( debug ) {
      static uint16_t cntr = 0;
      stringstream ss;
      anal.print( ss ); 
      cout << endl // LogTrace(mlDqmClient_) 
	   << ss.str();
      cntr++;
    }

  }
  
  cout << endl // LogTrace(mlDqmClient_) 
       << "[OptoScanHistograms::" << __func__ << "]"
       << " Analyzed histograms for " 
       << collations().size() 
       << " FED channels";
  
}

// -----------------------------------------------------------------------------
/** */
void OptoScanHistograms::createSummaryHisto( const sistrip::SummaryHisto& histo, 
					     const sistrip::SummaryType& type, 
					     const string& directory,
					     const sistrip::Granularity& gran ) {
  cout << endl // LogTrace(mlDqmClient_)
       << "[OptoScanHistograms::" << __func__ << "]";
  
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
