#include "DQM/SiStripCommissioningClients/interface/VpspScanHistograms.h"
#include "DQM/SiStripCommissioningSummary/interface/SummaryGenerator.h"
#include <iostream>
#include <sstream>
#include <iomanip>

using namespace std;

// -----------------------------------------------------------------------------
/** */
VpspScanHistograms::VpspScanHistograms( MonitorUserInterface* mui ) 
  : CommissioningHistograms(mui),
    factory_( new Factory )
{
  cout << "[" << __PRETTY_FUNCTION__ << "]"
       << " Created object for VPSP SCAN histograms" << endl;
}

// -----------------------------------------------------------------------------
/** */
VpspScanHistograms::~VpspScanHistograms() {
  cout << "[" << __PRETTY_FUNCTION__ << "]" << endl;
}

// -----------------------------------------------------------------------------	 
/** */	 
void VpspScanHistograms::histoAnalysis( bool debug ) {
  
  // Iterate through map containing vectors of profile histograms
  CollationsMap::const_iterator iter = collations().begin();
  for ( ; iter != collations().end(); iter++ ) {
    
    // Check vector of histos is not empty (should be 2 histos)
    if ( iter->second.empty() ) {
      cerr << "[" << __PRETTY_FUNCTION__ << "]"
	   << " Zero collation histograms found!" << endl;
      continue;
    }
    
    // Retrieve pointerd to profile histos for this FED channel 
    vector<TProfile*> profs;
    Collations::const_iterator ihis = iter->second.begin(); 
    for ( ; ihis != iter->second.end(); ihis++ ) {
      TProfile* prof = ExtractTObject<TProfile>().extract( mui()->get( *ihis ) );
      if ( !prof ) { 
	cerr << "[" << __PRETTY_FUNCTION__ << "]"
	     << " NULL pointer to MonitorElement!" << endl; 
	continue; 
      }
      profs.push_back(prof);
    } 
    
    // Perform histo analysis
    VpspScanAnalysis anal;
    anal.analysis( profs );
    data_[iter->first] = anal; 
    if ( debug ) {
      static uint16_t cntr = 0;
      stringstream ss;
      anal.print( ss ); 
      cout << ss.str() << endl;
      cntr++;
    }
    
  }
  
  cout << "[" << __PRETTY_FUNCTION__ << "]"
       << " Analyzed histograms for " 
       << collations().size() 
       << " FED channels" << endl;
  
}

// -----------------------------------------------------------------------------
/** */
void VpspScanHistograms::createSummaryHisto( const sistrip::SummaryHisto& histo, 
					     const sistrip::SummaryType& type, 
					     const string& directory,
					     const sistrip::Granularity& gran ) {
  cout << "[" << __PRETTY_FUNCTION__ <<"]" << endl;
  
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
