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
  cout << endl // LogTrace(mlDqmClient_) 
       << "[FedCablingHistograms::" << __func__ << "]"
       << " Constructing object...";
}

// -----------------------------------------------------------------------------
/** */
FedCablingHistograms::~FedCablingHistograms() {
  cout << endl // LogTrace(mlDqmClient_) 
       << "[FedCablingHistograms::" << __func__ << "]"
       << " Destructing object...";
}

// -----------------------------------------------------------------------------	 
/** */	 
void FedCablingHistograms::histoAnalysis( bool debug ) {
  cout << endl // LogTrace(mlDqmClient_)
       << "[FedCablingHistograms::" << __func__ << "]";

  // Clear map holding analysis objects
  data_.clear();

  // Iterate through map containing vectors of profile histograms
  uint32_t connected = 0;
  CollationsMap::const_iterator iter = collations().begin();
  for ( ; iter != collations().end(); iter++ ) {
    
    // Check vector of histos is not empty (should be 1 histo)
    if ( iter->second.empty() ) {
      cerr << endl // edm::LogWarning(mlDqmClient_)
	   << "[FedCablingHistograms::" << __func__ << "]"
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
    FedCablingAnalysis anal( iter->first );
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

    // Count connections
    if ( anal.fedId() < sistrip::maximum_ &&
	 anal.fedId() < sistrip::maximum_ ) {
      connected++;
    }

  }
  
  cout << endl // LogTrace(mlDqmClient_)
       << "[FedCablingHistograms::" << __func__ << "]"
       << " Analyzed histograms and found "
       << connected 
       << " connected FED channels out of a possible "
       << collations().size()
       << " (" << (100*connected/collations().size())
       << "%)";
  
}

// -----------------------------------------------------------------------------
/** */
void FedCablingHistograms::createSummaryHisto( const sistrip::SummaryHisto& histo, 
					       const sistrip::SummaryType& type, 
					       const string& directory,
					       const sistrip::Granularity& gran ) {
  cout << endl // LogTrace(mlDqmClient_)
       << "[FedCablingHistograms::" << __func__ << "]";
  
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




