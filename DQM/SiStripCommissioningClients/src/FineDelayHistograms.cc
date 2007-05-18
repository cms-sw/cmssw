#include "DQM/SiStripCommissioningClients/interface/FineDelayHistograms.h"
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
FineDelayHistograms::FineDelayHistograms( MonitorUserInterface* mui ) 
  : CommissioningHistograms( mui, sistrip::FINE_DELAY ),
    factory_( new Factory )
{
  cout << endl // LogTrace(mlDqmClient_) 
       << "[FineDelayHistograms::" << __func__ << "]"
       << " Constructing object...";
}

// -----------------------------------------------------------------------------
/** */
FineDelayHistograms::~FineDelayHistograms() {
  cout << endl // LogTrace(mlDqmClient_) 
       << "[FineDelayHistograms::" << __func__ << "]"
       << " Deleting object...";
}

// -----------------------------------------------------------------------------	 
/** */	 
void FineDelayHistograms::histoAnalysis( bool debug ) {

  // Clear map holding analysis objects
  data_.clear();
  
//   // Iterate through map containing vectors of profile histograms
//   CollationsMap::const_iterator iter = collations().begin();
//   for ( ; iter != collations().end(); iter++ ) {
    
//     // Check vector of histos is not empty (should be 1 histo)
//     if ( iter->second.empty() ) {
//       cerr << endl // edm::LogWarning(mlDqmClient_) 
// 	   << "[FineDelayHistograms::" << __func__ << "]"
// 	   << " Zero collation histograms found!";
//       continue;
//     }
    
//     // Retrieve pointers to profile histos for this FED channel 
//     vector<TH1*> profs;
//     Collations::const_iterator ihis = iter->second.begin(); 
//     for ( ; ihis != iter->second.end(); ihis++ ) {
//       TProfile* prof = ExtractTObject<TProfile>().extract( ihis->second->getMonitorElement() );
//       if ( prof ) { profs.push_back(prof); }
//     } 

//     // Perform histo analysis
//     FineDelayAnalysis anal( iter->first );
//     anal.analysis( profs );
//     data_[iter->first] = anal; 
    
//   }
  
}

// -----------------------------------------------------------------------------
/** */
void FineDelayHistograms::createSummaryHisto( const sistrip::Monitorable& mon, 
					      const sistrip::Presentation& pres, 
					      const string& directory,
					      const sistrip::Granularity& gran ) {

  cout << endl // LogTrace(mlDqmClient_)
       << "[FineDelayHistograms::" << __func__ << "]";
  
  // Check view 
  sistrip::View view = SiStripEnumsAndStrings::view(directory);
  if ( view == sistrip::UNKNOWN_VIEW ) { return; }

  // Analyze histograms
  histoAnalysis( false );

  // Extract data to be histogrammed
  factory_->init( mon, pres, view, directory, gran );
  uint32_t xbins = factory_->extract( data_ );

  // Create summary histogram (if it doesn't already exist)
  TH1* summary = histogram( mon, pres, view, directory, xbins );

  // Fill histogram with data
  factory_->fill( *summary );
  
}
