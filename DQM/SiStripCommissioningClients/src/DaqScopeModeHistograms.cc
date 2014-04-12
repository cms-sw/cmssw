#include "DQM/SiStripCommissioningClients/interface/DaqScopeModeHistograms.h"
#include "CondFormats/SiStripObjects/interface/DaqScopeModeAnalysis.h"
#include "DataFormats/SiStripCommon/interface/SiStripConstants.h"
#include "DataFormats/SiStripCommon/interface/SiStripEnumsAndStrings.h"
#include "DQM/SiStripCommissioningAnalysis/interface/DaqScopeModeAlgorithm.h"
#include "DQM/SiStripCommissioningSummary/interface/SummaryGenerator.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <iostream>
#include <sstream>
#include <iomanip>
 
using namespace std;
using namespace sistrip;

// -----------------------------------------------------------------------------
/** */
DaqScopeModeHistograms::DaqScopeModeHistograms( const edm::ParameterSet& pset,
                                                DQMStore* bei ) 
  : CommissioningHistograms( pset.getParameter<edm::ParameterSet>("DaqScopeModeParameters"),
                             bei,
                             sistrip::DAQ_SCOPE_MODE ),
    factory_( new Factory )
{
  cout << endl // LogTrace(mlDqmClient_) 
       << "[DaqScopeModeHistograms::" << __func__ << "]"
       << " Constructing object...";
}

// -----------------------------------------------------------------------------
/** */
DaqScopeModeHistograms::~DaqScopeModeHistograms() {
  cout << endl // LogTrace(mlDqmClient_) 
       << "[DaqScopeModeHistograms::" << __func__ << "]"
       << " Constructing object...";
}

// -----------------------------------------------------------------------------	 
/** */	 
void DaqScopeModeHistograms::histoAnalysis( bool debug ) {

  // Clear std::map holding analysis objects
  data_.clear();
  
//   // Iterate through std::map containing std::vectors of profile histograms
//   CollationsMap::const_iterator iter = collations().begin();
//   for ( ; iter != collations().end(); iter++ ) {
    
//     // Check std::vector of histos is not empty (should be 1 histo)
//     if ( iter->second.empty() ) {
//       edm::LogWarning(mlDqmClient_) 
// 	<< "[DaqScopeModeHistograms::" << __func__ << "]"
// 	<< " Zero collation histograms found!";
//       continue;
//     }
    
//     // Retrieve pointers to profile histos for this FED channel 
//     std::vector<TH1*> histos;
//     Collations::const_iterator ihis = iter->second.begin(); 
//     for ( ; ihis != iter->second.end(); ihis++ ) {
//       TH1F* his = ExtractTObject<TH1F>().extract( ihis->second->getMonitorElement() );
//       if ( his ) { histos.push_back(his); }
//     } 
    
//     // Perform histo analysis
//     DaqScopeModeAnalysis anal( iter->first );
//     DaqScopeModeAlgorithm algo( &anal );
//     algo.analysis( histos );
//     data_[iter->first] = anal; 
//     if ( debug ) {
//       std::stringstream ss;
//       anal.print( ss ); 
//       cout << ss.str() << endl;
//     }
    
//   }
  
//   cout << endl // LogTrace(mlDqmClient_) 
//        << "[DaqScopeModeHistograms::" << __func__ << "]"
//        << " Analyzed histograms for " 
//        << collations().size() 
//        << " FED channels";
  
}

// -----------------------------------------------------------------------------
/** */
void DaqScopeModeHistograms::createSummaryHisto( const sistrip::Monitorable& histo, 
						 const sistrip::Presentation& type, 
						 const std::string& directory,
						 const sistrip::Granularity& gran ) {
  cout << endl // LogTrace(mlDqmClient_)
       << "[DaqScopeModeHistograms::" << __func__ << "]";
  
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
