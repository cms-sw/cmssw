#include "DQM/SiStripCommissioningClients/interface/DaqScopeModeHistograms.h"
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
DaqScopeModeHistograms::DaqScopeModeHistograms( MonitorUserInterface* mui ) 
  : CommissioningHistograms( mui, sistrip::DAQ_SCOPE_MODE ),
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

  // Clear map holding analysis objects
  data_.clear();
  
  // Iterate through map containing vectors of profile histograms
  CollationsMap::const_iterator iter = collations().begin();
  for ( ; iter != collations().end(); iter++ ) {
    
    // Check vector of histos is not empty (should be 1 histo)
    if ( iter->second.empty() ) {
      edm::LogWarning(mlDqmClient_) 
	<< "[DaqScopeModeHistograms::" << __func__ << "]"
	<< " Zero collation histograms found!";
      continue;
    }
    
    // Retrieve pointers to profile histos for this FED channel 
    vector<TH1*> histos;
    Collations::const_iterator ihis = iter->second.begin(); 
    for ( ; ihis != iter->second.end(); ihis++ ) {
      TH1F* his = ExtractTObject<TH1F>().extract( mui()->get( ihis->first ) );
      if ( his ) { histos.push_back(his); }
    } 
    
    // Perform histo analysis
    DaqScopeModeAnalysis anal( iter->first );
    anal.analysis( histos );
    data_[iter->first] = anal; 
    if ( debug ) {
      stringstream ss;
      anal.print( ss ); 
      cout << ss.str() << endl;
    }
    
  }
  
  cout << endl // LogTrace(mlDqmClient_) 
       << "[DaqScopeModeHistograms::" << __func__ << "]"
       << " Analyzed histograms for " 
       << collations().size() 
       << " FED channels";
  
}

// -----------------------------------------------------------------------------
/** */
void DaqScopeModeHistograms::createSummaryHisto( const sistrip::SummaryHisto& histo, 
						 const sistrip::SummaryType& type, 
						 const string& directory,
						 const sistrip::Granularity& gran ) {
  cout << endl // LogTrace(mlDqmClient_)
       << "[DaqScopeModeHistograms::" << __func__ << "]";
  
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
