#include "DQM/SiStripCommissioningClients/interface/CalibrationHistograms.h"
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
CalibrationHistograms::CalibrationHistograms( MonitorUserInterface* mui,const sistrip::RunType& task ) 
  : CommissioningHistograms( mui, task ),
    factory_( new Factory )
{
  cout << endl // LogTrace(mlDqmClient_) 
       << "[CalibrationHistograms::" << __func__ << "]"
       << " Constructing object...";
}

// -----------------------------------------------------------------------------
/** */
CalibrationHistograms::~CalibrationHistograms() {
  cout << endl // LogTrace(mlDqmClient_) 
       << "[CalibrationHistograms::" << __func__ << "]"
       << " Deleting object...";
}

// -----------------------------------------------------------------------------	 
/** */	 
void CalibrationHistograms::histoAnalysis( bool debug ) {

  // Clear map holding analysis objects
  data_.clear();
  
  // Iterate through map containing vectors of profile histograms
  HistosMap::const_iterator iter = histos().begin();
  for ( ; iter != histos().end(); iter++ ) {
    // Check vector of histos is not empty (should be 1 histo)
    if ( iter->second.empty() ) {
      edm::LogWarning(mlDqmClient_)
 	   << "[CalibrationHistograms::" << __func__ << "]"
 	   << " Zero collation histograms found!";
      continue;
    }
    
    // Retrieve pointers to 1D histos for this FED channel 
    vector<TH1*> profs;
    Histos::const_iterator ihis = iter->second.begin();
    for ( ; ihis != iter->second.end(); ihis++ ) {
      TH1D* prof = ExtractTObject<TH1D>().extract( (*ihis)->me_ );
      if ( prof ) { profs.push_back(prof); }
    } 

    // Perform histo analysis (in peak mode)
    CalibrationAnalysis anal( iter->first, false );
    anal.analysis( profs );
    data_[iter->first] = anal; 
    
 }
 
}

// -----------------------------------------------------------------------------
/** */
void CalibrationHistograms::createSummaryHisto( const sistrip::Monitorable& mon, 
					      const sistrip::Presentation& pres, 
					      const string& directory,
					      const sistrip::Granularity& gran ) {

  cout << endl // LogTrace(mlDqmClient_)
       << "[CalibrationHistograms::" << __func__ << "]";
  
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
