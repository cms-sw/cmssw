#include "DQM/SiStripCommissioningSummary/interface/SummaryPlotFactoryBase.h"
#include "DQM/SiStripCommissioningSummary/interface/SummaryGenerator.h"
#include "DataFormats/SiStripCommon/interface/SiStripHistoNamingScheme.h"
#include <iostream>

using namespace std;

// -----------------------------------------------------------------------------
//
SummaryPlotFactoryBase::SummaryPlotFactoryBase() :
  mon_(sistrip::UNKNOWN_MONITORABLE),
  pres_(sistrip::UNKNOWN_PRESENTATION),
  view_(sistrip::UNKNOWN_VIEW),
  level_(sistrip::root_),
  gran_(sistrip::UNKNOWN_GRAN),
  generator_(0) 
{;} 

// -----------------------------------------------------------------------------
//
SummaryPlotFactoryBase::~SummaryPlotFactoryBase() {
  if ( generator_ ) { delete generator_; }
}

// -----------------------------------------------------------------------------
//
void SummaryPlotFactoryBase::init( const sistrip::Monitorable& mon, 
				   const sistrip::Presentation& pres,
				   const sistrip::View& view, 
				   const string& level, 
				   const sistrip::Granularity& gran ) {
  
  // Set parameters
  mon_ = mon;
  pres_ = pres;
  view_ = view;
  level_ = level;
  gran_ = gran;
  
  // Retrieve utility class used to generate summary histograms
  if ( generator_ ) { 
    delete generator_;
    generator_ = 0;
  }
  generator_ = SummaryGenerator::instance( view );
  
  // Check if instance of generator class exists
  if ( !generator_ ) { 
    cerr << "[SummaryPlotFactoryBase::" << __func__ << "]" 
	 << " NULL pointer to SummaryGenerator object!" << endl;
    return;
  }
  
  // Clear map 
  generator_->clearMap();
  
}

// -----------------------------------------------------------------------------
//
void SummaryPlotFactoryBase::fill( TH1& summary_histo ) {
  
  // Check if instance of generator class exists
  if ( !generator_ ) { 
    cerr << "[SummaryPlotFactoryBase::" << __func__ << "]" 
	 << " NULL pointer to SummaryGenerator object!" << endl;
    return;
  }
  
  // Check if map is filled
  if ( !generator_->nBins() ) { 
    cerr << "[SummaryPlotFactoryBase::" << __func__ << "]" 
	 << " SummaryGenerator::map_ is empty!" << endl;
    return; 
  } 
  
  // Check if instance of generator class exists
  if ( !(&summary_histo) ) { 
    cerr << "[SummaryPlotFactoryBase::" << __func__ << "]" 
	 << " NULL pointer to TH1 object!" << endl;
    return;
  }

  // Generate appropriate summary histogram 
  if ( pres_ == sistrip::SUMMARY_HISTO ) {
    generator_->summaryHisto( summary_histo );
  } else if ( pres_ == sistrip::SUMMARY_1D ) {
    generator_->summary1D( summary_histo );
  } else if ( pres_ == sistrip::SUMMARY_2D ) {
    generator_->summary2D( summary_histo );
  } else if ( pres_ == sistrip::SUMMARY_PROF ) {
    generator_->summaryProf( summary_histo );
  } else { 
    cerr << "[SummaryPlotFactoryBase::" << __func__ << "]" 
	 << " Unexpected presentation type: "
	 << SiStripHistoNamingScheme::presentation( pres_ )
	 << endl;
    return; 
  }
  
  // Histogram formatting
  generator_->format( sistrip::UNKNOWN_TASK, 
		      mon_, 
		      pres_, 
		      view_, 
		      level_, 
		      gran_, 
		      summary_histo );
  
}
