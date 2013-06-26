#include "DQM/SiStripCommissioningSummary/interface/SamplingSummaryFactory.h"
#include "CondFormats/SiStripObjects/interface/SamplingAnalysis.h"
#include "DataFormats/SiStripCommon/interface/SiStripEnumsAndStrings.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <iostream>
#include <sstream>

using namespace sistrip;

// -----------------------------------------------------------------------------
//
void SamplingSummaryFactory::extract( Iterator iter ) {

  SamplingAnalysis* anal = dynamic_cast<SamplingAnalysis*>( iter->second );
  if ( !anal ) { return; }

  if ( mon_ == sistrip::FINE_DELAY_POS ) { 
    generator_->fillMap( level_, gran_, iter->first, anal->maximum() ); 
  } else if ( mon_ == sistrip::FINE_DELAY_ERROR ) { 
    generator_->fillMap( level_, gran_, iter->first, anal->error() ); 
  } else {
    edm::LogWarning(mlSummaryPlots_)
        << "[SummaryPlotFactory::" << __func__ << "]"
	<< " Unexpected monitorable: "
	<< SiStripEnumsAndStrings::monitorable( SummaryPlotFactoryBase::mon_ );
    return;
  }
}

//------------------------------------------------------------------------------
//
void SamplingSummaryFactory::format() {

  // Histogram formatting
  if ( mon_ == sistrip::FINE_DELAY_POS ) {
    generator_->axisLabel( "Delay [ns]" );
  } else if ( mon_ == sistrip::FINE_DELAY_ERROR ) { 
    generator_->axisLabel( "Uncertainty [ns]" );
  } else { 
    edm::LogWarning(mlSummaryPlots_)
         << "[SummaryPlotFactory::" << __func__ << "]"
         <<  " Unexpected SummaryHisto value:"
         << SiStripEnumsAndStrings::monitorable( SummaryPlotFactoryBase::mon_ ) ;
  } 
  
}

