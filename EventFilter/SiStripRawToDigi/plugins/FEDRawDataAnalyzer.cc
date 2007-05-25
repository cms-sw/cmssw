#include "EventFilter/SiStripRawToDigi/plugins/FEDRawDataAnalyzer.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/SiStripCommon/interface/SiStripConstants.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <iostream>
#include <sstream>
#include <iomanip>
#include <unistd.h>

using namespace std;
using namespace sistrip;

// -----------------------------------------------------------------------------
// 
FEDRawDataAnalyzer::FEDRawDataAnalyzer( const edm::ParameterSet& pset ) 
  : label_( pset.getUntrackedParameter<std::string>("ProductLabel","source") ),
    instance_( pset.getUntrackedParameter<std::string>("ProductInstance","") ),
    sleep_( pset.getUntrackedParameter<int>("pause_us",0) ),
    initial_( pset.getUntrackedParameter<int>("initial_rate",0) ),
    increment_( pset.getUntrackedParameter<int>("rate_increment",0) ),
    period_( pset.getUntrackedParameter<int>("nevents",0) ),
    meas_(-1.),
    last_(0),
    event_(0)
{
  LogTrace(mlRawToDigi_)
    << "[FEDRawDataAnalyzer::" << __func__ << "]"
    << " Constructing object...";
}

// -----------------------------------------------------------------------------
// 
void FEDRawDataAnalyzer::analyze( const edm::Event& event,
				  const edm::EventSetup& setup ) {
  
  event_++;
  
  // Introduce optional delay
  if ( sleep_ > 0 ) { for ( int ii = 0; ii < sleep_; ++ii ) { usleep(1); } }
  
//   int rate = 0;
//   if ( sleep_ >= 0 ) { usleep( sleep_ ); }
//   else {
//     int temp = increment_ * (event_/period_);
//     if ( initial_ + temp < 0 ) { rate = 0; }
//     else { rate = initial_ + temp; }
//     if ( rate > 0 ) { 
//       int n_usleeps = static_cast<int>(500./static_cast<float>(rate));
//       for ( int ii = 0; ii < n_usleeps; ++ii ) { usleep(1); }
//       LogTrace(mlTest_) 
// 	<< "[FEDRawDataAnalyzer::" << __func__ << "]"
// 	<< " Number of usleep's: " << n_usleeps
// 	<< " (Rate[Hz]: " << rate << ")";
//     }
//   }

//   if ( sleep_ >= 0 ) { usleep( sleep_ ); }
//   else {
//     if ( increment_ < 0 ) { return; }
//     int temp = initial_ + increment_ * (event_/period_);
//     for ( int ii = 0; ii < temp; ++ii ) { usleep(1); }
//     LogTrace(mlTest_) 
//       << "[FEDRawDataAnalyzer::" << __func__ << "]"
//       << " Number of usleep's: " << temp
//       << " (Rate[Hz]: " << 1./(temp*0.002) << ")";
//   }
  
  // Retrieve FED raw data
  edm::Handle<FEDRawDataCollection> buffers;
  event.getByLabel( label_, instance_, buffers ); 
  
  // Some init
  std::stringstream ss; 
  ss << std::endl;
  uint16_t cntr = 0;
  
  // Check FEDRawDataCollection
  for ( uint16_t ifed = 0; ifed < sistrip::CMS_FED_ID_MAX; ifed++ ) {
    uint16_t size = buffers->FEDData( static_cast<int>(ifed) ).size();
    if ( size ) { 
      cntr++;
      if ( edm::isDebugEnabled() ) {
	ss << " FedId: " 
	   << std::setw(4) << std::setfill(' ') << ifed 
	   << " NumberOfChars: " 
	   << std::setw(6) << std::setfill(' ') << size
	   << std::endl; 
      }
    }
  }
  
  // Print out debug
  if ( edm::isDebugEnabled() ) {
    LogTrace(mlRawToDigi_)
      << "[FEDRawDataAnalyzer::" << __func__ << "] \n"
      << " FED buffers with non-zero size in FEDRawDataCollection object:"
      << ss.str()
      << " Total number of FED buffers with non-zero size: " << cntr;
  }
  
//   // Calc rate from system time
//   if ( last_ == 0 ) { last_ = clock(); }
//   else if ( !(event_%period_) ) { 
//     meas_ = static_cast<float>( period_ ) / 
//       ( static_cast<float>( clock() - last_ ) / 
// 	static_cast<float>(CLOCKS_PER_SEC) );
//     //if ( rate > 0 ) { temp_.push_back( Temp(rate,meas_,cntr) ); }
//     temp_.push_back( Temp(rate,meas_,cntr) ); 
//     //LogDebug(mlTest_) << " MEAS " << meas_;
//     last_ = clock();
//   }
  
}

// -----------------------------------------------------------------------------
// 
void FEDRawDataAnalyzer::beginJob( edm::EventSetup const& ) {
}

// -----------------------------------------------------------------------------
// 
void FEDRawDataAnalyzer::endJob() {
//   std::stringstream ss;
//   ss << "[FEDRawDataAnalyzer::" << __func__ << "]"
//      << " Results ( ii, rate[Hz], measured[Hz], nFEDs ):"
//      << std::endl;
//   for ( uint16_t ii = 0; ii < temp_.size(); ++ii ) { 
//     ss << ii << ", " 
//       //<< std::setprecision(1) 
//        << temp_[ii].rate_ << ", "
//       //<< std::setprecision(1) 
//        << temp_[ii].meas_ << ", "
//        << temp_[ii].nfeds_ << std::endl;
//   }
//   LogTrace(mlRawToDigi_) << ss.str();
}
