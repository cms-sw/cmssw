#include "DQM/SiStripCommissioningClients/interface/ApvTimingHistograms.h"
#include "DQM/SiStripCommissioningSummary/interface/SummaryGenerator.h"
#include <iostream>
#include <sstream>
#include <iomanip>

using namespace std;

// -----------------------------------------------------------------------------
/** */
ApvTimingHistograms::ApvTimingHistograms( MonitorUserInterface* mui ) 
  : CommissioningHistograms(mui),
    factory_( new Factory ),
    optimumSamplingPoint_(15.),
    minDelay_(sistrip::invalid_),
    maxDelay_(-1.*sistrip::invalid_),
    deviceWithMinDelay_(sistrip::invalid_),
    deviceWithMaxDelay_(sistrip::invalid_)
{
  cout << "[" << __PRETTY_FUNCTION__ << "]"
       << " Created object for APV TIMING histograms" << endl;
}

// -----------------------------------------------------------------------------
/** */
ApvTimingHistograms::~ApvTimingHistograms() {
  cout << "[" << __PRETTY_FUNCTION__ << "]" << endl;
}

// -----------------------------------------------------------------------------	 
/** */	 
void ApvTimingHistograms::histoAnalysis() {
  cout << "[" << __PRETTY_FUNCTION__ << "]" << endl;

  // Reset minimum / maximum delays
  minDelay_ =  1. * sistrip::invalid_;
  maxDelay_ = -1. * sistrip::invalid_;

  // Iterate through map containing vectors of profile histograms
  CollationsMap::const_iterator iter = collations().begin();
  for ( ; iter != collations().end(); iter++ ) {
    
    // Check vector of histos is not empty (should be 1 histo)
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
    ApvTimingAnalysis anal;
    anal.analysis( profs );
    data_[iter->first] = anal; 
    //stringstream ss;
    //anal.print( ss ); 
    //cout << ss.str() << endl;

    // Check tick height is valid
    if ( anal.height() < 100. ) { continue; }
    
    // Set maximum delay
    if ( anal.delay() < sistrip::maximum_ && 
	 anal.delay() > maxDelay_ ) { 
      maxDelay_ = anal.delay(); 
      deviceWithMaxDelay_ = iter->first;
    }

    // Set minimum delay
    if ( anal.delay() < sistrip::maximum_ && 
	 anal.delay() < minDelay_ ) { 
      minDelay_ = anal.delay(); 
      deviceWithMinDelay_ = iter->first;
    }
    
  }
  
  cout << "[" << __PRETTY_FUNCTION__ << "]"
       << " Analyzed histograms for " 
       << collations().size() 
       << " FED channels" << endl;

  // Adjust maximum (and minimum) delay(s) to find optimum sampling point(s)
  if ( maxDelay_ > 0. ) { 

    SiStripControlKey::ControlPath max = SiStripControlKey::path( deviceWithMaxDelay_ );
    SiStripControlKey::ControlPath min = SiStripControlKey::path( deviceWithMinDelay_ );
    cout << " Device (FEC/slot/ring/CCU/module/channel) " 
	 << max.fecCrate_ << "/" 
	 << max.fecSlot_ << "/" 
	 << max.fecRing_ << "/" 
	 << max.ccuAddr_ << "/"
	 << max.ccuChan_ << "/"
	 << " has maximum delay (rising edge) [ns]:" << maxDelay_ << endl;
    cout << " Device (FEC/slot/ring/CCU/module/channel): " 
	 << min.fecCrate_ << "/" 
	 << min.fecSlot_ << "/" 
	 << min.fecRing_ << "/" 
	 << min.ccuAddr_ << "/"
	 << min.ccuChan_ << "/"
	 << " has minimum delay (rising edge) [ns]:" << minDelay_ << endl;

    float adjustment = 25 - int( rint(maxDelay_+optimumSamplingPoint_) ) % 25;
    cout << " Adjustment required to find optimum sampling point: " << adjustment << endl;
    maxDelay_ += adjustment;
    minDelay_ += adjustment;

    cout << " Device (FEC/slot/ring/CCU/module/channel) " 
	 << max.fecCrate_ << "/" 
	 << max.fecSlot_ << "/" 
	 << max.fecRing_ << "/" 
	 << max.ccuAddr_ << "/"
	 << max.ccuChan_ << "/"
	 << " has maximum delay (sampling point) [ns]:" << maxDelay_ << endl;
    cout << " Device (FEC/slot/ring/CCU/module/channel): " 
	 << min.fecCrate_ << "/" 
	 << min.fecSlot_ << "/" 
	 << min.fecRing_ << "/" 
	 << min.ccuAddr_ << "/"
	 << min.ccuChan_ << "/"
	 << " has minimum delay (sampling point) [ns]:" << minDelay_ << endl;
    
  }
  
}

// -----------------------------------------------------------------------------
/** */
void ApvTimingHistograms::createSummaryHisto( const sistrip::SummaryHisto& histo, 
					      const sistrip::SummaryType& type, 
					      const string& directory,
					      const sistrip::Granularity& gran ) {
  cout << "[" << __PRETTY_FUNCTION__ <<"]" << endl;
  
  // Check view 
  sistrip::View view = SiStripHistoNamingScheme::view(directory);
  if ( view == sistrip::UNKNOWN_VIEW ) { return; }

  // Analyze histograms
  histoAnalysis();

  // Extract data to be histogrammed
  factory_->init( histo, type, view, directory, gran );
  uint32_t xbins = factory_->extract( data_ );

  // Create summary histogram (if it doesn't already exist)
  TH1* summary = histogram( histo, type, view, directory, xbins );

  // Fill histogram with data
  factory_->fill( *summary );
  
}
