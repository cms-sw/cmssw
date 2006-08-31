#include "DQM/SiStripCommissioningClients/interface/ApvTimingHistograms.h"
#include "DQM/SiStripCommon/interface/SummaryGenerator.h"
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
  uint32_t cntr = 0;
  uint32_t nchans = collations().size();
  CollationsMap::const_iterator iter = collations().begin();
  for ( ; iter != collations().end(); iter++ ) {
    
    if ( iter->second.empty() ) {
      cerr << "[" << __PRETTY_FUNCTION__ << "]"
	   << " Zero collation histograms found!" << endl;
      continue;
    }
    cntr++;
    cout << "[" << __PRETTY_FUNCTION__ << "]"
	 << " Analyzing histograms from " << cntr
	 << " of " << nchans << " FED channels..." << endl;
    
    // Retrieve pointer to profile histo 
    TProfile* prof = ExtractTObject<TProfile>().extract( mui()->get(iter->second[0]) );
    if ( !prof ) { 
      cerr << "[" << __PRETTY_FUNCTION__ << "]"
	   << " NULL pointer to MonitorElement!" << endl; 
      continue; 
    }
    
    // Retrieve control key
    static SiStripHistoNamingScheme::HistoTitle title;
    title = SiStripHistoNamingScheme::histoTitle( prof->GetName() );
    
    // Some checks
    if ( title.task_ != sistrip::APV_TIMING ) {
      cerr << "[" << __PRETTY_FUNCTION__ << "]"
	   << " Unexpected commissioning task!"
	   << "(" << SiStripHistoNamingScheme::task( title.task_ ) << ")"
	   << endl;
    }
    
    // Perform histo analysis
    ApvTimingAnalysis::Monitorables mons;
    ApvTimingAnalysis::analysis( prof, mons );
    
    // Store delay in map
    data_[iter->first] = mons; 

    // Check tick height is valid
    if ( mons.height_ < 100. ) { continue; }
    
    // Set maximum delay
    if ( mons.delay_ < sistrip::maximum_ && 
	 mons.delay_ > maxDelay_ ) { 
      maxDelay_ = mons.delay_; 
      deviceWithMaxDelay_ = iter->first;
    }

    // Set minimum delay
    if ( mons.delay_ < sistrip::maximum_ && 
	 mons.delay_ < minDelay_ ) { 
      minDelay_ = mons.delay_; 
      deviceWithMinDelay_ = iter->first;
    }
    
  }
  
  // Adjust maximum (and minimum) delay(s) to find optimum sampling point(s)
  if ( maxDelay_ > 0. ) { 
    SiStripControlKey::ControlPath max = SiStripControlKey::path( deviceWithMaxDelay_ );
    SiStripControlKey::ControlPath min = SiStripControlKey::path( deviceWithMinDelay_ );
    cout << " Device (FEC/slot/ring/CCU/module/channel) " << deviceWithMaxDelay_
      //<< max.fecCrate_ << "/" << max.fecSlot_ << "/" max.fecRing_ << "/" << max.
	 << " has maximum delay (rising edge) [ns]:" << maxDelay_ << endl;
    cout << " Device (FEC/slot/ring/CCU/module/channel): " << deviceWithMinDelay_
	 << " has minimum delay (rising edge) [ns]:" << minDelay_ << endl;
    float adjustment = 25 - int( rint(maxDelay_+optimumSamplingPoint_) ) % 25;
    cout << " Adjustment required to find optimum sampling point: " << adjustment << endl;
    maxDelay_ += adjustment;
    minDelay_ += adjustment;
    cout << " Device (FEC/slot/ring/CCU/module/channel): " << deviceWithMaxDelay_
	 << " has maximum delay (sampling point) [ns]:" << maxDelay_ << endl;
    cout << " Device (FEC/slot/ring/CCU/module/channel): " << deviceWithMinDelay_
	 << " has minimum delay (sampling point) [ns]:" << minDelay_ << endl;
  }
  
}

// -----------------------------------------------------------------------------
/** */
void ApvTimingHistograms::createSummaryHisto( const sistrip::SummaryHisto& histo, 
					      const sistrip::SummaryType& type, 
					      const string& directory ) {
  cout << "[" << __PRETTY_FUNCTION__ <<"]" << endl;
  
  // Check view 
  sistrip::View view = SiStripHistoNamingScheme::view(directory);
  if ( view == sistrip::UNKNOWN_VIEW ) { return; }

  // Change to appropriate directory
  mui()->setCurrentFolder( directory );
  
  // Create MonitorElement (if it doesn't already exist) and update contents
  string name = SummaryGenerator::name( histo, type, view, directory );
  MonitorElement* me = mui()->get( mui()->pwd() + "/" + name );
  if ( !me ) { me = mui()->getBEInterface()->book1D( name, "", 0, 0., 0. ); }
  TH1F* summary = ExtractTObject<TH1F>().extract( me ); 
  factory_->generate( histo, 
		      type, 
		      view, 
		      directory, 
		      data_,
		      *summary );
  
}

