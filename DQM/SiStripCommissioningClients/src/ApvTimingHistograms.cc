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
    factory_( new Factory )
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
void ApvTimingHistograms::histoAnalysis( bool debug ) {
  cout << "[" << __PRETTY_FUNCTION__ << "]" << endl;

  data_.clear();
  
  // Reset minimum / maximum delays
  float time_min =  1. * sistrip::invalid_;
  float time_max = -1. * sistrip::invalid_;
  uint32_t device_min = sistrip::invalid_;
  uint32_t device_max = sistrip::invalid_;
  
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
    
    // Check tick height is valid
    if ( anal.height() < 100. ) { 
      cerr << "[" << __PRETTY_FUNCTION__ << "]"
	   << " Tick mark height too small: " << anal.height() << endl;
      continue; 
    }
    
    // Find maximum time
    if ( anal.time() < sistrip::maximum_ && 
	 anal.time() > time_max ) { 
      time_max = anal.time(); 
      device_max = iter->first;
    }

    // Find minimum time
    if ( anal.time() < sistrip::maximum_ && 
	 anal.time() < time_min ) { 
      time_min = anal.time(); 
      device_min = iter->first;
    }
    
  }
  
  cout << "[" << __PRETTY_FUNCTION__ << "]"
       << " Analyzed histograms for " 
       << collations().size() 
       << " FED channels" << endl;
  
  // Adjust maximum (and minimum) delay(s) to find optimum sampling point(s)
  if ( time_max > 0. ) { 
    
    SiStripControlKey::ControlPath max = SiStripControlKey::path( device_max );
    cout << " Device (FEC/slot/ring/CCU/module/channel) " 
	 << max.fecCrate_ << "/" 
	 << max.fecSlot_ << "/" 
	 << max.fecRing_ << "/" 
	 << max.ccuAddr_ << "/"
	 << max.ccuChan_ << "/"
	 << " has maximum delay (rising edge) [ns]:" << time_max << endl;
    
    SiStripControlKey::ControlPath min = SiStripControlKey::path( device_min );
    cout << " Device (FEC/slot/ring/CCU/module/channel): " 
	 << min.fecCrate_ << "/" 
	 << min.fecSlot_ << "/" 
	 << min.fecRing_ << "/" 
	 << min.ccuAddr_ << "/"
	 << min.ccuChan_ << "/"
	 << " has minimum delay (rising edge) [ns]:" << time_min << endl;
    
  } else {
    cerr << "[" << __PRETTY_FUNCTION__ << "]"
	 << " Negative value for maximum time: "
	 << time_max << endl;
  }
  
  map<uint32_t,ApvTimingAnalysis>::iterator ianal = data_.begin();
  for ( ; ianal != data_.end(); ianal++ ) { 
    ianal->second.max( time_max ); 
    static uint16_t cntr = 0;
    if ( debug ) {
      stringstream ss;
      ss << " ControlKey: " << hex << setw(8) << setfill('0') << ianal->first << dec << endl;
      ianal->second.print( ss ); 
      cout << ss.str() << endl;
      cntr++;
    }
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
  histoAnalysis( false );

  // Extract data to be histogrammed
  factory_->init( histo, type, view, directory, gran );
  uint32_t xbins = factory_->extract( data_ );

  // Create summary histogram (if it doesn't already exist)
  TH1* summary = histogram( histo, type, view, directory, xbins );

  // Fill histogram with data
  factory_->fill( *summary );
  
}
