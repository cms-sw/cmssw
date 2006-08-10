#include "DQM/SiStripCommon/interface/SummaryGeneratorControlView.h"
#include "DQM/SiStripCommon/interface/SiStripHistoNamingScheme.h"
#include "DataFormats/SiStripCommon/interface/SiStripEnumeratedTypes.h"
#include "DataFormats/SiStripDetId/interface/SiStripControlKey.h"
#include <sstream>
#include <cmath>

using namespace std;

// -----------------------------------------------------------------------------
// 
void SummaryGeneratorControlView::fillMap( const uint32_t& level,
					   const uint32_t& key, 
					   const float& value,
					   const float& error ) {
  
  // Clear map
  map_.clear();

  // Create control path structs for both histo level and key
  SiStripControlKey::ControlPath level_path = SiStripControlKey::path( level );
  SiStripControlKey::ControlPath key_path = SiStripControlKey::path( key );

  if ( ( key_path.fecCrate_ && level_path.fecCrate_ == level_path.fecCrate_ ) && 
       ( key_path.fecSlot_  && level_path.fecSlot_  == level_path.fecSlot_  ) && 
       ( key_path.fecRing_  && level_path.fecRing_  == level_path.fecRing_  ) && 
       ( key_path.ccuAddr_  && level_path.ccuAddr_  == level_path.ccuAddr_  ) && 
       ( key_path.ccuChan_  && level_path.ccuChan_  == level_path.ccuChan_  ) ) { 
    
    stringstream bin;
    if ( key_path.fecCrate_ != sistrip::all_ ) { bin << key_path.fecCrate_ << sistrip::pipe_; }
    if ( key_path.fecSlot_  != sistrip::all_ ) { bin << key_path.fecSlot_  << sistrip::pipe_; }
    if ( key_path.fecRing_  != sistrip::all_ ) { bin << key_path.fecRing_  << sistrip::pipe_; }
    if ( key_path.ccuAddr_  != sistrip::all_ ) { bin << key_path.ccuAddr_  << sistrip::pipe_; }
    if ( key_path.ccuChan_  != sistrip::all_ ) { bin << key_path.ccuChan_  << sistrip::pipe_; }
    if ( key_path.channel_  != sistrip::all_ ) { bin << key_path.channel_; }
    
    if ( map_.find( bin.str() ) == map_.end() ) { 
      map_[bin.str()].first = value; 
      map_[bin.str()].second = error; 
    } else { /* warning here? */ }
  
  }
  
}

//------------------------------------------------------------------------------

void SummaryGeneratorControlView::logicalView( TH1& histo ) {
  
  // Check number of entries in map
  if ( map_.empty() ) { return; }
  
  // Set histogram number of bins and min/max
  histo.SetBins( map_.size(), 0., (Double_t)map_.size() );

  // Iterate through map, set bin labels and fill histogram
  uint16_t ibin = 0;
  map< string, pair<float,float> >::const_iterator idevice = map_.begin();
  for ( ; idevice != map_.end(); idevice++ ) {
    histo.GetXaxis()->SetBinLabel( (Int_t)ibin, idevice->first.c_str() );
    histo.SetBinContent( (Int_t)ibin, idevice->second.first );
    histo.SetBinError( (Int_t)ibin, idevice->second.second );
  }
  
}

//------------------------------------------------------------------------------
//
void SummaryGeneratorControlView::simpleDistr( TH1& histo ) {
  
  // Check number of entries in map
  if ( map_.empty() ) { return; }
  
  // Calculate bin range
  pair<float,float> ran = range();
  int high = (int)ceil( fabs(ran.first) );
  int low  = (int)ceil( fabs(ran.second) );
  int diff = abs(high) - abs(low);
  
  // Set histogram binning
  if ( diff ) {
    histo.SetBins( int(diff+abs(0.2*diff)), low-abs(0.1*diff), high+abs(0.1*diff) );
  } else { histo.SetBins( 2, low-1., low+1. ); }
  
  // Fill histogram
  map< string, pair<float,float> >::const_iterator idevice = map_.begin();
  for ( ; idevice != map_.end(); idevice++ ) {
    histo.Fill( (Int_t)(idevice->second.first) );
  }

}




/*
  OLD:
  if ( ( ( device_path.fecCrate_ == path.fecCrate_ ) || ( path.fecCrate_ == sistrip::all_ ) ) &&
  ( ( device_path.fecSlot_  == path.fecSlot_  ) || ( path.fecSlot_  == sistrip::all_ ) ) &&
  ( ( device_path.fecRing_  == path.fecRing_  ) || ( path.fecRing_  == sistrip::all_ ) ) && 
  ( ( device_path.ccuAddr_  == path.ccuAddr_  ) || ( path.ccuAddr_  == sistrip::all_ ) ) &&
  ( ( device_path.ccuChan_  == path.ccuChan_  ) || ( path.ccuChan_  == sistrip::all_ ) ) ) { }
*/

/*
  NEW:
  if ( ( device_path.fecCrate_ && path.fecCrate_ == path.fecCrate_ ) && 
  ( device_path.fecSlot_  && path.fecSlot_  == path.fecSlot_  ) && 
  ( device_path.fecRing_  && path.fecRing_  == path.fecRing_  ) && 
  ( device_path.ccuAddr_  && path.ccuAddr_  == path.ccuAddr_  ) && 
  ( device_path.ccuChan_  && path.ccuChan_  == path.ccuChan_  ) ) { }
*/
