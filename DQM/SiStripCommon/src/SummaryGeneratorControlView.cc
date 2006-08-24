#include "DQM/SiStripCommon/interface/SummaryGeneratorControlView.h"
#include "DQM/SiStripCommon/interface/SiStripHistoNamingScheme.h"
#include "DataFormats/SiStripCommon/interface/SiStripEnumeratedTypes.h"
#include "DataFormats/SiStripDetId/interface/SiStripControlKey.h"
#include <iostream>
#include <sstream>
#include <cmath>

using namespace std;

// -----------------------------------------------------------------------------
// 
void SummaryGeneratorControlView::fillMap( const string& directory,
					   const uint32_t& key, 
					   const float& value,
					   const float& error ) {
  
  // Create control path structs for both histo level and key
  SiStripHistoNamingScheme::ControlPath level = SiStripHistoNamingScheme::controlPath( directory );
  SiStripControlKey::ControlPath path = SiStripControlKey::path( key );
  
  if ( ( ( path.fecCrate_ == level.fecCrate_ ) || ( level.fecCrate_ == sistrip::invalid_ ) ) &&
       ( ( path.fecSlot_  == level.fecSlot_  ) || ( level.fecSlot_  == sistrip::invalid_ ) ) &&
       ( ( path.fecRing_  == level.fecRing_  ) || ( level.fecRing_  == sistrip::invalid_ ) ) && 
       ( ( path.ccuAddr_  == level.ccuAddr_  ) || ( level.ccuAddr_  == sistrip::invalid_ ) ) &&
       ( ( path.ccuChan_  == level.ccuChan_  ) || ( level.ccuChan_  == sistrip::invalid_ ) ) ) { 
    
    stringstream bin;
    if ( path.fecCrate_ != sistrip::invalid_ ) { bin << path.fecCrate_ << sistrip::pipe_; }
    if ( path.fecSlot_  != sistrip::invalid_ ) { bin << path.fecSlot_  << sistrip::pipe_; }
    if ( path.fecRing_  != sistrip::invalid_ ) { bin << path.fecRing_  << sistrip::pipe_; }
    if ( path.ccuAddr_  != sistrip::invalid_ ) { bin << path.ccuAddr_  << sistrip::pipe_; }
    if ( path.ccuChan_  != sistrip::invalid_ ) { bin << path.ccuChan_  << sistrip::pipe_; }
    if ( path.channel_  != sistrip::invalid_ ) { bin << path.channel_; }

    if ( map_.find( bin.str() ) == map_.end() ) { 
      map_[bin.str()].first = value; 
      map_[bin.str()].second = error; 
//       cout << " bin: " << bin.str()
// 	   << " value: " << value 
// 	   << " error: " << error 
// 	   << endl;
    } else { /* warning here? */ }
  
  }
//   cout << "map_.size(): " << map_.size() << endl;
  
}

//------------------------------------------------------------------------------

void SummaryGeneratorControlView::logicalView( TH1& histo ) {
  
  // Check number of entries in map
  if ( map_.empty() ) { return; }
  
  // Set histogram number of bins and min/max
  histo.SetBins( map_.size(), 0., (Double_t)map_.size() );
  
  // Iterate through map, set bin labels and fill histogram
  uint16_t ibin = 1;
  map< string, pair<float,float> >::const_iterator idevice = map_.begin();
  for ( ; idevice != map_.end(); idevice++ ) {
    histo.GetXaxis()->SetBinLabel( (Int_t)ibin, idevice->first.c_str() );
    histo.SetBinContent( (Int_t)ibin, idevice->second.first );
    histo.SetBinError( (Int_t)ibin, idevice->second.second );
    ibin++;
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
  if ( diff ) { histo.SetBins( int(diff+abs(0.2*diff)), low-abs(0.1*diff), high+abs(0.1*diff) ); }
  else { histo.SetBins( 2, low-1., low+1. ); }
  
  // Fill histogram
  map< string, pair<float,float> >::const_iterator idevice = map_.begin();
  for ( ; idevice != map_.end(); idevice++ ) {
    histo.Fill( (Int_t)(idevice->second.first) );
  }

}






/*
  OLD:
  if ( ( ( device_path.fecCrate_ == path.fecCrate_ ) || ( path.fecCrate_ == sistrip::invalid_ ) ) &&
  ( ( device_path.fecSlot_  == path.fecSlot_  ) || ( path.fecSlot_  == sistrip::invalid_ ) ) &&
  ( ( device_path.fecRing_  == path.fecRing_  ) || ( path.fecRing_  == sistrip::invalid_ ) ) && 
  ( ( device_path.ccuAddr_  == path.ccuAddr_  ) || ( path.ccuAddr_  == sistrip::invalid_ ) ) &&
  ( ( device_path.ccuChan_  == path.ccuChan_  ) || ( path.ccuChan_  == sistrip::invalid_ ) ) ) { }
*/

/*
  NEW:
  if ( ( device_path.fecCrate_ && path.fecCrate_ == path.fecCrate_ ) && 
  ( device_path.fecSlot_  && path.fecSlot_  == path.fecSlot_  ) && 
  ( device_path.fecRing_  && path.fecRing_  == path.fecRing_  ) && 
  ( device_path.ccuAddr_  && path.ccuAddr_  == path.ccuAddr_  ) && 
  ( device_path.ccuChan_  && path.ccuChan_  == path.ccuChan_  ) ) { }
*/
