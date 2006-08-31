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
void SummaryGeneratorControlView::fillMap( const string& top_level_dir,
					   //const string& granularity,
					   const uint32_t& key, 
					   const float& value,
					   const float& error ) {

  SiStripHistoNamingScheme::ControlPath top = SiStripHistoNamingScheme::controlPath( top_level_dir );
  //SiStripHistoNamingScheme::ControlPath gran = SiStripHistoNamingScheme::controlPath( granularity );
  SiStripControlKey::ControlPath pwd = SiStripControlKey::path( key );
  
  if ( ( ( pwd.fecCrate_ == top.fecCrate_ ) || ( top.fecCrate_ == sistrip::invalid_ ) ) &&
       ( ( pwd.fecSlot_  == top.fecSlot_  ) || ( top.fecSlot_  == sistrip::invalid_ ) ) &&
       ( ( pwd.fecRing_  == top.fecRing_  ) || ( top.fecRing_  == sistrip::invalid_ ) ) && 
       ( ( pwd.ccuAddr_  == top.ccuAddr_  ) || ( top.ccuAddr_  == sistrip::invalid_ ) ) &&
       ( ( pwd.ccuChan_  == top.ccuChan_  ) || ( top.ccuChan_  == sistrip::invalid_ ) ) ) { 
    
    stringstream bin;
    if ( pwd.fecCrate_ != sistrip::invalid_ ) { bin << pwd.fecCrate_ << sistrip::pipe_; }
    if ( pwd.fecSlot_  != sistrip::invalid_ ) { bin << pwd.fecSlot_  << sistrip::pipe_; }
    if ( pwd.fecRing_  != sistrip::invalid_ ) { bin << pwd.fecRing_  << sistrip::pipe_; }
    if ( pwd.ccuAddr_  != sistrip::invalid_ ) { bin << pwd.ccuAddr_  << sistrip::pipe_; }
    if ( pwd.ccuChan_  != sistrip::invalid_ ) { bin << pwd.ccuChan_  << sistrip::pipe_; }
    if ( pwd.channel_  != sistrip::invalid_ ) { bin << pwd.channel_; }

    if ( map_.find( bin.str() ) == map_.end() ) { 
      map_[bin.str()].first = value; 
      map_[bin.str()].second = error; 
//       cout << " bin: " << bin.str()
// 	   << " value: " << value 
// 	   << " error: " << error 
// 	   << endl;
    } else { /* warning here? */ }
  
  }
  
}

//------------------------------------------------------------------------------
//
void SummaryGeneratorControlView::summaryDistr( TH1& histo ) {
  cout << "[" << __PRETTY_FUNCTION__ << "]" << endl;
  
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
  cout << " List of monitorables for " << map_.size() << " devices: ";
  map< string, pair<float,float> >::const_iterator idevice = map_.begin();
  for ( ; idevice != map_.end(); idevice++ ) {
    histo.Fill( (Int_t)(idevice->second.first) );
    cout << idevice->second.first << "+/-" << idevice->second.second << ", ";
  }
  cout << endl;

}

//------------------------------------------------------------------------------

void SummaryGeneratorControlView::summary1D( TH1& histo ) {
  cout << "[" << __PRETTY_FUNCTION__ << "]" << endl;

  // Check number of entries in map
  if ( map_.empty() ) { return; }
  
  // Set histogram number of bins and min/max
  histo.SetBins( map_.size(), 0., (Double_t)map_.size() );
 
  // Iterate through map, set bin labels and fill histogram
  cout << " List of monitorables for " << map_.size() << " devices: ";
  uint16_t ibin = 1;
  map< string, pair<float,float> >::const_iterator idevice = map_.begin();
  for ( ; idevice != map_.end(); idevice++ ) {
    histo.GetXaxis()->SetBinLabel( (Int_t)ibin, idevice->first.c_str() );
    histo.SetBinContent( (Int_t)ibin, idevice->second.first );
    histo.SetBinError( (Int_t)ibin, idevice->second.second );
    ibin++;
    cout << idevice->second.first << "+/-" << idevice->second.second << ", ";
  }
  cout << endl;

}

// //------------------------------------------------------------------------------
// //
// void SummaryGeneratorControlView::summary2D( TH1& histo ) {
//   cout << "[" << __PRETTY_FUNCTION__ << "]" << endl;

//   // Check number of entries in map
//   if ( map_.empty() ) { return; }
  
//   // Set histogram number of bins and min/max
//   histo.SetBins( map_.size(), 0., (Double_t)map_.size() );
 
//   // Iterate through map, set bin labels and fill histogram
//   cout << " List of monitorables for " << map_.size() << " devices: ";
//   uint16_t ibin = 1;
//   map< string, pair<float,float> >::const_iterator idevice = map_.begin();
//   for ( ; idevice != map_.end(); idevice++ ) {
//     histo.GetXaxis()->SetBinLabel( (Int_t)ibin, idevice->first.c_str() );
//     histo.SetBinContent( (Int_t)ibin, idevice->second.first );
//     histo.SetBinError( (Int_t)ibin, idevice->second.second );
//     ibin++;
//     cout << idevice->second.first << "+/-" << idevice->second.second << ", ";
//   }
//   cout << endl;

// }






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
