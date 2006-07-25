#include "EventFilter/SiStripRawToDigi/interface/SiStripTrivialDigiAnalysis.h"
#include <iomanip>

using namespace std;

// -----------------------------------------------------------------------------
//
void SiStripTrivialDigiAnalysis::print( stringstream& ss ) {
  ss << "  [SiStripTrivialDigiAnalysis::print]"
     << " events: "   << events_
     << " feds: "     << feds_
     << " channels: " << channels_
     << " strips: "   << strips_
     << " digis: "    << digis_;
  // Signal distribution (strip position vs frequency)
  ss << "\n  strip: ";
  for ( uint16_t ii = 0; ii < size_; ii+=(size_/16) ) { ss << setw(4) << ii << " "; }
  ss << "ovrflw";
  ss << "\n  freq : ";
  for ( uint16_t ii = 0; ii < size_; ii+=(size_/16) ) { ss << setw(4) << pos_[ii] << " "; }
  ss << "  " << setw(4) << pos_.back();
  // Signal landau (ADC counts vs frequency)
  ss << "\n  adc  : ";
  for ( uint16_t ii = 0; ii < size_; ii+=(size_/16) ) { ss << setw(4) << ii << " "; }
  ss << "ovrflw";
  ss << "\n  freq : ";
  for ( uint16_t ii = 0; ii < size_; ii+=(size_/16) ) { ss << setw(4) << adc_[ii] << " "; }
  ss << "  " << setw(4) << adc_.back();
  // Misc
  uint16_t cntr = 0;
  uint32_t tmp = 0;
  ss << "\n  adc/freq: ";
  for ( uint16_t ii = 0; ii < size_; ii++ ) { 
    if ( adc_[ii] ) { 
      if ( cntr<8 ) { ss << ii << "/" << adc_[ii] << ", "; cntr++; } 
      tmp+=adc_[ii];
    }
  }
  ss << "ovrflw: " << adc_.back();
  ss << ", total: " << tmp;
}  
