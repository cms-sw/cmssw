#include "EventFilter/SiStripRawToDigi/interface/SiStripTrivialDigiAnalysis.h"
#include <iostream>
#include <sstream>
#include <iomanip>

// -----------------------------------------------------------------------------
/** */
SiStripTrivialDigiAnalysis::SiStripTrivialDigiAnalysis( string class_name ) : 
  name_(class_name),
  zsPos_(), zsAdc_(),
  vrPos_(), vrAdc_(),
  prPos_(), prAdc_(),
  smPos_(), smAdc_(),
  nEvents_(0), nFeds_(0), nChans_(0), nDets_(0), 
  zsDigis_(0),
  vrDigis_(0),
  prDigis_(0),
  smDigis_(0)
{
  cout << "[SiStripTrivialDigiAnalysis::SiStripTrivialDigiAnalysis]" 
       << " Constructing object (owned by class " << name_ << ")..." << endl;
  zsPos_.clear(); zsAdc_.clear(); 
  vrPos_.clear(); vrAdc_.clear(); 
  prPos_.clear(); prAdc_.clear(); 
  smPos_.clear(); smAdc_.clear(); 
  zsPos_.resize(768,0);  zsAdc_.resize(1024,0); //@@ should be 256?!
  vrPos_.resize(768,0);  vrAdc_.resize(1024,0);
  prPos_.resize(768,0);  prAdc_.resize(1024,0);
  smPos_.resize(1024,0); smAdc_.resize(1024,0);
}

// -----------------------------------------------------------------------------
/** */
SiStripTrivialDigiAnalysis::~SiStripTrivialDigiAnalysis() {
  cout << "[SiStripTrivialDigiAnalysis::~SiStripTrivialDigiAnalysis]" 
       << " Destructing object..." << endl;
  print();
  zsPos_.clear(); zsAdc_.clear(); 
  vrPos_.clear(); vrAdc_.clear(); 
  prPos_.clear(); prAdc_.clear(); 
  smPos_.clear(); smAdc_.clear(); 
}

// -----------------------------------------------------------------------------
/** */
void SiStripTrivialDigiAnalysis::zsDigi( uint16_t strip, uint16_t adc ) { 
  if ( strip <= 768 && adc <= 1024 ) { //@@ should be 256?!
    zsDigis_++; 
    zsPos_[strip]++; 
    zsAdc_[adc]++; 
  } else {
    cerr << "[SiStripTrivialDigiAnalysis::zsDigi]"
	 << " Unexpected value!"
	 << " Strip: " << strip
	 << " ADC: " << adc
	 << endl;
  }
}

// -----------------------------------------------------------------------------
/** */
void SiStripTrivialDigiAnalysis::vrDigi( uint16_t strip, uint16_t adc ) { 
  if ( strip <= 768 && adc <= 1024 ) {
    vrDigis_++; 
    vrPos_[strip]++; 
    vrAdc_[adc]++; 
  } else {
    cerr << "[SiStripTrivialDigiAnalysis::vrDigi]"
	 << " Unexpected value! "
	 << " Strip: " << strip
	 << " ADC: " << adc
	 << endl;
  }
}

// -----------------------------------------------------------------------------
/** */
void SiStripTrivialDigiAnalysis::prDigi( uint16_t strip, uint16_t adc ) { 
  if ( strip <= 768 && adc <= 1024 ) {
    prDigis_++; 
    prPos_[strip]++; 
    prAdc_[adc]++; 
  } else {
    cerr << "[SiStripTrivialDigiAnalysis::prDigi]"
	 << " Unexpected value! "
	 << " Strip: " << strip
	 << " ADC: " << adc
	 << endl;
  }
}

// -----------------------------------------------------------------------------
/** */
void SiStripTrivialDigiAnalysis::smDigi( uint16_t strip, uint16_t adc ) { 
  if ( strip <= 1024 && adc <= 1024 ) {
    smDigis_++; 
    smPos_[strip]++; 
    smAdc_[adc]++; 
  } else {
    cerr << "[SiStripTrivialDigiAnalysis::smDigi]"
	 << " Unexpected value! "
	 << " Strip: " << strip
	 << " ADC: " << adc
	 << endl;
  }
}

// -----------------------------------------------------------------------------
/** */
void SiStripTrivialDigiAnalysis::print() {
  cout << "[SiStripTrivialDigiAnalysis::print]"
       << " class: " << name_ 
       << "  nEvents_: " << nEvents_ 
       << "  nFeds_: " << nFeds_ 
       << "  nChans_: " << nChans_ 
       << "  nDets_: " << nDets_ 
       << "  zsDigis_: " << zsDigis_ 
       << "  vrDigis_: " << vrDigis_ 
       << "  prDigis_: " << prDigis_ 
       << "  smDigis_: " << smDigis_ 
       << endl;
  
  stringstream ss;
  uint16_t tmp;
  uint16_t total;
  
  // POSITION

  total = 0; 
  
  // ZS
  tmp = 0; ss.str("");
  ss << "[SiStripTrivialDigiAnalysis::print]"
     << " ZS Digi position (strip = 0->9): ";
  for ( uint16_t i = 0; i < zsPos_.size(); i++ ) {
    if ( i<10 ) { ss << setw(5) << zsPos_[i] << ", "; }
    tmp += zsPos_[i];
  }
  ss << " (total: " << tmp << ")";
  cout << ss.str() << endl;
  total += tmp;

  // VR
  tmp = 0; ss.str("");
  ss << "[SiStripTrivialDigiAnalysis::print]"
     << " VR Digi position (strip = 0->9): ";
  for ( uint16_t i = 0; i < vrPos_.size(); i++ ) {
    if ( i<10 ) { ss << setw(5) << vrPos_[i] << ", "; }
    tmp += vrPos_[i];
  }
  ss << " (total: " << tmp << ")";
  cout << ss.str() << endl;
  total += tmp;

  // PR
  tmp = 0; ss.str("");
  ss << "[SiStripTrivialDigiAnalysis::print]"
     << " PR Digi position (strip = 0->9): ";
  for ( uint16_t i = 0; i < prPos_.size(); i++ ) {
    if ( i<10 ) { ss << setw(5) << prPos_[i] << ", "; }
    tmp += prPos_[i];
  }
  ss << " (total: " << tmp << ")";
  cout << ss.str() << endl;
  total += tmp;
  
  // SM
  tmp = 0; ss.str("");
  ss << "[SiStripTrivialDigiAnalysis::print]"
     << " SM Digi position (strip = 0->9): ";
  for ( uint16_t i = 0; i < smPos_.size(); i++ ) {
    if ( i<10 ) { ss << setw(5) << smPos_[i] << ", "; }
    tmp += smPos_[i];
  }
  ss << " (total: " << tmp << ")";
  cout << ss.str() << endl;
  total += tmp;
  
  cout << "[SiStripTrivialDigiAnalysis::print]"
       << " Total number of digis (ZS+VR+PR+SM): "
       << total << endl;

  // LANDAU

  total = 0; 
  
  // ZS
  tmp = 0; ss.str("");
  ss << "[SiStripTrivialDigiAnalysis::print]"
     << " ZS Digi Landau (ADC val = 0->9): ";
  for ( uint16_t i = 0; i < zsAdc_.size(); i++ ) {
    if ( i<10 ) { ss << setw(5) << zsAdc_[i] << ", "; }
    tmp += zsAdc_[i];
  }
  ss << " (total: " << tmp << ")";
  cout << ss.str() << endl;
  total += tmp;

  // VR
  tmp = 0; ss.str("");
  ss << "[SiStripTrivialDigiAnalysis::print]"
     << " VR Digi Landau (ADC val = 0->9): ";
  for ( uint16_t i = 0; i < vrAdc_.size(); i++ ) {
    if ( i<10 ) { ss << setw(5) << vrAdc_[i] << ", "; }
    tmp += vrAdc_[i];
  }
  ss << " (total: " << tmp << ")";
  cout << ss.str() << endl;
  total += tmp;

  // PR
  tmp = 0; ss.str("");
  ss << "[SiStripTrivialDigiAnalysis::print]"
     << " PR Digi Landau (ADC val = 0->9): ";
  for ( uint16_t i = 0; i < prAdc_.size(); i++ ) {
    if ( i<10 ) { ss << setw(5) << prAdc_[i] << ", "; }
    tmp += prAdc_[i];
  }
  ss << " (total: " << tmp << ")";
  cout << ss.str() << endl;
  total += tmp;
  
  // SM
  tmp = 0; ss.str("");
  ss << "[SiStripTrivialDigiAnalysis::print]"
     << " SM Digi Landau (ADC val = 0->9): ";
  for ( uint16_t i = 0; i < smAdc_.size(); i++ ) {
    if ( i<10 ) { ss << setw(5) << smAdc_[i] << ", "; }
    tmp += smAdc_[i];
  }
  ss << " (total: " << tmp << ")";
  cout << ss.str() << endl;
  total += tmp;
  
  cout << "[SiStripTrivialDigiAnalysis::print]"
       << " Total number of digis (ZS+VR+PR+SM): "
       << total << endl;
  
}



