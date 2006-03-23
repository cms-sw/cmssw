#include "EventFilter/SiStripRawToDigi/interface/SiStripTrivialDigiAnalysis.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
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
  edm::LogInfo("TrivialDigiAnalysis") << "[SiStripTrivialDigiAnalysis::SiStripTrivialDigiAnalysis]" 
				      << " Constructing object (owned by class " << name_ << ")...";
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
  edm::LogInfo("TrivialDigiAnalysis") << "[SiStripTrivialDigiAnalysis::~SiStripTrivialDigiAnalysis]" 
				      << " Destructing object...";
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
    edm::LogError("TrivialDigiSource") << "[SiStripTrivialDigiAnalysis::zsDigi]"
				       << " Unexpected value!"
				       << " Strip: " << strip
				       << " ADC: " << adc;
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
    edm::LogError("TrivialDigiSource") << "[SiStripTrivialDigiAnalysis::vrDigi]"
	 << " Unexpected value! "
	 << " Strip: " << strip
	 << " ADC: " << adc
	;
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
    edm::LogError("TrivialDigiSource") << "[SiStripTrivialDigiAnalysis::prDigi]"
				       << " Unexpected value! "
				       << " Strip: " << strip
				       << " ADC: " << adc;
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
    edm::LogError("TrivialDigiSource") << "[SiStripTrivialDigiAnalysis::smDigi]"
				       << " Unexpected value! "
				       << " Strip: " << strip
				       << " ADC: " << adc;
  }
}

// -----------------------------------------------------------------------------
/** */
void SiStripTrivialDigiAnalysis::print() {
  edm::LogInfo("TrivialDigiAnalysis") << "[SiStripTrivialDigiAnalysis::print]"
				      << " class: " << name_ 
				      << "  nEvents_: " << nEvents_ 
				      << "  nFeds_: " << nFeds_ 
				      << "  nChans_: " << nChans_ 
				      << "  nDets_: " << nDets_ 
				      << "  zsDigis_: " << zsDigis_ 
				      << "  vrDigis_: " << vrDigis_ 
				      << "  prDigis_: " << prDigis_ 
				      << "  smDigis_: " << smDigis_;
  
  stringstream ss;
  uint16_t cntr;
  uint16_t tmp;
  uint16_t total;
  
  // POSITION

  total = 0; 
  
  // ZS
  tmp = 0; ss.str("");
  ss << "[SiStripTrivialDigiAnalysis::print]"
     << " ZS Digi position (strip/freq): ";
  for ( uint16_t i = 0; i < zsPos_.size(); i++ ) {
    if ( !(i%128) && zsPos_[i] ) { ss << i << "/" << zsPos_[i] << ", "; }
    tmp += zsPos_[i];
  }
  ss << " (total: " << tmp << ")";
  edm::LogInfo("TrivialDigiAnalysis") << ss.str();
  total += tmp;
  
  // VR
  tmp = 0; ss.str("");
  ss << "[SiStripTrivialDigiAnalysis::print]"
     << " VR Digi position (strip/freq): ";
  for ( uint16_t i = 0; i < vrPos_.size(); i++ ) {
    if ( !(i%128) && vrPos_[i] ) { ss << i << "/" << vrPos_[i] << ", "; }
    tmp += vrPos_[i];
  }
  ss << " (total: " << tmp << ")";
  edm::LogInfo("TrivialDigiAnalysis") << ss.str();
  total += tmp;

  // PR
  tmp = 0; ss.str("");
  ss << "[SiStripTrivialDigiAnalysis::print]"
     << " PR Digi position (strip/freq): ";
  for ( uint16_t i = 0; i < prPos_.size(); i++ ) {
    if ( !(i%128) && prPos_[i] ) { ss << i << "/" << prPos_[i] << ", "; }
    tmp += prPos_[i];
  }
  ss << " (total: " << tmp << ")";
  edm::LogInfo("TrivialDigiAnalysis") << ss.str();
  total += tmp;
  
  // SM
  tmp = 0; ss.str("");
  ss << "[SiStripTrivialDigiAnalysis::print]"
     << " SM Digi position (strip/freq): ";
  for ( uint16_t i = 0; i < smPos_.size(); i++ ) {
    if ( !(i%128) && smPos_[i] ) { ss << i << "/" << smPos_[i] << ", "; }
    tmp += smPos_[i];
  }
  ss << " (total: " << tmp << ")";
  edm::LogInfo("TrivialDigiAnalysis") << ss.str();
  total += tmp;
  
  edm::LogInfo("TrivialDigiAnalysis") << "[SiStripTrivialDigiAnalysis::print]"
				      << " Total number of digis (ZS+VR+PR+SM): " << total;

  // LANDAU

  total = 0; 
  
  // ZS
  cntr = 0; tmp = 0; ss.str("");
  ss << "[SiStripTrivialDigiAnalysis::print]"
     << " ZS Digi Landau (ADC/freq): ";
  for ( uint16_t i = 0; i < zsAdc_.size(); i++ ) {
    if ( zsAdc_[i] ) {
      if ( cntr<10 ) { ss << i << "/" << zsAdc_[i] << ", "; } 
      cntr++;
    }
    tmp += zsAdc_[i];
  }
  ss << " (total: " << tmp << ")";
  edm::LogInfo("TrivialDigiAnalysis") << ss.str();
  total += tmp;

  // VR
  cntr = 0; tmp = 0; ss.str("");
  ss << "[SiStripTrivialDigiAnalysis::print]"
     << " VR Digi Landau (ADC/freq): ";
  for ( uint16_t i = 0; i < vrAdc_.size(); i++ ) {
    if ( vrAdc_[i] ) {
      if ( cntr<10 ) { ss << i << "/" << vrAdc_[i] << ", "; } 
      cntr++;
    }
    tmp += vrAdc_[i];
  }
  ss << " (total: " << tmp << ")";
  edm::LogInfo("TrivialDigiAnalysis") << ss.str();
  total += tmp;

  // PR
  cntr = 0; tmp = 0; ss.str("");
  ss << "[SiStripTrivialDigiAnalysis::print]"
     << " PR Digi Landau (ADC/freq): ";
  for ( uint16_t i = 0; i < prAdc_.size(); i++ ) {
    if ( prAdc_[i] ) {
      if ( cntr<10 ) { ss << i << "/" << prAdc_[i] << ", "; } 
      cntr++;
    }
    tmp += prAdc_[i];
  }
  ss << " (total: " << tmp << ")";
  edm::LogInfo("TrivialDigiAnalysis") << ss.str();
  total += tmp;
  
  // SM
  cntr = 0; tmp = 0; ss.str("");
  ss << "[SiStripTrivialDigiAnalysis::print]"
     << " SM Digi Landau (ADC/freq): ";
  for ( uint16_t i = 0; i < smAdc_.size(); i++ ) {
    if ( smAdc_[i] ) {
      if ( cntr<10 ) { ss << i << "/" << smAdc_[i] << ", "; } 
      cntr++;
    }
    tmp += smAdc_[i];
  }
  ss << " (total: " << tmp << ")";
  edm::LogInfo("TrivialDigiAnalysis") << ss.str();
  total += tmp;
  
  edm::LogInfo("TrivialDigiAnalysis") << "[SiStripTrivialDigiAnalysis::print]"
				      << " Total number of digis (ZS+VR+PR+SM): " << total;
  
}



