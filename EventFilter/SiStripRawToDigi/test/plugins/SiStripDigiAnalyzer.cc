#include "EventFilter/SiStripRawToDigi/test/plugins/SiStripDigiAnalyzer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "CondFormats/DataRecord/interface/SiStripFedCablingRcd.h"
#include "CondFormats/SiStripObjects/interface/SiStripFedCabling.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/SiStripCommon/interface/SiStripConstants.h"
#include "DataFormats/SiStripCommon/interface/SiStripFedKey.h"
#include "DataFormats/SiStripCommon/interface/SiStripEventSummary.h"
#include "DataFormats/SiStripDigi/interface/SiStripDigi.h"
#include "DataFormats/SiStripDigi/interface/SiStripRawDigi.h"
#include <sstream>
#include <iomanip>

using namespace std;

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

// -----------------------------------------------------------------------------
//
SiStripDigiAnalyzer::SiStripDigiAnalyzer( const edm::ParameterSet& pset ) :
  inputModuleLabel_( pset.getParameter<string>( "InputModuleLabel" ) )
{
  LogDebug("SiStripDigiAnalyzer")
    << "[SiStripDigiAnalyzer::SiStripDigiAnalyzer]"
    << " Constructing object...";
}

// -----------------------------------------------------------------------------
//
SiStripDigiAnalyzer::~SiStripDigiAnalyzer() {
  LogDebug("SiStripDigiAnalyzer")
    << "[SiStripDigiAnalyzer::~SiStripDigiAnalyzer]"
    << " Destructing object...";
}

// -----------------------------------------------------------------------------
// 
void SiStripDigiAnalyzer::beginJob() {
  LogDebug("SiStripDigiAnalyzer")
    << "[SiStripDigiAnalyzer::beginJob]";
}

// -----------------------------------------------------------------------------
// 
void SiStripDigiAnalyzer::endJob() {
  stringstream ss;
  ss << "PSEUDO DIGI ANALYSIS:" << "\n";
  anal_.print(ss);
  ss << "\n";
  ss << "REAL DIGI (VIRGIN RAW) ANALYSIS:" << "\n";
  vr_r.print(ss);
  ss << "\n";
  ss << "REAL DIGI (PROCESSED RAW) ANALYSIS:" << "\n";
  pr_r.print(ss);
  ss << "\n";
  ss << "REAL DIGI (SCOPE MODE) ANALYSIS:" << "\n";
  sm_r.print(ss);
  ss << "\n";
  ss << "REAL DIGI (ZERO SUPPR) ANALYSIS:" << "\n";
  zs_r.print(ss);
  ss << "\n";
  LogDebug("SiStripDigiAnalyzer") << ss.str();

}

// -----------------------------------------------------------------------------
//
void SiStripDigiAnalyzer::analyze( const edm::Event& event, const edm::EventSetup& setup ) {

  LogDebug("SiStripDigiAnalyzer")
    << "["<<__PRETTY_FUNCTION__<<"]" 
    << " Analyzing run " << event.id().run() 
    << " and event " << event.id().event();

  // Retrieve FED (reatout) and FEC (control) cabling
  edm::ESHandle<SiStripFedCabling> fed_cabling;
  setup.get<SiStripFedCablingRcd>().get( fed_cabling ); 

  // Retrieve "real" digis
  edm::Handle< edm::DetSetVector<SiStripRawDigi> > vr;
  edm::Handle< edm::DetSetVector<SiStripRawDigi> > pr;
  edm::Handle< edm::DetSetVector<SiStripRawDigi> > sm;
  edm::Handle< edm::DetSetVector<SiStripDigi> > zs;
  event.getByLabel( inputModuleLabel_, "VirginRaw", vr );
  event.getByLabel( inputModuleLabel_, "ProcessedRaw", pr );
  event.getByLabel( inputModuleLabel_, "ScopeMode", sm );
  event.getByLabel( inputModuleLabel_, "ZeroSuppressed", zs );

  // Retrieve SiStripEventSummary
  edm::Handle<SiStripEventSummary> summary;
  event.getByLabel( inputModuleLabel_, summary );
  
  // Analyse digis
  anal_.events_++;
  vr_r.events_++;
  pr_r.events_++;
  sm_r.events_++;
  zs_r.events_++;
  vector<uint16_t>::const_iterator ifed = fed_cabling->feds().begin();
  for ( ; ifed != fed_cabling->feds().end(); ifed++ ) {
    anal_.feds_++;
    vr_r.feds_++;
    pr_r.feds_++;
    sm_r.feds_++;
    zs_r.feds_++;
    for ( uint16_t ichan = 0; ichan < sistrip::FEDCH_PER_FED; ichan++ ) { 
      anal_.channels_++;
      vr_r.channels_++;
      pr_r.channels_++;
      sm_r.channels_++;
      zs_r.channels_++;

      // Analyse digis
	
      uint32_t key = SiStripFedKey( *ifed, 
				    SiStripFedKey::feUnit(ichan),
				    SiStripFedKey::feChan(ichan) ).key();

      vector< edm::DetSet<SiStripRawDigi> >::const_iterator raw;
      vector< edm::DetSet<SiStripDigi> >::const_iterator digis;

      // virgin raw
      raw = vr->find( key );
      if ( raw != vr->end() ) { 
	for ( uint16_t istrip = 0; istrip < raw->size(); istrip++ ) { 
	  if ( raw->data[istrip].adc() ) {
	    vr_r.strips_++;
	    vr_r.pos(istrip);
	    vr_r.adc( raw->data[istrip].adc() );
	  }
	}
      }

      // processed raw
      raw = pr->find( key );
      if ( raw != pr->end() ) { 
	for ( uint16_t istrip = 0; istrip < raw->size(); istrip++ ) { 
	  if ( raw->data[istrip].adc() ) {
	    pr_r.strips_++;
	    pr_r.pos(istrip);
	    pr_r.adc( raw->data[istrip].adc() );
	  }
	}
      }

      // scope mode
      raw = sm->find( key );
      if ( raw != sm->end() ) { 
	for ( uint16_t istrip = 0; istrip < raw->size(); istrip++ ) { 
	  if ( raw->data[istrip].adc() ) {
	    sm_r.strips_++;
	    sm_r.pos(istrip);
	    sm_r.adc( raw->data[istrip].adc() );
	  }
	}
      }

      // scope mode
      digis = zs->find( key );
      if ( digis != zs->end() ) { 
	for ( uint16_t iadc = 0; iadc < digis->size(); iadc++ ) { 
	  if ( digis->data[iadc].adc() ) {
	    zs_r.strips_++;
	    zs_r.pos( digis->data[iadc].strip() );
	    zs_r.adc( digis->data[iadc].adc() );
	  }
	}
      }
      
    } // channel loop
  } // fed loop
  
}
