#include "DQM/SiStripCommissioningClients/interface/OptoScanHistograms.h"
#include <iostream>

using namespace std;

// -----------------------------------------------------------------------------
/** */
OptoScanHistograms::OptoScanHistograms( MonitorUserInterface* mui ) 
  : CommissioningHistograms(mui),
    gain0digital0_(),
    gain0digital1_(),
    gain1digital0_(),
    gain1digital1_(),
    gain2digital0_(),
    gain2digital1_(),
    gain3digital0_(),
    gain3digital1_()
{
  cout << "[OptoScanHistograms::OptoScanHistograms]"
       << " Created object for OPTO (bias and gain) SCAN histograms" << endl;
  task( sistrip::OPTO_SCAN ); 
}

// -----------------------------------------------------------------------------
/** */
OptoScanHistograms::~OptoScanHistograms() {
  cout << "[OptoScanHistograms::~OptoScanHistograms]" << endl;
}

// -----------------------------------------------------------------------------
/** */
void OptoScanHistograms::book( const vector<string>& me_list ) {
  
  const SiStripHistoNamingScheme::ControlPath& path = SiStripHistoNamingScheme::controlPath( this->mui()->pwd() );
  
  vector<string>::const_iterator ime = me_list.begin();
  for ( ; ime != me_list.end(); ime++ ) {
    
    SiStripHistoNamingScheme::HistoTitle title = SiStripHistoNamingScheme::histoTitle( *ime );
    if ( title.task_ != task() ) { return; }
    
    string path_and_title = this->mui()->pwd() + "/" + *ime;
    MonitorElement* me = this->mui()->get( path_and_title );
    
    uint32_t key = SiStripControlKey::key( path.fecCrate_,
					   path.fecSlot_,
					   path.fecRing_,
					   path.ccuAddr_,
					   path.ccuChan_,
					   title.channel_ );
    
    uint32_t gain_pos = title.extraInfo_.find( sistrip::gain_ );
    uint32_t digi_pos = title.extraInfo_.find( sistrip::digital_ );
    if ( gain_pos == string::npos || gain_pos == string::npos ) { 
      cerr << "[OptoScanHistograms::book]"
	   << " Unexpected value in 'HistoTitle::extraInfo_' field!" << endl;
      continue;
    }
    
    int16_t gain = -1;
    int16_t digi = -1;
    stringstream ss;
    ss.str(""); 
    ss << title.extraInfo_.substr( gain_pos+sistrip::gain_.size(), 1 ); 
    ss >> dec >> gain;
    ss.str(""); 
    ss << title.extraInfo_.substr( digi_pos+sistrip::digital_.size(), 1 ); 
    ss >> dec >> digi;
    
    if ( gain == 0 && digi == 0 ) {
      initHistoSet( title, gain0digital0_[key], me ); 
    } else if ( gain == 0 && digi == 1 ) {
      initHistoSet( title, gain0digital1_[key], me );
    } else if ( gain == 1 && digi == 0 ) {
      initHistoSet( title, gain1digital0_[key], me ); 
    } else if ( gain == 1 && digi == 1 ) {
      initHistoSet( title, gain1digital1_[key], me );
    } else if ( gain == 2 && digi == 0 ) {
      initHistoSet( title, gain2digital0_[key], me ); 
    } else if ( gain == 2 && digi == 1 ) {
      initHistoSet( title, gain2digital1_[key], me );
    } else if ( gain == 3 && digi == 0 ) {
      initHistoSet( title, gain3digital0_[key], me ); 
    } else if ( gain == 3 && digi == 1 ) {
      initHistoSet( title, gain3digital1_[key], me );
    } else {
      cerr << "[OptoScanHistograms::book]"
	   << " Unexpected value in 'HistoTitle::extraInfo_' field!" 
	   << " gain: " << gain << " digital: " << digi << endl;
    }
    
  }
  
}

// -----------------------------------------------------------------------------
/** */
void OptoScanHistograms::update() {
  map< uint32_t, HistoSet >::iterator ihis;
  
  ihis = gain0digital0_.begin();
  for ( ; ihis != gain0digital0_.end(); ihis++ ) {
    if ( ihis->second.profile_ ) { updateHistoSet( ihis->second ); }
  }
  ihis = gain0digital1_.begin();
  for ( ; ihis != gain0digital1_.end(); ihis++ ) {
    if ( ihis->second.profile_ ) { updateHistoSet( ihis->second ); }
  }
  ihis = gain1digital0_.begin();
  for ( ; ihis != gain1digital0_.end(); ihis++ ) {
    if ( ihis->second.profile_ ) { updateHistoSet( ihis->second ); }
  }
  ihis = gain1digital1_.begin();
  for ( ; ihis != gain1digital1_.end(); ihis++ ) {
    if ( ihis->second.profile_ ) { updateHistoSet( ihis->second ); }
  }
  ihis = gain2digital0_.begin();
  for ( ; ihis != gain2digital0_.end(); ihis++ ) {
    if ( ihis->second.profile_ ) { updateHistoSet( ihis->second ); }
  }
  ihis = gain2digital1_.begin();
  for ( ; ihis != gain2digital1_.end(); ihis++ ) {
    if ( ihis->second.profile_ ) { updateHistoSet( ihis->second ); }
  }
  ihis = gain3digital0_.begin();
  for ( ; ihis != gain3digital0_.end(); ihis++ ) {
    if ( ihis->second.profile_ ) { updateHistoSet( ihis->second ); }
  }
  ihis = gain3digital1_.begin();
  for ( ; ihis != gain3digital1_.end(); ihis++ ) {
    if ( ihis->second.profile_ ) { updateHistoSet( ihis->second ); }
  }

}

