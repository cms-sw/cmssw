#include "DQM/SiStripCommissioningClients/interface/VpspScanHistograms.h"
#include <iostream>

using namespace std;

// -----------------------------------------------------------------------------
/** */
VpspScanHistograms::VpspScanHistograms( MonitorUserInterface* mui ) 
  : CommissioningHistograms(mui),
    vpspApv0_(),
    vpspApv1_()
{
  cout << "[VpspScanHistograms::VpspScanHistograms]"
       << " Created object for PEDESTALS histograms" << endl;
  task( sistrip::VPSP_SCAN ); 
}

// -----------------------------------------------------------------------------
/** */
VpspScanHistograms::~VpspScanHistograms() {
  cout << "[VpspScanHistograms::~VpspScanHistograms]" << endl;
}

// -----------------------------------------------------------------------------
/** */
void VpspScanHistograms::book( const vector<string>& me_list ) {
  
  const SiStripHistoNamingScheme::ControlPath& path = SiStripHistoNamingScheme::controlPath( this->mui()->pwd() );
  
  vector<string>::const_iterator ime = me_list.begin();
  for ( ; ime != me_list.end(); ime++ ) {
    
    SiStripHistoNamingScheme::HistoTitle title = SiStripHistoNamingScheme::histoTitle( *ime );
    if ( title.task_ != task() ) { return; }
    
    string path_and_title = this->mui()->pwd() + "/" + *ime;
    MonitorElement* me = this->mui()->get( path_and_title );
    
    if ( title.channel_ < 32 ||
	 title.channel_ > 37 ) { 
      cerr << "[VpspScanHistograms::book]"
	   << " Unexpected value in 'HistoTitle::channel_' field! " 
	   << title.channel_ << endl;
      continue;
    }

    uint16_t lld_channel = ( title.channel_ - 32 ) / 2;
    uint32_t key = SiStripControlKey::key( path.fecCrate_,
					   path.fecSlot_,
					   path.fecRing_,
					   path.ccuAddr_,
					   path.ccuChan_,
					   lld_channel );
    if ( title.channel_%2 == 0 ) { 
      initHistoSet( title, vpspApv0_[key], me ); 
    } else { 
      initHistoSet( title, vpspApv1_[key], me ); 
    }
    
  }
  
}

// -----------------------------------------------------------------------------
/** */
void VpspScanHistograms::update() {

  map< uint32_t, HistoSet >::iterator ihis;
  ihis = vpspApv0_.begin();
  for ( ; ihis != vpspApv0_.end(); ihis++ ) {
    if ( ihis->second.profile_ ) { updateHistoSet( ihis->second ); }
  }
  ihis = vpspApv1_.begin();
  for ( ; ihis != vpspApv1_.end(); ihis++ ) {
    if ( ihis->second.profile_ ) { updateHistoSet( ihis->second ); }
  }

}

