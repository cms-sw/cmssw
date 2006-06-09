#include "DQM/SiStripCommissioningClients/interface/FedTimingHistograms.h"
#include <iostream>

using namespace std;

// -----------------------------------------------------------------------------
/** */
FedTimingHistograms::FedTimingHistograms( MonitorUserInterface* mui ) 
  : CommissioningHistograms(mui),
    timing_()
{
  cout << "[FedTimingHistograms::FedTimingHistograms]"
       << " Created object for FED TIMING histograms" << endl;
  task( sistrip::FED_TIMING ); 
}

// -----------------------------------------------------------------------------
/** */
FedTimingHistograms::~FedTimingHistograms() {
  cout << "[FedTimingHistograms::~FedTimingHistograms]" << endl;
}

// -----------------------------------------------------------------------------
/** */
void FedTimingHistograms::book( const vector<string>& me_list ) {

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
    initHistoSet( title, timing_[key], me ); 
    
  }
  
}

// -----------------------------------------------------------------------------
/** */
void FedTimingHistograms::update() {
  map< uint32_t, HistoSet >::iterator ihis = timing_.begin();
  for ( ; ihis != timing_.end(); ihis++ ) {
    if ( ihis->second.profile_ ) { updateHistoSet( ihis->second ); }
  }
}

