#include "DQM/SiStripCommissioningClients/interface/FedCablingHistograms.h"
#include <iostream>

using namespace std;

// -----------------------------------------------------------------------------
/** */
FedCablingHistograms::FedCablingHistograms( MonitorUserInterface* mui ) 
  : CommissioningHistograms(mui),
    cabling_()
{
  cout << "[FedCablingHistograms::FedCablingHistograms]"
       << " Created object for FED CABLING histograms" << endl;
  task( sistrip::FED_CABLING ); 
}

// -----------------------------------------------------------------------------
/** */
FedCablingHistograms::~FedCablingHistograms() {
  cout << "[FedCablingHistograms::~FedCablingHistograms]" << endl;
}

// -----------------------------------------------------------------------------
/** */
void FedCablingHistograms::book( const vector<string>& me_list ) {

//   const SiStripHistoNamingScheme::ControlPath& path = SiStripHistoNamingScheme::controlPath( this->mui()->pwd() );
  
//   vector<string>::const_iterator ime = me_list.begin();
//   for ( ; ime != me_list.end(); ime++ ) {
    
//     SiStripHistoNamingScheme::HistoTitle title = SiStripHistoNamingScheme::histoTitle( *ime );
//     if ( title.task_ != task() ) { return; }

//     string path_and_title = this->mui()->pwd() + "/" + *ime;
//     MonitorElement* me = this->mui()->get( path_and_title );
    
//     uint32_t key = SiStripControlKey::key( path.fecCrate_,
// 					   path.fecSlot_,
// 					   path.fecRing_,
// 					   path.ccuAddr_,
// 					   path.ccuChan_,
// 					   title.channel_ );
//     initHistoSet( title, timing_[key], me ); 
    
//   }
  
}

// -----------------------------------------------------------------------------
/** */
void FedCablingHistograms::update() {
//   map< uint32_t, HistoSet >::iterator ihis = timing_.begin();
//   for ( ; ihis != timing_.end(); ihis++ ) {
//     if ( ihis->second.profile_ ) { updateHistoSet( ihis->second ); }
//   }
}

