#include "DQM/SiStripCommissioningClients/interface/PedestalsHistograms.h"
#include <iostream>

using namespace std;

// -----------------------------------------------------------------------------
/** */
PedestalsHistograms::PedestalsHistograms( MonitorUserInterface* mui ) 
  : CommissioningHistograms(mui),
    mePeds_(),
    meNoise_(),
    meCommonModeApv0_(),
    meCommonModeApv1_()
{
  cout << "[PedestalsHistograms::PedestalsHistograms]"
       << " Created object for pedestals histograms" << endl;
  task( sistrip::PEDESTALS ); 
}

// -----------------------------------------------------------------------------
/** */
PedestalsHistograms::~PedestalsHistograms() {
  cout << "[PedestalsHistograms::~PedestalsHistograms]" << endl;
}

// -----------------------------------------------------------------------------
/** */
void PedestalsHistograms::book( const vector<string>& me_list ) {

  //@@ PROBLEM!!! this is slow? with line below uncommented, sources do not update
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
    if ( title.extraInfo_.find( sistrip::pedsAndRawNoise_ ) ) { 
      fillHistoSet( title, mePeds_[key], me ); 
    } else if ( title.extraInfo_.find( sistrip::residualsAndNoise_ ) ) { 
      fillHistoSet( title, meNoise_[key], me );
    } else if ( title.extraInfo_.find( sistrip::commonMode_ ) ) {
      if ( title.channel_%2 == 0 ) { 
	fillHistoSet( title, meCommonModeApv0_[key], me ); 
      } else { 
	fillHistoSet( title, meCommonModeApv1_[key], me ); 
      }
    } else {
      cerr << "[PedestalsHistograms::book]"
	   << " Unexpected value in 'HistoTitle::extraInfo_' field!" << endl;
    }
    
    //if ( mePeds_.find( key ) == mePeds_.end() ) { 
    
  }
  
}

// -----------------------------------------------------------------------------
/** */
void PedestalsHistograms::update() {
  map< uint32_t, HistoSet >::iterator ipeds = mePeds_.begin();
  for ( ; ipeds != mePeds_.end(); ipeds++ ) {
    if ( ipeds->second.profile_ ) { updateHistoSet( ipeds->second ); }
  }
  map< uint32_t, HistoSet >::iterator inoise = meNoise_.begin();
  for ( ; inoise != meNoise_.end(); inoise++ ) {
    if ( inoise->second.profile_ ) { updateHistoSet( inoise->second ); }
  }
  map< uint32_t, HistoSet >::iterator iapv0 = meCommonModeApv0_.begin();
  for ( ; iapv0 != meCommonModeApv0_.end(); iapv0++ ) {
    if ( iapv0->second.profile_ ) { updateHistoSet( iapv0->second ); }
  }
  map< uint32_t, HistoSet >::iterator iapv1 = meCommonModeApv1_.begin();
  for ( ; iapv1 != meCommonModeApv1_.end(); iapv1++ ) {
    if ( iapv1->second.profile_ ) { updateHistoSet( iapv1->second ); }
  }
}
