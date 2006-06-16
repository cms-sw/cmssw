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
  cout << "[PedestalsHistograms::book]" << endl;

  // Retrieve control path params based on directory path
  const SiStripHistoNamingScheme::ControlPath& path = SiStripHistoNamingScheme::controlPath( mui()->pwd() );
  
  cout << "[PedestalsHistograms::book]"
       << " Dir: " << mui()->pwd()
       << " FecCrate/FecRing/FecSlot/CcuAddr/CcuChan: "
       << path.fecCrate_ << "/"
       << path.fecSlot_ << "/"
       << path.fecRing_ << "/"
       << path.ccuAddr_ << "/"
       << path.ccuChan_ << endl;
  
  // Iterate through monitor element strings ("locations") in pwd
  cout << "[PedestalsHistograms::book]"
       << " Found " << me_list.size() << " MEs in " << mui()->pwd() << endl;
  vector<string>::const_iterator ime = me_list.begin();
  for ( ; ime != me_list.end(); ime++ ) {
    cout << "[PedestalsHistograms::book]"
	 << " ME string: " << *ime << endl;
    
    // Extract various parameters from histogram name
    SiStripHistoNamingScheme::HistoTitle title = SiStripHistoNamingScheme::histoTitle( *ime );
    if ( title.task_ != task() ) { 
      cout << "[PedestalsHistograms::book]" 
	   << " Histo title is inconsistent with commissioning task!" << endl;
      return; 
    }
    
    // Retrieve pointer to monitor element
    string path_and_title = this->mui()->pwd() + "/" + *ime;
    MonitorElement* me = this->mui()->get( path_and_title );

    uint32_t key = SiStripControlKey::key( path.fecCrate_,
					   path.fecSlot_,
					   path.fecRing_,
					   path.ccuAddr_,
					   path.ccuChan_,
					   title.channel_ );

    cout << "[PedestalsHistograms::book]" 
	 << " Path and title: " << path_and_title
	 << " MonitorElement ptr: 0x" << hex << me << dec
	 << " Control key: 0x" << hex << key << dec
	 << " FecCrate/FecRing/FecSlot/CcuAddr/CcuChan/channel: "
	 << path.fecCrate_ << "/"
	 << path.fecSlot_ << "/"
	 << path.fecRing_ << "/"
	 << path.ccuAddr_ << "/"
	 << path.ccuChan_ << "/"
	 << title.channel_
	 << endl;
    
    // Depending on "extra info", initialise different client histos
    if ( title.extraInfo_.find( sistrip::pedsAndRawNoise_ ) != string::npos ) { 
      cout << "[PedestalsHistograms::book]" 
	   << " Initialising PedsAndRawNoise histo..." << endl;
      initHistoSet( title, mePeds_[key], me ); 
    } else if ( title.extraInfo_.find( sistrip::residualsAndNoise_ ) != string::npos ) { 
      cout << "[PedestalsHistograms::book]" 
	   << " Initialising ResidualsAndNoise histo..." << endl;
      initHistoSet( title, meNoise_[key], me );
    } else if ( title.extraInfo_.find( sistrip::commonMode_ ) != string::npos ) {
      if ( title.channel_%2 == 0 ) { 
	cout << "[PedestalsHistograms::book]" 
	     << " Initialising CommonModeApv0 histo..." << endl;
	initHistoSet( title, meCommonModeApv0_[key], me ); 
      } else { 
	cout << "[PedestalsHistograms::book]" 
	     << " Initialising CommonModeApv1 histo..." << endl;
	initHistoSet( title, meCommonModeApv1_[key], me ); 
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
  cout << "[PedestalsHistograms::book]" << endl;
  
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

// -----------------------------------------------------------------------------
/** */
void PedestalsHistograms::histoAnalysis() {
  cout << "[PedestalsHistograms::histoAnalysis]" << endl;
}

// -----------------------------------------------------------------------------
/** */
void PedestalsHistograms::createSummaryHistos() {
  cout << "[PedestalsHistograms::createSummaryHistos]" << endl;
}

// -----------------------------------------------------------------------------
/** */
void PedestalsHistograms::createTrackerMap() {
  cout << "[PedestalsHistograms::createTrackerMap]" << endl;
}

// -----------------------------------------------------------------------------
/** */
void PedestalsHistograms::uploadToConfigDb() {
  cout << "[PedestalsHistograms::uploadToConfigDb]" << endl;
}
