#include "DQM/SiStripCommissioningClients/interface/ApvTimingHistograms.h"
#include "DQM/SiStripCommissioningAnalysis/interface/ApvTimingAnalysis.h"
#include <iostream>

using namespace std;

// -----------------------------------------------------------------------------
/** */
ApvTimingHistograms::ApvTimingHistograms( MonitorUserInterface* mui ) 
  : CommissioningHistograms(mui),
    timing_(),
    delays_()
{
  cout << "[ApvTimingHistograms::ApvTimingHistograms]"
       << " Created object for APV TIMING histograms" << endl;
  task( sistrip::APV_TIMING ); 
}

// -----------------------------------------------------------------------------
/** */
ApvTimingHistograms::~ApvTimingHistograms() {
  cout << "[ApvTimingHistograms::~ApvTimingHistograms]" << endl;
}

// -----------------------------------------------------------------------------
/** */
void ApvTimingHistograms::book( const vector<string>& me_list ) {
  
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
void ApvTimingHistograms::update() {
  map< uint32_t, HistoSet >::iterator ihis = timing_.begin();
  for ( ; ihis != timing_.end(); ihis++ ) {
    if ( ihis->second.profile_ ) { updateHistoSet( ihis->second ); }
  }
}

// -----------------------------------------------------------------------------
/** */
void ApvTimingHistograms::histoAnalysis() {
  cout << "[ApvTimingHistograms::histoAnalysis]" << endl;
  
  // Iterate through profile histograms in order to to fill delay map
  delays_.clear();
  map< uint32_t, HistoSet >::iterator ihis = timing_.begin();
  for ( ; ihis != timing_.end(); ihis++ ) {
    
    // Extract profile histo from map
    vector<const TProfile*> histos;
    TProfile* prof = ExtractTObject<TProfile>()( ihis->second.profile_ );
    if ( !prof ) { continue; }
    histos.push_back( prof );
    
    // Perform histo analysis and calc delay [ns]
    vector<uint16_t> monitorables;
    ApvTimingAnalysis anal;
    anal.analysis( histos, monitorables );
    uint32_t delay_ns = 24*monitorables[0] + monitorables[1];
    uint32_t key = ihis->first;
    
    // Store delay in map
    if ( delays_.find( key ) != delays_.end() ) { 
      cerr << "[ApvTimingHistograms::histoAnalysis]"
	   << " Monitorable already exists!" << endl;
    } else { delays_[key] = delay_ns; }
    
  }
  
}

// -----------------------------------------------------------------------------
/** */
void ApvTimingHistograms::createSummaryHistos() {
  cout << "[ApvTimingHistograms::createSummaryHistos]" << endl;

  // Extract monitorables from histos
  histoAnalysis();
  
  // Retrive histo title and create summary histogram object
  CommissioningSummary* summary = 0;
  if ( !timing_.empty() ) {
    string title = SiStripHistoNamingScheme::task( timing_.begin()->second.title_.task_ );
    summary = new CommissioningSummary( title, timing_.begin()->second.title_.granularity_ ); 
  }
  if ( !summary ) { return; }
  
  // Iterate through monitorables, retrieve key and update summary histo
  map< uint32_t, uint32_t >::iterator imon = delays_.begin();
  for ( ; imon != delays_.end(); imon++ ) {
    const SiStripControlKey::ControlPath& path = SiStripControlKey::path( imon->first );
    CommissioningSummary::ReadoutId readout( imon->first, path.channel_ ); 
    if ( summary ) { summary->update( readout, imon->second ); }
  }
  
  // Generate summary histograms (based on control view)
  TH1F* control_his;
  TH1F* summary_his;
  if ( summary ) {
    control_his = summary->controlSummary( sistrip::controlView_ );
    summary_his = summary->summary( sistrip::controlView_ );
    delete summary;
  }
  
  //@@ do something with summary histos?...
  
}

