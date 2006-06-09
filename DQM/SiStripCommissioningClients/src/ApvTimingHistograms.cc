#include "DQM/SiStripCommissioningClients/interface/ApvTimingHistograms.h"
#include <iostream>

using namespace std;

// -----------------------------------------------------------------------------
/** */
ApvTimingHistograms::ApvTimingHistograms( MonitorUserInterface* mui ) 
  : CommissioningHistograms(mui),
    timing_()
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
#include "DataFormats/SiStripDetId/interface/SiStripControlKey.h"
#include "DQM/SiStripCommon/interface/SiStripHistoNamingScheme.h"
#include "DQM/SiStripCommissioningAnalysis/interface/ApvTimingAnalysis.h"
#include "DQM/SiStripCommissioningSummary/interface/CommissioningSummary.h"
#include "TProfile.h"
#include "TH1F.h"
/** */
void ApvTimingHistograms::createSummaryHistos() {
  cout << "[ApvTimingHistograms::createSummaryHistos]" << endl;

  // Create summary histo and iterate through profile histograms
  CommissioningSummary* summary = 0;
  map< uint32_t, HistoSet >::iterator ihis = timing_.begin();
  for ( ; ihis != timing_.end(); ihis++ ) {

    // Create summary histogram object
    if ( !summary ) { 
      summary = new CommissioningSummary( SiStripHistoNamingScheme::task( ihis->second.title_.task_ ),
					  ihis->second.title_.granularity_ ); }
    
    // Extract profile histo from map
    vector<const TProfile*> histos;
    TProfile* prof = ExtractTObject<TProfile>()( ihis->second.profile_ );
    if ( !prof ) { continue; }
    histos.push_back( prof );

    // Define monitorables container
    vector<uint16_t> monitorables;

    // Perform histo analysis and calc delay [ns]
    ApvTimingAnalysis anal;
    anal.analysis( histos, monitorables );
    uint32_t delay_ns = 24*monitorables[0] + monitorables[1];

    // Retrieve title and update summary histo
    CommissioningSummary::ReadoutId readout( ihis->first, ihis->second.title_.channel_ ); 
    if ( summary ) { summary->update( readout, delay_ns ); }
    
  }
  
  // Generate summary histograms 
  TH1F* control_his;
  TH1F* summary_his;
  if ( summary ) {
    control_his = summary->controlSummary( sistrip::controlView_ );
    summary_his = summary->summary( sistrip::controlView_ );
    delete summary;
  }
  
}


