#include "DQM/SiStripCommissioningClients/interface/VpspScanHistograms.h"
#include "DQM/SiStripCommon/interface/SummaryGenerator.h"
#include <iostream>
#include <sstream>
#include <iomanip>

using namespace std;

// -----------------------------------------------------------------------------
/** */
VpspScanHistograms::VpspScanHistograms( MonitorUserInterface* mui ) 
  : CommissioningHistograms(mui),
    factory_( new Factory )
{
  cout << "[" << __PRETTY_FUNCTION__ << "]"
       << " Created object for VPSP SCAN histograms" << endl;
}

// -----------------------------------------------------------------------------
/** */
VpspScanHistograms::~VpspScanHistograms() {
  cout << "[" << __PRETTY_FUNCTION__ << "]" << endl;
}

// -----------------------------------------------------------------------------	 
/** */	 
void VpspScanHistograms::histoAnalysis() {
  
  uint32_t cntr = 0;
  uint32_t nchans = collations().size();
  
  // Iterate through map containing vectors of profile histograms
  CollationsMap::const_iterator iter = collations().begin();
  for ( ; iter != collations().end(); iter++ ) {
    
    // Check vector of histos is not empty (should be 2 histos)
    if ( iter->second.empty() ) {
      cerr << "[" << __PRETTY_FUNCTION__ << "]"
	   << " Zero collation histograms found!" << endl;
      continue;
    }
    cntr++;
    cout << "[" << __PRETTY_FUNCTION__ << "]"
	 << " Analyzing histograms from " << cntr
	 << " of " << nchans << " FED channels..." << endl;
    
    // Iterate through vector of histos
    VpspScanAnalysis::TProfiles profs;
    Collations::const_iterator ihis = iter->second.begin(); 
    for ( ; ihis != iter->second.end(); ihis++ ) {
      
      // Retrieve pointer to profile histo 
      TProfile* prof = ExtractTObject<TProfile>().extract( mui()->get( *ihis ) );
      if ( !prof ) { 
	cerr << "[" << __PRETTY_FUNCTION__ << "]"
	     << " NULL pointer to MonitorElement!" << endl; 
	continue; 
      }
      
      // Retrieve control key
      static SiStripHistoNamingScheme::HistoTitle title;
      title = SiStripHistoNamingScheme::histoTitle( prof->GetName() );
      
      // Some checks
      if ( title.task_ != sistrip::VPSP_SCAN ) {
	cerr << "[" << __PRETTY_FUNCTION__ << "]"
	     << " Unexpected commissioning task!"
	     << "(" << SiStripHistoNamingScheme::task( title.task_ ) << ")"
	     << endl;
      }
      
      // Store histo pointers
      if ( title.channel_ >= 32 &&
	   title.channel_ <= 37 ) { 
	if ( title.channel_%2 == 0 ) { 
	  profs.vpsp0_ = prof;
	} else if ( title.channel_%2 == 1 ) { 
	  profs.vpsp1_ = prof;
	} else {
	  cerr << "[" << __PRETTY_FUNCTION__ << "]"
	     << " Unexpected channel setting! (" << title.channel_ << ")" << endl;
	}
      }
      
    }
     
    // Perform histo analysis
    VpspScanAnalysis::Monitorables mons;
    VpspScanAnalysis::analysis( profs, mons );
    
    // Store delay in map
    data_[iter->first] = mons; 
    
  }

}

// -----------------------------------------------------------------------------
/** */
void VpspScanHistograms::createSummaryHisto( const sistrip::SummaryHisto& histo, 
					     const sistrip::SummaryType& type, 
					     const string& directory ) {
  cout << "[" << __PRETTY_FUNCTION__ <<"]" << endl;
  
  // Check view 
  sistrip::View view = SiStripHistoNamingScheme::view(directory);
  if ( view == sistrip::UNKNOWN_VIEW ) { return; }

  // Change to appropriate directory
  mui()->setCurrentFolder( directory );
  
  // Create MonitorElement (if it doesn't already exist) and update contents
  string name = SummaryGenerator::name( histo, type, view, directory );
  MonitorElement* me = mui()->get( mui()->pwd() + "/" + name );
  if ( !me ) { me = mui()->getBEInterface()->book1D( name, "", 0, 0., 0. ); }
  TH1F* summary = ExtractTObject<TH1F>().extract( me ); 
  factory_->generate( histo, 
		      type, 
		      view, 
		      directory, 
		      data_,
		      *summary );
  
}

