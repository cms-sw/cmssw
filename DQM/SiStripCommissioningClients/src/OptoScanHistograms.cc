#include "DQM/SiStripCommissioningClients/interface/OptoScanHistograms.h"
#include "DQM/SiStripCommon/interface/SummaryGenerator.h"
#include <iostream>
#include <sstream>
#include <iomanip>

using namespace std;

// -----------------------------------------------------------------------------
/** */
OptoScanHistograms::OptoScanHistograms( MonitorUserInterface* mui ) 
  : CommissioningHistograms(mui),
    factory_( new Factory )
{
  cout << "[" << __PRETTY_FUNCTION__ << "]"
       << " Created object for OPTO (bias and gain) SCAN histograms" << endl;
}

// -----------------------------------------------------------------------------
/** */
OptoScanHistograms::~OptoScanHistograms() {
  cout << "[" << __PRETTY_FUNCTION__ << "]" << endl;
}

// -----------------------------------------------------------------------------	 
/** */	 
void OptoScanHistograms::histoAnalysis() {
  
  uint32_t cntr = 0;
  uint32_t nchans = collations().size();
  
  // Iterate through map containing vectors of profile histograms
  CollationsMap::const_iterator iter = collations().begin();
  for ( ; iter != collations().end(); iter++ ) {
    
    // Check vector of histos is not empty (should be 8 histos)
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
    OptoScanAnalysis::TProfiles profs;
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
      if ( title.task_ != sistrip::OPTO_SCAN ) {
	cerr << "[" << __PRETTY_FUNCTION__ << "]"
	     << " Unexpected commissioning task!"
	     << "(" << SiStripHistoNamingScheme::task( title.task_ ) << ")"
	     << endl;
      }

      // Extract gain setting and digital high/low info
      uint16_t gain = sistrip::invalid_; 
      if ( title.extraInfo_.find(sistrip::gain_) != string::npos ) {
	stringstream ss;
	ss << title.extraInfo_.substr( title.extraInfo_.find(sistrip::gain_) + sistrip::gain_.size(), 1 );
	ss >> dec >> gain;
      }
      uint16_t digital = sistrip::invalid_; 
      if ( title.extraInfo_.find(sistrip::digital_) != string::npos ) {
	stringstream ss;
	ss << title.extraInfo_.substr( title.extraInfo_.find(sistrip::digital_) + sistrip::digital_.size(), 1 );
	ss >> dec >> digital;
      }

      // Store histo pointers
      if ( digital == 0 ) { 
	if      ( gain == 0 ) { profs.g0d0_ = prof; }
	else if ( gain == 1 ) { profs.g1d0_ = prof; }
	else if ( gain == 2 ) { profs.g2d0_ = prof; }
	else if ( gain == 3 ) { profs.g3d0_ = prof; }
	else {
	  cerr << "[" << __PRETTY_FUNCTION__ << "]"
	       << " Unexpected gain setting! (" << gain << ")" << endl;
	}
      } else if ( digital == 1 ) { 
	if      ( gain == 0 ) { profs.g0d1_ = prof; }
	else if ( gain == 1 ) { profs.g1d1_ = prof; }
	else if ( gain == 2 ) { profs.g2d1_ = prof; }
	else if ( gain == 3 ) { profs.g3d1_ = prof; }
	else {
	  cerr << "[" << __PRETTY_FUNCTION__ << "]"
	       << " Unexpected gain setting! (" << gain << ")" << endl;
	}
      } else {
	cerr << "[" << __PRETTY_FUNCTION__ << "]"
	     << " Unexpected ditigal setting! (" << digital << ")" << endl;
      }
      
    }
     
    // Perform histo analysis
    OptoScanAnalysis::Monitorables mons;
    OptoScanAnalysis::analysis( profs, mons );
    
    // Store delay in map
    data_[iter->first] = mons; 
    
  }

}

// -----------------------------------------------------------------------------
/** */
void OptoScanHistograms::createSummaryHisto( const sistrip::SummaryHisto& histo, 
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

