#include "DQM/SiStripCommissioningClients/interface/ApvTimingHistograms.h"

using namespace std;

// -----------------------------------------------------------------------------
/** */
ApvTimingHistograms::ApvTimingHistograms( MonitorUserInterface* mui ) 
  : CommissioningHistograms(mui),
    factory_( new Factory )
{
  cout << "[ApvTimingHistograms::ApvTimingHistograms]"
       << " Created object for APV TIMING histograms" << endl;
}

// -----------------------------------------------------------------------------
/** */
ApvTimingHistograms::~ApvTimingHistograms() {
  cout << "[ApvTimingHistograms::~ApvTimingHistograms]" << endl;
}

// -----------------------------------------------------------------------------	 
/** */	 
void ApvTimingHistograms::histoAnalysis() {
  
  uint32_t cntr = 0;
  uint32_t nhis = collations().size();
  
  // Iterate through profile histograms in order to to fill delay map 
  std::vector<std::string>::const_iterator ihis = collations().begin();
  for ( ; ihis != collations().end(); ihis++ ) {
    cntr++;
    cout << "[" << __PRETTY_FUNCTION__ << "]"
	 << " Analyzing " << cntr << " of " << nhis << " histograms..." << endl;

    // Extract profile histo from map	 
    MonitorElement* me = mui()->get( *ihis );
    TProfile* prof = ExtractTObject<TProfile>().extract( me );
    if ( !prof ) { 
      cerr << "[" << __PRETTY_FUNCTION__ << "]"
	   << " NULL pointer to MonitorElement!" << endl; 
      continue; 
    }
    
    // Perform histo analysis
    ApvTimingAnalysis::Monitorables mons;
    ApvTimingAnalysis::analysis( prof, mons );
    
    // Retrieve control key
    SiStripHistoNamingScheme::HistoTitle title = SiStripHistoNamingScheme::histoTitle( prof->GetName() );
    if ( title.task_ != sistrip::APV_TIMING ) {
      cerr << "[" << __PRETTY_FUNCTION__ << "]"
	   << " Unexpected commissioning task!"
	   << "(" << SiStripHistoNamingScheme::task( title.task_ ) << ")"
	   << endl;
    }
    
    // Store delay in map
    if ( data_.find( title.keyValue_ ) == data_.end() ) {
      data_[title.keyValue_] = mons; 
    } else { 
      if ( mons.delay_ != data_[title.keyValue_].delay_ ) {
	stringstream ss;
	ss << "[" << __PRETTY_FUNCTION__ << "]"
	   << " Monitorable data already exist!" << "\n";
	ss << "Existing Monitorable data:" << "\n";
	data_[title.keyValue_].print( ss );
	ss << "New Monitorable data:" << "\n";
	mons.print( ss );
	cerr << ss.str();
      }
    }
    
  }
  
}

// -----------------------------------------------------------------------------
/** */
void ApvTimingHistograms::createSummaryHisto( const sistrip::SummaryHisto& histo, 
					      const sistrip::SummaryType& type, 
					      const string& directory ) {
  cout << "[" << __PRETTY_FUNCTION__ <<"]" << endl;
  
  // Check view 
  sistrip::View view = SiStripHistoNamingScheme::view(directory);
  if ( view == sistrip::UNKNOWN_VIEW ) { return; }

  // Change to appropriate directory
  mui()->setCurrentFolder( directory );
  
  // Create MonitorElement (if it doesn't already exist) and update contents
  string name = factory_->name( histo, type, view, directory );
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

