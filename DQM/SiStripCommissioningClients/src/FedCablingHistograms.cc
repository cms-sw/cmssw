#include "DQM/SiStripCommissioningClients/interface/FedCablingHistograms.h"
#include "DQM/SiStripCommon/interface/SummaryGenerator.h"
#include <iostream>
#include <sstream>
#include <iomanip>

using namespace std;

// -----------------------------------------------------------------------------
/** */
FedCablingHistograms::FedCablingHistograms( MonitorUserInterface* mui ) 
  : CommissioningHistograms(mui),
    factory_( new Factory )
{
  cout << "[FedCablingHistograms::FedCablingHistograms]"
       << " Created object for FED CABLING histograms" << endl;
}

// -----------------------------------------------------------------------------
/** */
FedCablingHistograms::~FedCablingHistograms() {
  cout << "[FedCablingHistograms::~FedCablingHistograms]" << endl;
}

// -----------------------------------------------------------------------------	 
/** */	 
void FedCablingHistograms::histoAnalysis() {
  cout << "[" << __PRETTY_FUNCTION__ << "]" << endl;

  uint32_t cntr = 0;
  uint32_t nchans = collations().size();

  // Iterate through map containing vectors of profile histograms
  CollationsMap::const_iterator iter = collations().begin();
  for ( ; iter != collations().end(); iter++ ) {
    
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
    FedCablingAnalysis::TProfiles profs;
    Collations::const_iterator ihis = iter->second.begin(); 
    for ( ; ihis != iter->second.end(); ihis++ ) {
      
      // Retrieve pointer to profile histo 
      TProfile* prof = ExtractTObject<TProfile>().extract( mui()->get(iter->second[0]) );
      if ( !prof ) { 
	cerr << "[" << __PRETTY_FUNCTION__ << "]"
	     << " NULL pointer to MonitorElement!" << endl; 
	continue; 
      }
      
      // Retrieve control key
      static SiStripHistoNamingScheme::HistoTitle title;
      title = SiStripHistoNamingScheme::histoTitle( prof->GetName() );
    
      // Some checks
      if ( title.task_ != sistrip::APV_TIMING ) {
	cerr << "[" << __PRETTY_FUNCTION__ << "]"
	     << " Unexpected commissioning task!"
	     << "(" << SiStripHistoNamingScheme::task( title.task_ ) << ")"
	     << endl;
      }
      
      // Extract FED id and channel histos
      if ( title.extraInfo_.find(sistrip::fedId_) != string::npos ) {
	profs.fedId_ = prof;
      } else if ( title.extraInfo_.find(sistrip::fedChannel_) != string::npos ) {
	profs.fedCh_ = prof;
      } else { 
	cerr << "[" << __PRETTY_FUNCTION__ << "]"
	     << " Unexpected 'extra info': " << title.extraInfo_ << endl;
      }

    }
    
    // Perform histo analysis
    FedCablingAnalysis::Monitorables mons;
    FedCablingAnalysis::analysis( profs, mons );
    
    stringstream ss;
    mons.print( ss ); 
    cout << ss.str() << endl;
    
    // Store delay in map
    data_[iter->first] = mons; 
    
  }
  
}

// -----------------------------------------------------------------------------
/** */
void FedCablingHistograms::createSummaryHisto( const sistrip::SummaryHisto& histo, 
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



