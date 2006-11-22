#include "DQM/SiStripCommissioningClients/interface/CommissioningHistograms.h"
#include "DQM/SiStripCommissioningSummary/interface/SummaryGenerator.h"
#include "DataFormats/SiStripCommon/interface/SiStripConstants.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <sstream>

using namespace std;
using namespace sistrip;

// -----------------------------------------------------------------------------
/** */
CommissioningHistograms::CommissioningHistograms( MonitorUserInterface* mui,
						  const sistrip::Task& task ) 
  : mui_(mui),
    collations_(),
    action_(sistrip::NO_ACTION),
    task_(task)
{
  cout << endl // LogTrace(mlDqmClient_)
       << "[CommissioningHistograms::" << __func__ << "]"
       << " Constructing object...";
  collations_.clear();
}

// -----------------------------------------------------------------------------
/** */
CommissioningHistograms::~CommissioningHistograms() {
  cout << endl // LogTrace(mlDqmClient_)
       << "[CommissioningHistograms::" << __func__ << "]"
       << " Destructing object...";
}

// -----------------------------------------------------------------------------
/** */
void CommissioningHistograms::createCollations( const vector<string>& contents ) {
  cout << endl // LogTrace(mlDqmClient_)
       << "[CommissioningHistograms::" << __func__ << "]"
       << " Creating CollateMonitorElements...";

  if ( contents.empty() ) { return; }
  
  vector<string>::const_iterator idir;
  for ( idir = contents.begin(); idir != contents.end(); idir++ ) {
    
    // Extract directory paths
    string collector_dir = idir->substr( 0, idir->find(":") );
    SiStripFecKey::Path path = SiStripHistoNamingScheme::controlPath( collector_dir );
    string client_dir = SiStripHistoNamingScheme::controlPath( path );
    // client_dir = "Client" + client_dir;
    
    if ( path.fecCrate_ == sistrip::invalid_ ||
	 path.fecSlot_ == sistrip::invalid_ ||
	 path.fecRing_ == sistrip::invalid_ ||
	 path.ccuAddr_ == sistrip::invalid_ ||
	 path.ccuChan_ == sistrip::invalid_ ) { continue; } 

    // Retrieve MonitorElements from pwd directory
    mui()->setCurrentFolder( collector_dir );
    vector<string> me_list = mui()->getMEs();

    // Iterate through MEs and create CMEs
    CollateMonitorElement* cme = 0;
    vector<string>::iterator ime = me_list.begin(); 
    for ( ; ime != me_list.end(); ime++ ) {
      
      // Retrieve pointer to monitor element
      MonitorElement* me = mui()->get( mui()->pwd()+"/"+(*ime) ); // path + name
      TProfile* prof = ExtractTObject<TProfile>().extract( me );
      TH1F* his = ExtractTObject<TH1F>().extract( me );
      if ( prof ) { prof->SetErrorOption("s"); } //@@ is this necessary? (until bug fix applied to dqm)...
      
      // Retrieve granularity from histogram title (necessary?)
      HistoTitle title = SiStripHistoNamingScheme::histoTitle( *ime );
      cout << title << endl;
      uint16_t channel = sistrip::invalid_;
      if ( title.granularity_ == sistrip::APV ) {
	channel = (title.channel_-32)/2;
      } else if ( title.granularity_ == sistrip::LLD_CHAN ) {
	channel = title.channel_;
      } else {
	cerr << endl // edm::LogWarning(mlDqmClient_)
	     << "[CommissioningHistograms::" << __func__ << "]"
	     << " Unexpected histogram granularity: "
	     << title.granularity_;
      }

      // Build FEC key and fill FED-FEC map
      uint32_t key = SiStripFecKey::key( sistrip::invalid_, //@@ WARNING: only good for one partition only!!!
					 path.fecSlot_,
					 path.fecRing_,
					 path.ccuAddr_,
					 path.ccuChan_,
					 channel );
      mapping_[title.keyValue_] = key;
      
      //cout << "Checking for CME..." << endl;
      // Create collation MEs
      CollationsMap::iterator iter = collations_.find( key );
      if ( iter == collations_.end() ) {
	cout << "Found new channel (control key)" << endl;
	if ( prof )     { cme = mui()->collateProf( *ime, *ime, client_dir ); cout << "GOT HERE!!!" << endl; }
	else if ( his ) { cme = mui()->collate1D( *ime, *ime, client_dir ); }
	else { 
	  cme = 0; 
	  cerr << endl // edm::LogWarning(mlDqmClient_)
	       << "[CommissioningHistograms::" << __func__ << "]"
	       << " NULL pointers to histos!"; 
	}
	if ( cme ) {

	  cout << "Booked new CME and adding MEs" << endl;
	  cout << " client: ptr/dir/name: " << cme << " " << client_dir << " " << (*ime) << endl;
	  cout << " collector: ptr/dir/name: " << cme << " " << collector_dir << " " << (*ime) << endl;
	  cout << " pwd: ptr/dir/name: " << cme << " " << mui()->pwd() << " " << (*ime) << endl;
	  //cout << " coll: " << collector_dir << " cli: " << client_dir << endl;

	  MonitorElement* cme1 = mui()->get( client_dir + "/" + (*ime) );
	  cout << " CME: ptr/dir/name: " << cme1 << " " << client_dir << " " << (*ime) << endl;

	  mui()->add( cme, collector_dir+"/"+(*ime) ); // note search pattern
	  //mui()->add( cme, "*"+client_dir+(*ime) ); // note search pattern
	  cout << "ME added to new CME" << endl;
	  if ( collations_[key].capacity() != 10 ) { collations_[key].reserve(10); }
	  collations_[key].push_back( client_dir+(*ime) ); // store "path + name"
	  cout << "New CME stored in map" << endl;
	}
      } else {
	//cout << "Found some existing CMEs for this channel" << endl;
	if ( find( iter->second.begin(), iter->second.end(), client_dir+(*ime) ) == iter->second.end() ) {
	  cout << "Did not find CME in existing channel" << endl;
	  if ( prof )     { cme = mui()->collateProf( *ime, *ime, client_dir ); }
	  else if ( his ) { cme = mui()->collate1D( *ime, *ime, client_dir ); }
	  else { 
	    cme = 0; 
	    cerr << endl // edm::LogWarning(mlDqmClient_)
		 << "[CommissioningHistograms::" << __func__ << "]"
		 << " NULL pointers to histos!"; 
	  }
	  if ( cme ) {
	    cout << "Booked new CME in existing channel and adding MEs" << endl;
	    
	    mui()->add( cme, collector_dir+"/"+(*ime) ); // note search pattern
	    //mui()->add( cme, "*"+client_dir+(*ime) ); // note search pattern
	    cout << "Added ME to CME in existing channel" << endl;
	    if ( collations_[key].capacity() != 10 ) { collations_[key].reserve(10); }
	    collations_[key].push_back( client_dir+(*ime) ); // store "path + name"
	    cout << "CME in existing channel stored in map" << endl;
	  }
	} else {
	  //cout << "CME already exists" << endl;
	}
      }
      //cout << "End of checking" << endl;
    }
  }
  
}

// -----------------------------------------------------------------------------
/** */
void CommissioningHistograms::histoAnalysis( bool debug ) {
  cout << endl // LogTrace(mlDqmClient_)
       << "[CommissioningHistograms::" << __func__ << "]"
       << " (Derived) implementation to come...";
}

// -----------------------------------------------------------------------------
/** */
void CommissioningHistograms::createSummaryHisto( const sistrip::SummaryHisto& histo, 
						  const sistrip::SummaryType& type, 
						  const string& directory,
						  const sistrip::Granularity& gran ) {
  cout << endl // LogTrace(mlDqmClient_)
       << "[CommissioningHistograms::" << __func__ << "]"
       << " (Derived) implementation to come...";
}

// -----------------------------------------------------------------------------
/** */
void CommissioningHistograms::uploadToConfigDb() {
  cout << endl // LogTrace(mlDqmClient_)
       << "[CommissioningHistograms::" << __func__ << "]"
       << " (Derived) implementation to come..."; 
}

// -----------------------------------------------------------------------------
/** Wraps other createSummaryHisto() method. */
void CommissioningHistograms::createSummaryHisto( pair<sistrip::SummaryHisto,
						  sistrip::SummaryType> summ0, 
						  pair<string,
						  sistrip::Granularity> summ1 ) {
  createSummaryHisto( summ0.first, summ0.second, summ1.first, summ1.second );
}

// -----------------------------------------------------------------------------
// 
TH1* CommissioningHistograms::histogram( const sistrip::SummaryHisto& histo, 
					 const sistrip::SummaryType& type, 
					 const sistrip::View& view,
					 const string& directory,
					 const uint32_t& xbins ) {
  
  string name = SummaryGenerator::name( task_, histo, type, view, directory );
  mui()->setCurrentFolder( directory );
  MonitorElement* me = mui()->get( mui()->pwd() + "/" + name );
  if ( !me ) { 
    if ( type == sistrip::SUMMARY_DISTR ) { 
      me = mui()->getBEInterface()->book1D( name, name, xbins, 0., static_cast<float>(xbins) ); 
    } else if ( type == sistrip::SUMMARY_1D ) { 
      me = mui()->getBEInterface()->book1D( name, name, xbins, 0., static_cast<float>(xbins) ); 
    } else if ( type == sistrip::SUMMARY_2D ) { 
      me = mui()->getBEInterface()->book2D( name, name, xbins, 0., static_cast<float>(xbins), 1025, 0., 1025 ); 
    } else if ( type == sistrip::SUMMARY_PROF ) { 
      me = mui()->getBEInterface()->bookProfile( name, name, xbins, 0., static_cast<float>(xbins), 1025, 0., 1025 ); 
    } else { me = 0; 
    }
  }
  TH1F* summary = ExtractTObject<TH1F>().extract( me ); 
  return summary;
}

