#include "DQM/SiStripCommissioningClients/interface/CommissioningHistograms.h"
#include "CondFormats/SiStripObjects/interface/CommissioningAnalysis.h"
#include "DataFormats/SiStripCommon/interface/SiStripFecKey.h"
#include "DataFormats/SiStripCommon/interface/SiStripFedKey.h"
#include "DataFormats/SiStripCommon/interface/SiStripDetKey.h"
#include "DQM/SiStripCommissioningSummary/interface/SummaryGenerator.h"
#include "DataFormats/SiStripCommon/interface/SiStripConstants.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <iomanip>

using namespace std;
using namespace sistrip;

// -----------------------------------------------------------------------------
/** */
CommissioningHistograms::CommissioningHistograms( const edm::ParameterSet& pset,
                                                  DQMOldReceiver* mui,
						  const sistrip::RunType& task ) 
  : factory_(0),
    task_(task),
    mui_(mui),
    bei_(0),
    data_(),
    histos_(),
    pset_(pset)
{
  LogTrace(mlDqmClient_)
    << "[" << __PRETTY_FUNCTION__ << "]"
    << " Constructing object...";

  // DQMOldReceiver
  if ( mui_ ) { bei_ = mui_->getBEInterface(); }
  else {
    edm::LogError(mlDqmClient_)
      << "[CommissioningHistograms::" << __func__ << "]"
      << " NULL pointer to DQMOldReceiver!";
  }
  
  // DQMStore
  if ( !bei_ ) {
    edm::LogError(mlDqmClient_)
      << "[CommissioningHistograms::" << __func__ << "]"
      << " NULL pointer to DQMStore!";
  }
  
  clearHistosMap();
}

// -----------------------------------------------------------------------------
/** */
CommissioningHistograms::CommissioningHistograms( const edm::ParameterSet& pset,
                                                  DQMStore* bei,
						  const sistrip::RunType& task ) 
  : factory_(0),
    task_(task),
    mui_(0),
    bei_(bei),
    data_(),
    histos_(),
    pset_(pset)
{
  LogTrace(mlDqmClient_)
    << "[" << __PRETTY_FUNCTION__ << "]"
    << " Constructing object...";

  // DQMStore
  if ( !bei_ ) {
    edm::LogError(mlDqmClient_)
      << "[CommissioningHistograms::" << __func__ << "]"
      << " NULL pointer to DQMStore!";
  }
  
  clearHistosMap();
}

// -----------------------------------------------------------------------------
/** */
CommissioningHistograms::CommissioningHistograms() 
  : factory_(0),
    task_(sistrip::UNDEFINED_RUN_TYPE),
    mui_(0),
    bei_(0),
    data_(),
    histos_()
{
  LogTrace(mlDqmClient_)
    << "[" << __PRETTY_FUNCTION__ << "]"
    << " Constructing object...";
}

// -----------------------------------------------------------------------------
/** */
CommissioningHistograms::~CommissioningHistograms() {
  LogTrace(mlDqmClient_)
    << "[" << __PRETTY_FUNCTION__ << "]"
    << " Destructing object...";
  clearHistosMap();
  //@@ do not delete MUI or BEI ptrs!
}

// -----------------------------------------------------------------------------
/** */
void CommissioningHistograms::Histo::print( std::stringstream& ss ) const {
  ss << " [Histo::" << __func__ << "]" << std::endl
     << " Histogram title   : " << title_ << std::endl
     << " MonitorElement*   : 0x" 
     << std::hex
     << std::setw(8) << std::setfill('0') << me_ << std::endl
     << std::dec
     << " CollateME*        : 0x" 
     << std::hex
     << std::setw(8) << std::setfill('0') << cme_ << std::endl
     << std::dec;
}

// -----------------------------------------------------------------------------
//
uint32_t CommissioningHistograms::runNumber( DQMStore* const bei,
					     const std::vector<std::string>& contents ) {
  
  // Check if histograms present
  if ( contents.empty() ) { 
    edm::LogError(mlDqmClient_)
      << "[CommissioningHistograms::" << __func__ << "]"
      << " Found no histograms!";
    return 0; 
  }
  
  // Iterate through added contents
  std::vector<std::string>::const_iterator istr = contents.begin();
  while ( istr != contents.end() ) {
    
    // Extract source directory path 
    std::string source_dir = istr->substr( 0, istr->find(":") );
    
    // Generate corresponding client path (removing trailing "/")
    SiStripFecKey path( source_dir );
    std::string client_dir = path.path();
    std::string slash = client_dir.substr( client_dir.size()-1, 1 ); 
    if ( slash == sistrip::dir_ ) { client_dir = client_dir.substr( 0, client_dir.size()-1 ); }
    client_dir = std::string(sistrip::collate_) + sistrip::dir_ + client_dir;
    
    // Iterate though MonitorElements from source directory
    std::vector<MonitorElement*> me_list = bei->getContents( source_dir );
    std::vector<MonitorElement*>::iterator ime = me_list.begin(); 
    for ( ; ime != me_list.end(); ime++ ) {
      
      if ( !(*ime) ) {
	edm::LogError(mlDqmClient_)
	  << "[CommissioningHistograms::" << __func__ << "]"
	  << " NULL pointer to MonitorElement!";
	continue;
      }

      // Search for run type in string
      std::string title = (*ime)->getName();
      std::string::size_type pos = title.find( sistrip::runNumber_ );
      
      // Extract run number from string 
      if ( pos != std::string::npos ) { 
	std::string value = title.substr( pos+sizeof(sistrip::runNumber_) , std::string::npos ); 
	if ( !value.empty() ) { 
	  LogTrace(mlDqmClient_)
	    << "[CommissioningHistograms::" << __func__ << "]"
	    << " Found string \"" <<  title.substr(pos,std::string::npos)
	    << "\" with value \"" << value << "\"";
	  if ( !(bei->get(client_dir+"/"+title.substr(pos,std::string::npos))) ) { 
	    bei->setCurrentFolder(client_dir);
	    bei->bookString( title.substr(pos,std::string::npos), value ); 
	    LogTrace(mlDqmClient_)
	      << "[CommissioningHistograms::" << __func__ << "]"
	      << " Booked string \"" << title.substr(pos,std::string::npos)
	      << "\" in directory \"" << client_dir << "\"";
	  }
	  uint32_t run;
	  std::stringstream ss;
	  ss << value;
	  ss >> std::dec >> run;
	  return run; 
	}
      }

    }

    istr++;
    
  }
  return 0;
}

// -----------------------------------------------------------------------------
/** Extract run type string from "added contents". */
sistrip::RunType CommissioningHistograms::runType( DQMStore* const bei,
						   const std::vector<std::string>& contents ) {
  
  // Check if histograms present
  if ( contents.empty() ) { 
    edm::LogError(mlDqmClient_)
      << "[CommissioningHistograms::" << __func__ << "]"
      << " Found no histograms!";
    return sistrip::UNKNOWN_RUN_TYPE; 
  }
  
  // Iterate through added contents
  std::vector<std::string>::const_iterator istr = contents.begin();
  while ( istr != contents.end() ) {

    // Extract source directory path 
    std::string source_dir = istr->substr( 0, istr->find(":") );

    // Generate corresponding client path (removing trailing "/")
    SiStripFecKey path( source_dir );
    std::string client_dir = path.path();
    std::string slash = client_dir.substr( client_dir.size()-1, 1 ); 
    if ( slash == sistrip::dir_ ) { client_dir = client_dir.substr( 0, client_dir.size()-1 ); } 
    client_dir = std::string(sistrip::collate_) + sistrip::dir_ + client_dir;
    
    // Iterate though MonitorElements from source directory
    std::vector<MonitorElement*> me_list = bei->getContents( source_dir );

    if ( me_list.empty() ) {
      edm::LogError(mlDqmClient_)
	<< "[CommissioningHistograms::" << __func__ << "]"
	<< " No MonitorElements found in dir " << source_dir;
      return sistrip::UNKNOWN_RUN_TYPE;
    }

    std::vector<MonitorElement*>::iterator ime = me_list.begin(); 
    for ( ; ime != me_list.end(); ime++ ) {

      if ( !(*ime) ) {
	edm::LogError(mlDqmClient_)
	  << "[CommissioningHistograms::" << __func__ << "]"
	  << " NULL pointer to MonitorElement!";
	continue;
      }

      // Search for run type in string
      std::string title = (*ime)->getName();
      std::string::size_type pos = title.find( sistrip::taskId_ );

      // Extract commissioning task from string 
      if ( pos != std::string::npos ) { 
	std::string value = title.substr( pos+sizeof(sistrip::taskId_) , std::string::npos ); 
	if ( !value.empty() ) { 
	  LogTrace(mlDqmClient_)
	    << "[CommissioningHistograms::" << __func__ << "]"
	    << " Found string \"" <<  title.substr(pos,std::string::npos)
	    << "\" with value \"" << value << "\"";
	  if ( !(bei->get(client_dir+sistrip::dir_+title.substr(pos,std::string::npos))) ) { 
	    bei->setCurrentFolder(client_dir);
	    bei->bookString( title.substr(pos,std::string::npos), value ); 
	    LogTrace(mlDqmClient_)
	      << "[CommissioningHistograms::" << __func__ << "]"
	      << " Booked string \"" << title.substr(pos,std::string::npos)
	      << "\" in directory \"" << client_dir << "\"";
	  }
	  return SiStripEnumsAndStrings::runType( value ); 
	}
      }

    }

    istr++;
    
  }

  edm::LogError(mlDqmClient_)
    << "[CommissioningHistograms::" << __func__ << "]"
    << " Unable to extract RunType!";
  return sistrip::UNKNOWN_RUN_TYPE;

}

// -----------------------------------------------------------------------------
// Temporary fix: builds a list of histogram directories
void CommissioningHistograms::getContents( DQMStore* const bei,
					   std::vector<std::string>& contents ) {

#ifndef USING_NEW_COLLATE_METHODS
  
  LogTrace(mlDqmClient_) 
    << "[CommissioningHistograms::" << __func__ << "]"
    << " Building histogram list...";

  contents.clear();

  if ( !bei ) {
    edm::LogError(mlDqmClient_)
      << "[CommissioningHistograms::" << __func__ << "]"
      << " NULL pointer to DQMStore!";
  }

  bei->cd();
  std::vector<MonitorElement*> mons;
  mons = bei->getAllContents( bei->pwd() );
  std::vector<MonitorElement*>::const_iterator iter = mons.begin();
  for ( ; iter != mons.end(); iter++ ) {
    std::vector<std::string>::iterator istr = contents.begin();
    bool found = false;
    while ( !found && istr != contents.end()  ) {
      if ( std::string((*iter)->getPathname()+"/:") == *istr ) { found = true; }
      istr++;
    }
    if ( istr == contents.end() ) { 
      std::string temp = (*iter)->getPathname() + "/:"; // + (*iter)->getName();
      contents.push_back( temp ); 
    }
  }
  
  LogTrace(mlDqmClient_) 
    << "[CommissioningHistograms::" << __func__ << "]"
    << " Found " << contents.size() << " directories!";

  if ( contents.empty() ) { 
    edm::LogError(mlDqmClient_) 
      << "[CommissioningHistograms::" << __func__ << "]"
      << " No directories found when building list!";
  }

#endif
  
}

// -----------------------------------------------------------------------------
//
void CommissioningHistograms::copyCustomInformation( DQMStore* const bei,
						     const std::vector<std::string>& contents ) {
  
  // Check if histograms present
  if ( contents.empty() ) {
    edm::LogWarning(mlDqmClient_)
      << "[CommissioningHistograms::" << __func__  << "]"
      << " Found no histograms!";
    return;
  }
  
  // Iterate through added contents
  std::vector<std::string>::const_iterator istr = contents.begin();
  while ( istr != contents.end() ) {

    // Extract source directory path
    std::string source_dir = istr->substr( 0, istr->find(":") );

    // Generate corresponding client path (removing trailing "/")
    SiStripFecKey path( source_dir );
    std::string client_dir = path.path();
    std::string slash = client_dir.substr( client_dir.size()-1, 1 );
    if ( slash == sistrip::dir_ ) { client_dir = client_dir.substr( 0, client_dir.size()-1 ); }

    // Iterate though MonitorElements from source directory
    std::vector<MonitorElement*> me_list = bei->getContents( source_dir );
    std::vector<MonitorElement*>::iterator ime = me_list.begin();
    for ( ; ime != me_list.end(); ime++ ) {

      if ( !(*ime) ) {
	edm::LogWarning(mlDqmClient_)
	  << "[CommissioningHistograms::" << __func__ << "]"
	  << " NULL pointer to MonitorElement!";
	continue;
      }
      // Search for calchan, isha or vfs
      if((*ime)->kind()==MonitorElement::DQM_KIND_INT) {
        std::string title = (*ime)->getName();
        std::string::size_type pos = title.find("calchan");
        if( pos == std::string::npos ) pos = title.find("isha");
        if( pos == std::string::npos ) pos = title.find("vfs");
        if( pos != std::string::npos ) {
          int value = (*ime)->getIntValue();
          if ( value>=0 ) {
            edm::LogVerbatim(mlDqmClient_)
	      << "[CommissioningHistograms::" << __func__ << "]"
	      << " Found \"" << title.substr(pos,std::string::npos)
	      << "\" with value \"" << value << "\"";
	    if ( !(bei->get(client_dir+"/"+title.substr(pos,std::string::npos))) ) {
	      bei->setCurrentFolder(client_dir);
	      bei->bookInt( title.substr(pos,std::string::npos))->Fill(value);
	      edm::LogVerbatim(mlDqmClient_)
	        << "[CommissioningHistograms::" << __func__ << "]"
	        << " Booked \"" << title.substr(pos,std::string::npos)
	        << "\" in directory \"" << client_dir << "\"";
	    }
	  }
        }
      }
    }
    istr++;
  }
}

// -----------------------------------------------------------------------------

/** */
void CommissioningHistograms::extractHistograms( const std::vector<std::string>& contents ) {
  LogTrace(mlDqmClient_)
    << "[CommissioningHistograms::" << __func__ << "]"
    << " Extracting available histograms...";
  
  // Check pointer
  if ( !bei_ ) {
    edm::LogError(mlDqmClient_)
      << "[CommissioningHistograms::" << __func__ << "]"
      << " NULL pointer to DQMStore!";
    return;
  }
  
  // Check list of histograms
  if ( contents.empty() ) { 
    edm::LogError(mlDqmClient_)
      << "[CommissioningHistograms::" << __func__ << "]"
      << " Empty contents vector!";
    return; 
  }
  
  // Iterate through list of histograms
  std::vector<std::string>::const_iterator idir;
  for ( idir = contents.begin(); idir != contents.end(); idir++ ) {
    
    // Ignore "DQM source" directories if looking in client file
    if ( idir->find(sistrip::collate_) == std::string::npos ) { continue; }
    
    // Extract source directory path 
    std::string source_dir = idir->substr( 0, idir->find(":") );

    // Extract view and create key
    sistrip::View view = SiStripEnumsAndStrings::view( source_dir );
    SiStripKey path;
    if ( view == sistrip::CONTROL_VIEW ) { path = SiStripFecKey( source_dir ); }
    else if ( view == sistrip::READOUT_VIEW ) { path = SiStripFedKey( source_dir ); }
    else if ( view == sistrip::DETECTOR_VIEW ) { path = SiStripDetKey( source_dir ); }
    else { path = SiStripKey(); }
    
    // Check path is valid
    if ( path.granularity() == sistrip::UNKNOWN_GRAN ||
	 path.granularity() == sistrip::UNDEFINED_GRAN ) { 
      continue; 
    }
    
    // Generate corresponding client path (removing trailing "/")
    std::string client_dir(sistrip::undefinedView_);
    if ( view == sistrip::CONTROL_VIEW ) { client_dir = SiStripFecKey( path.key() ).path(); }
    else if ( view == sistrip::READOUT_VIEW ) { client_dir = SiStripFedKey( path.key() ).path(); }
    else if ( view == sistrip::DETECTOR_VIEW ) { client_dir = SiStripDetKey( path.key() ).path(); }
    else { client_dir = SiStripKey( path.key() ).path(); }
    std::string slash = client_dir.substr( client_dir.size()-1, 1 ); 
    if ( slash == sistrip::dir_ ) { client_dir = client_dir.substr( 0, client_dir.size()-1 ); }
    client_dir = std::string(sistrip::collate_) + sistrip::dir_ + client_dir;

    // Retrieve MonitorElements from source directory
    std::vector<MonitorElement*> me_list = bei_->getContents( source_dir );

    // Iterate though MonitorElements and create CMEs
    std::vector<MonitorElement*>::iterator ime = me_list.begin(); 
    for ( ; ime != me_list.end(); ime++ ) {

      // Retrieve histogram title
      SiStripHistoTitle title( (*ime)->getName() );

      // Check histogram type
      //if ( title.histoType() != sistrip::EXPERT_HISTO ) { continue; }

      // Check granularity
      uint16_t channel = sistrip::invalid_;
      if ( title.granularity() == sistrip::APV ) {
	channel = SiStripFecKey::lldChan( title.channel() );
      } else if ( title.granularity() == sistrip::UNKNOWN_GRAN || 
		  title.granularity() == sistrip::UNDEFINED_GRAN ) {
	std::stringstream ss;
	ss << "[CommissioningHistograms::" << __func__ << "]"
	   << " Unexpected granularity for histogram title: " 
	   << std::endl << title 
	   << " found in path " 
	   << std::endl << path;
	edm::LogError(mlDqmClient_) << ss.str();
      } else {
	channel = title.channel();
      }

      // Build key 
      uint32_t key = sistrip::invalid32_;

      if ( view == sistrip::CONTROL_VIEW ) { 

	// for all runs except cabling
	SiStripFecKey temp( path.key() ); 
	key = SiStripFecKey( temp.fecCrate(),
			     temp.fecSlot(),
			     temp.fecRing(),
			     temp.ccuAddr(),
			     temp.ccuChan(),
			     channel ).key();
	mapping_[title.keyValue()] = key;

      } else if ( view == sistrip::READOUT_VIEW ) { 

	// for cabling run
	key = SiStripFedKey( path.key() ).key();
	uint32_t temp = SiStripFecKey( sistrip::invalid_,
				       sistrip::invalid_,
				       sistrip::invalid_,
				       sistrip::invalid_,
				       sistrip::invalid_,
				       channel ).key(); // just record lld channel
	mapping_[title.keyValue()] = temp;

      } else if ( view == sistrip::DETECTOR_VIEW ) { 

	SiStripDetKey temp( path.key() ); 
	key = SiStripDetKey( temp.partition() ).key();
	mapping_[title.keyValue()] = key;

      } else { key = SiStripKey( path.key() ).key(); }
      
      // Find CME in histos map
      Histo* histo = 0;
      HistosMap::iterator ihistos = histos_.find( key );
      if ( ihistos != histos_.end() ) { 
	Histos::iterator ihis = ihistos->second.begin();
	while ( !histo && ihis < ihistos->second.end() ) {
	  if ( (*ime)->getName() == (*ihis)->title_ ) { histo = *ihis; }
	  ihis++;
	}
      }

      // Create CollateME if it doesn't exist
      if ( !histo ) {

	histos_[key].push_back( new Histo() );
	histo = histos_[key].back();
	histo->title_ = (*ime)->getName();

	// If histogram present in client directory, add to map
	if ( source_dir.find(sistrip::collate_) != std::string::npos ) { 
	  histo->me_ = bei_->get( client_dir +"/"+(*ime)->getName() ); 
	  if ( !histo->me_ ) { 
	    edm::LogError(mlDqmClient_)
	      << "[CommissioningHistograms::" << __func__ << "]"
	      << " NULL pointer to MonitorElement!";
	  }
	}

      }

    }
    
  }
  
  //printHistosMap();
  
  edm::LogVerbatim(mlDqmClient_)
    << "[CommissioningHistograms::" << __func__ << "]"
    << " Found histograms for " << histos_.size()
    << " structures in cached histogram map!";
  
}

// -----------------------------------------------------------------------------
/** */
void CommissioningHistograms::createCollations( const std::vector<std::string>& contents ) {

#ifndef USING_NEW_COLLATE_METHODS

  LogTrace(mlDqmClient_)
    << "[CommissioningHistograms::" << __func__ << "]"
    << " Creating collated histograms...";

  // Check pointer
  if ( !mui_ ) {
    edm::LogError(mlDqmClient_)
      << "[CommissioningHistograms::" << __func__ << "]"
      << " NULL pointer to DQMOldReceiver!";
    return;
  }
  
  // Check list of histograms
  if ( contents.empty() ) { return; }
  
  // Iterate through list of histograms
  std::vector<std::string>::const_iterator idir;
  for ( idir = contents.begin(); idir != contents.end(); idir++ ) {
    
    // Ignore directories on client side
    //if ( idir->find(sistrip::collate_) != std::string::npos ) { continue; }
    
    // Extract source directory path 
    std::string source_dir = idir->substr( 0, idir->find(":") );

    // Extract view and create key
    sistrip::View view = SiStripEnumsAndStrings::view( source_dir );
    SiStripKey path;
    if ( view == sistrip::CONTROL_VIEW ) { path = SiStripFecKey( source_dir ); }
    else if ( view == sistrip::READOUT_VIEW ) { path = SiStripFedKey( source_dir ); }
    else if ( view == sistrip::DETECTOR_VIEW ) { path = SiStripDetKey( source_dir ); }
    else { path = SiStripKey(); }
    
    // Check path is valid
    if ( path.granularity() == sistrip::FEC_SYSTEM ||
	 path.granularity() == sistrip::FED_SYSTEM || 
	 path.granularity() == sistrip::UNKNOWN_GRAN ||
	 path.granularity() == sistrip::UNDEFINED_GRAN ) { 
      continue; 
    }
    
    // Generate corresponding client path (removing trailing "/")
    std::string client_dir(sistrip::undefinedView_);
    if ( view == sistrip::CONTROL_VIEW ) { client_dir = SiStripFecKey( path.key() ).path(); }
    else if ( view == sistrip::READOUT_VIEW ) { client_dir = SiStripFedKey( path.key() ).path(); }
    else if ( view == sistrip::DETECTOR_VIEW ) { client_dir = SiStripDetKey( path.key() ).path(); }
    else { client_dir = SiStripKey( path.key() ).path(); }
    std::string slash = client_dir.substr( client_dir.size()-1, 1 ); 
    if ( slash == sistrip::dir_ ) { client_dir = client_dir.substr( 0, client_dir.size()-1 ); }
    client_dir = sistrip::collate_ + sistrip::dir_ + client_dir;

    // Retrieve MonitorElements from pwd directory
    bei_->setCurrentFolder( source_dir );
    std::vector<std::string> me_list = bei_->getMEs();

    // Iterate through MonitorElements and create CMEs
    std::vector<std::string>::iterator ime = me_list.begin(); 
    for ( ; ime != me_list.end(); ime++ ) {
      
      // Retrieve histogram title
      SiStripHistoTitle title( *ime );

      // Check histogram type
      //if ( title.histoType() != sistrip::EXPERT_HISTO ) { continue; }
      
      // Check granularity
      uint16_t channel = sistrip::invalid_;
      if ( title.granularity() == sistrip::APV ) {
	channel = SiStripFecKey::lldChan( title.channel() );
      } else if ( title.granularity() == sistrip::UNKNOWN_GRAN || 
		  title.granularity() == sistrip::UNDEFINED_GRAN ) {
	edm::LogError(mlDqmClient_)
	  << "[CommissioningHistograms::" << __func__ << "]"
	  << " Unexpected granularity for histogram title: "
	  << title << " found in path " << path;
      } else {
	channel = title.channel();
      }
      
      // Build key 
      uint32_t key = sistrip::invalid32_;

      if ( view == sistrip::CONTROL_VIEW ) { 

	// for all runs except cabling
	SiStripFecKey temp( path.key() ); 
	key = SiStripFecKey( temp.fecCrate(),
			     temp.fecSlot(),
			     temp.fecRing(),
			     temp.ccuAddr(),
			     temp.ccuChan(),
			     channel ).key();
	mapping_[title.keyValue()] = key;

      } else if ( view == sistrip::READOUT_VIEW ) { 

	// for cabling run
	key = SiStripFedKey( path.key() ).key();
	uint32_t temp = SiStripFecKey( sistrip::invalid_,
				       sistrip::invalid_,
				       sistrip::invalid_,
				       sistrip::invalid_,
				       sistrip::invalid_,
				       channel ).key(); // just record lld channel
	mapping_[title.keyValue()] = temp;

      } else if ( view == sistrip::DETECTOR_VIEW ) { 

	// for all runs except cabling
	SiStripDetKey temp( path.key() ); 
	key = SiStripDetKey( temp.partition() ).key();
	mapping_[title.keyValue()] = key;

      } else { key = SiStripKey( path.key() ).key(); }
      
      // Find CME in histos map
      Histo* histo = 0;
      HistosMap::iterator ihistos = histos_.find( key );
      if ( ihistos != histos_.end() ) { 
	Histos::iterator ihis = ihistos->second.begin();
	while ( !histo && ihis < ihistos->second.end() ) {
	  if ( (*ime) == (*ihis)->title_ ) { histo = *ihis; }
	  ihis++;
	}
      }
      
      // Create CollateME if it doesn't exist
      if ( !histo ) {

	// Retrieve ME pointer
	MonitorElement* me = bei_->get( bei_->pwd()+"/"+(*ime) );
	
	// Check if profile or 1D
	TProfile* prof = ExtractTObject<TProfile>().extract( me );
	TH1F* his = ExtractTObject<TH1F>().extract( me );

	// Create CollateME and extract pointer to ME
	if ( prof || his ) { 
	  histos_[key].push_back( new Histo() );
	  histo = histos_[key].back();
	  histo->title_ = *ime;
	  if ( prof ) {
	    prof->SetErrorOption("s"); //@@ necessary?
	    histo->cme_ = mui_->collateProf( (*ime), (*ime), client_dir ); 
	  } else if ( his ) {
	    histo->cme_ = mui_->collate1D( (*ime), (*ime), client_dir ); 
	  }
	  if ( histo->cme_ ) { 
	    mui_->add( histo->cme_, bei_->pwd()+"/"+(*ime) );
	    histo->me_ = histo->cme_->getMonitorElement(); 
	  }
	}
	
      }

      // Add to CollateME if found in histos map
      HistosMap::iterator jhistos = histos_.find( key );
      if ( jhistos != histos_.end() ) { 
	Histos::iterator ihis = jhistos->second.begin();
	while ( ihis < jhistos->second.end() ) {
	  if ( (*ime) == (*ihis)->title_ ) { 
	    if ( (*ihis)->cme_ ) {
	      mui_->add( (*ihis)->cme_, bei_->pwd()+"/"+(*ime) );
	    }
	    break; 
	  }
	  ihis++;
	}
      }
      
    }
  }
  
  //printHistosMap();

  edm::LogVerbatim(mlDqmClient_)
    << "[CommissioningHistograms::" << __func__ << "]"
    << " Found histograms for " << histos_.size()
    << " structures in cached histogram map!";
  
#endif

}

// -----------------------------------------------------------------------------
/** */
void CommissioningHistograms::histoAnalysis( bool debug ) {
  LogTrace(mlDqmClient_)
    << "[CommissioningHistograms::" << __func__ << "]"
    << " (Derived) implementation to come...";
}

// -----------------------------------------------------------------------------
/** */
void CommissioningHistograms::printAnalyses() {
  Analyses::iterator ianal = data().begin();
  Analyses::iterator janal = data().end();
  for ( ; ianal != janal; ++ianal ) { 
    if ( ianal->second ) { 
      std::stringstream ss;
      ianal->second->print( ss ); 
      if ( ianal->second->isValid() ) { LogTrace(mlDqmClient_) << ss.str(); 
      } else { edm::LogWarning(mlDqmClient_) << ss.str(); }
    }
  }
}

// -----------------------------------------------------------------------------
/** */
void CommissioningHistograms::printSummary() {

  std::stringstream good;
  std::stringstream bad;
  
  Analyses::iterator ianal = data().begin();
  Analyses::iterator janal = data().end();
  for ( ; ianal != janal; ++ianal ) { 
    if ( ianal->second ) { 
      if ( ianal->second->isValid() ) { ianal->second->summary( good ); }
      else { ianal->second->summary( bad ); }
    }
  }

  if ( good.str().empty() ) { good << "None found!"; }
  LogTrace(mlDqmClient_) 
    << "[CommissioningHistograms::" << __func__ << "]"
    << " Printing summary of good analyses:" << "\n"
    << good.str();
  
  if ( bad.str().empty() ) { return; } //@@ bad << "None found!"; }
  LogTrace(mlDqmClient_) 
    << "[CommissioningHistograms::" << __func__ << "]"
    << " Printing summary of bad analyses:" << "\n"
    << bad.str();
  
}

// -----------------------------------------------------------------------------
/** */
void CommissioningHistograms::printHistosMap() {
  LogTrace(mlDqmClient_)
    << "[CommissioningHistograms::" << __func__ << "]"
    << " Printing histogram map, which has "
    << histos_.size() << " entries...";
  HistosMap::const_iterator ihistos = histos_.begin();
  for ( ; ihistos != histos_.end(); ihistos++ ) {
    std::stringstream ss;
    ss << " Found " << ihistos->second.size()
       << " histogram(s) for key: " << std::endl
       << SiStripFedKey(ihistos->first) << std::endl;
    Histos::const_iterator ihisto = ihistos->second.begin();
    for ( ; ihisto != ihistos->second.end(); ihisto++ ) {
      if ( *ihisto ) { (*ihisto)->print(ss); }
      else { ss << " NULL pointer to Histo object!"; }
    }
    LogTrace(mlDqmClient_) << ss.str();
  }
}

// -----------------------------------------------------------------------------
/** */
void CommissioningHistograms::clearHistosMap() {
  LogTrace(mlDqmClient_)
    << "[CommissioningHistograms::" << __func__ << "]"
    << " Clearing histogram map...";
  HistosMap::iterator ihistos = histos_.begin();
  for ( ; ihistos != histos_.end(); ihistos++ ) {
    Histos::iterator ihisto = ihistos->second.begin();
    for ( ; ihisto != ihistos->second.end(); ihisto++ ) {
      if ( *ihisto ) { delete *ihisto; }
    }
    ihistos->second.clear();
  }
  histos_.clear();
}

// -----------------------------------------------------------------------------
/** */
void CommissioningHistograms::createSummaryHisto( const sistrip::Monitorable& mon, 
						  const sistrip::Presentation& pres, 
						  const std::string& dir,
						  const sistrip::Granularity& gran ) {
  LogTrace(mlDqmClient_)
    << "[CommissioningHistograms::" << __func__ << "]";
  
  // Check view 
  sistrip::View view = SiStripEnumsAndStrings::view(dir);
  if ( view == sistrip::UNKNOWN_VIEW ) { return; }
  
  // Analyze histograms
  if ( data().empty() ) { histoAnalysis( false ); }

  // Check
  if ( data().empty() ) { 
    edm::LogError(mlDqmClient_)
      << "[CommissioningHistograms::" << __func__ << "]"
      << " No analyses generated!";
    return;
  }
  
  // Extract data to be histogrammed
  uint32_t xbins = factory()->init( mon, pres, view, dir, gran, data() );
  
  // Only create histograms if entries are found!
  if ( !xbins ) { return; }
  
  // Create summary histogram (if it doesn't already exist)
  TH1* summary = 0;
  if ( pres != sistrip::HISTO_1D ) { summary = histogram( mon, pres, view, dir, xbins ); }
  else { summary = histogram( mon, pres, view, dir, sistrip::FED_ADC_RANGE, 0., sistrip::FED_ADC_RANGE*1. ); }
  
  // Fill histogram with data
  factory()->fill( *summary );
  
}

// -----------------------------------------------------------------------------
/** */
void CommissioningHistograms::remove( std::string pattern ) {
  
  if ( !mui_ ) { 
    edm::LogError(mlDqmClient_)
      << "[CommissioningHistograms::" << __func__ << "]"
      << " NULL pointer to DQMOldReceiver!"; 
    return;
  }

  if ( !mui_->getBEInterface() ) { 
    edm::LogError(mlDqmClient_)
      << "[CommissioningHistograms::" << __func__ << "]"
      << " NULL pointer to DQMStore!"; 
    return;
  }
  
  mui_->getBEInterface()->setVerbose(0);

  LogTrace(mlDqmClient_)
    << "[CommissioningHistograms::" << __func__ << "]"
    << " Removing histograms...";
  
  if ( !pattern.empty() ) {
    
    if ( mui_->getBEInterface()->dirExists(pattern) ) {
      mui_->getBEInterface()->rmdir(pattern); 
    }
    
    LogTrace(mlDqmClient_)
      << "[CommissioningHistograms::" << __func__ << "]"
      << " Removing directories (and MonitorElements"
      << " therein) that match the pattern \""
      << pattern << "\"";
    
  } else {
    
    mui_->getBEInterface()->cd();
    mui_->getBEInterface()->removeContents(); 
    
    if( mui_->getBEInterface()->dirExists("Collector") ) {
      mui_->getBEInterface()->rmdir("Collector");
    }
    if( mui_->getBEInterface()->dirExists("EvF") ) {
      mui_->getBEInterface()->rmdir("EvF");
    }
    if( mui_->getBEInterface()->dirExists("SiStrip") ) {
      mui_->getBEInterface()->rmdir("SiStrip");
    }

    LogTrace(mlDqmClient_)
      << "[CommissioningHistograms::" << __func__ << "]"
      << " Removing \"DQM source\" directories (and MonitorElements therein)";
    
  }

  LogTrace(mlDqmClient_)
    << "[CommissioningHistograms::" << __func__ << "]"
    << " Removed histograms!";

  mui_->getBEInterface()->setVerbose(1);

}

// -----------------------------------------------------------------------------
/** */
void CommissioningHistograms::save( std::string& path,
				    uint32_t run_number ) {
  
  if ( !mui_ ) { 
    edm::LogError(mlDqmClient_)
      << "[CommissioningHistograms::" << __func__ << "]"
      << " NULL pointer to DQMOldReceiver!"; 
    return;
  }

  // Construct path and filename
  std::stringstream ss; 

  if ( !path.empty() ) { 

    ss << path; 
    if ( ss.str().find(".root") == std::string::npos ) { ss << ".root"; }

  } else {

    // Retrieve SCRATCH directory
    std::string scratch = "SCRATCH";
    std::string dir = "";
    if ( getenv(scratch.c_str()) != NULL ) { 
      dir = getenv(scratch.c_str()); 
    }
    
    // Add directory path 
    if ( !dir.empty() ) { ss << dir << "/"; }
    else { ss << "/tmp/"; }
    
    // Add filename with run number and ".root" extension
    ss << sistrip::dqmClientFileName_ << "_" 
       << std::setfill('0') << std::setw(8) << run_number
       << ".root";
    
  }
  
  // Save file with appropriate filename
  LogTrace(mlDqmClient_)
    << "[CommissioningHistograms::" << __func__ << "]"
    << " Saving histograms to root file"
    << " (This may take some time!)";
  path = ss.str();
  bei_->save( path, sistrip::collate_ ); 
  edm::LogVerbatim(mlDqmClient_)
    << "[CommissioningHistograms::" << __func__ << "]"
    << " Saved histograms to root file \""
    << ss.str() << "\"!";
  
}

// -----------------------------------------------------------------------------
// 
TH1* CommissioningHistograms::histogram( const sistrip::Monitorable& mon, 
					 const sistrip::Presentation& pres, 
					 const sistrip::View& view,
					 const std::string& directory,
					 const uint32_t& xbins, 
					 const float& xlow,
					 const float& xhigh ) {
  
  // Remember pwd 
  std::string pwd = bei_->pwd();
  bei_->setCurrentFolder( std::string(sistrip::collate_) + sistrip::dir_ + directory );
  
  // Construct histogram name 
  std::string name = SummaryGenerator::name( task_, mon, pres, view, directory );
  
  // Check if summary plot already exists and remove
  MonitorElement* me = bei_->get( bei_->pwd() + "/" + name );
  if ( me ) { 
    mui_->getBEInterface()->removeElement( name );
    me = 0;
  } 
  
  // Create summary plot
  float high = static_cast<float>( xbins );
  if ( pres == sistrip::HISTO_1D ) { 
    if ( xlow < 1. * sistrip::valid_ && 
	 xhigh < 1. * sistrip::valid_ ) { 
      me = mui_->getBEInterface()->book1D( name, name, xbins, xlow, xhigh ); 
    } else {
      me = mui_->getBEInterface()->book1D( name, name, xbins, 0., high ); 
    }
  } else if ( pres == sistrip::HISTO_2D_SUM ) { 
    me = mui_->getBEInterface()->book1D( name, name, 
					 xbins, 0., high ); 
  } else if ( pres == sistrip::HISTO_2D_SCATTER ) { 
    me = mui_->getBEInterface()->book2D( name, name, xbins, 0., high, 
					 sistrip::FED_ADC_RANGE+1, 
					 0., 
					 sistrip::FED_ADC_RANGE*1. ); 
  } else if ( pres == sistrip::PROFILE_1D ) { 
    me = mui_->getBEInterface()->bookProfile( name, name, xbins, 0., high, 
					      sistrip::FED_ADC_RANGE+1, 
					      0., 
					      sistrip::FED_ADC_RANGE*1. ); 
  } else { 
    me = 0; 
    edm::LogWarning(mlDqmClient_)
      << "[CommissioningHistograms::" << __func__ << "]"
      << " Unexpected presentation \"" 
      << SiStripEnumsAndStrings::presentation( pres )
      << "\" Unable to create summary plot!";
  }
  
  // Check pointer
  if ( me ) { 
    LogTrace(mlDqmClient_)
      << "[CommissioningHistograms::" << __func__ << "]"
      << " Created summary plot with name \"" << me->getName()
      << "\" in directory \""
      << bei_->pwd() << "\"!"; 
  } else {
    edm::LogWarning(mlDqmClient_)
      << "[CommissioningHistograms::" << __func__ << "]"
      << " NULL pointer to MonitorElement!"
      << " Unable to create summary plot!";
  }
  
  // Extract root object
  TH1* summary = ExtractTObject<TH1>().extract( me ); 
  if ( !summary ) {
    edm::LogWarning(mlDqmClient_)
      << "[CommissioningHistograms::" << __func__ << "]"
      << " Unable to extract root object!"
      << " Returning NULL pointer!"; 
  }
  
  // Return to pwd
  bei_->setCurrentFolder( pwd );
  
  return summary;
  
}

