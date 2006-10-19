#include "DQM/SiStripCommissioningSources/interface/SiStripCommissioningSource.h"
// edm
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
// dqm
#include "DQMServices/Core/interface/DaqMonitorBEInterface.h"
#include "DQM/SiStripCommon/interface/SiStripHistoNamingScheme.h"
#include "DQM/SiStripCommissioningSources/interface/Averages.h"
#include "DQM/SiStripCommissioningSources/interface/ApvTimingTask.h"
#include "DQM/SiStripCommissioningSources/interface/FedCablingTask.h"
#include "DQM/SiStripCommissioningSources/interface/FedTimingTask.h"
#include "DQM/SiStripCommissioningSources/interface/OptoScanTask.h"
#include "DQM/SiStripCommissioningSources/interface/PedestalsTask.h"
#include "DQM/SiStripCommissioningSources/interface/VpspScanTask.h"
// conditions
#include "CondFormats/DataRecord/interface/SiStripFedCablingRcd.h"
#include "CondFormats/SiStripObjects/interface/SiStripFedCabling.h"
// calibrations
#include "CalibFormats/SiStripObjects/interface/SiStripFecCabling.h"
// data formats
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/SiStripCommon/interface/SiStripConstants.h"
#include "DataFormats/SiStripDigi/interface/SiStripDigi.h"
#include "DataFormats/SiStripDigi/interface/SiStripRawDigi.h"
#include "DataFormats/SiStripDigi/interface/SiStripEventSummary.h"
#include "DataFormats/SiStripDetId/interface/SiStripControlKey.h"
#include "DataFormats/SiStripDetId/interface/SiStripReadoutKey.h"
// std, utilities
#include <boost/cstdint.hpp>
#include <memory>
#include <vector>
#include <iomanip>
#include <sstream>

//#define TEST

using namespace std;
using namespace sistrip;

// -----------------------------------------------------------------------------
//
SiStripCommissioningSource::SiStripCommissioningSource( const edm::ParameterSet& pset ) :
  inputModuleLabel_( pset.getParameter<string>( "InputModuleLabel" ) ),
  dqm_(0),
  taskFromCfg_( pset.getUntrackedParameter<string>("CommissioningTask","UNDEFINED") ),
  task_(sistrip::UNDEFINED_TASK),
  tasks_(),
  updateFreq_( pset.getUntrackedParameter<int>("HistoUpdateFreq",1) ),
  filename_( pset.getUntrackedParameter<string>("RootFileName","Source") ),
  run_(0),
  createTask_(true),
  fecCabling_(0),
  cablingTask_(false)
{
  /* LogDebug(mlCommSource_) */ cout << endl
    << "[SiStripCommissioningSource::" << __func__ << "]"
    << " Constructing object...";
}

// -----------------------------------------------------------------------------
//
SiStripCommissioningSource::~SiStripCommissioningSource() {
  /* LogDebug(mlCommSource_) */ cout << endl
    << "[SiStripCommissioningSource::" << __func__ << "]"
    << " Destructing object...";
}

// -----------------------------------------------------------------------------
//
DaqMonitorBEInterface* const SiStripCommissioningSource::dqm( string method ) const {
  if ( !dqm_ ) { 
    stringstream ss;
    if ( method != "" ) { ss << "[SiStripCommissioningSource::" << method << "]" << endl; }
    else { ss << "[SiStripCommissioningSource]" << endl; }
    ss << " NULL pointer to DaqMonitorBEInterface";
    edm::LogError(mlCommSource_) << ss.str();
    return 0;
  } else { return dqm_; }
}

// -----------------------------------------------------------------------------
// Retrieve DQM interface, control cabling and "control view" utility
// class, create histogram directory structure and generate "reverse"
// control cabling.
void SiStripCommissioningSource::beginJob( const edm::EventSetup& setup ) {
  /* LogDebug(mlCommSource_) */ cout << endl 
    << "[SiStripCommissioningSource::" << __func__ << "]"
    << " Configuring..." << endl;
  
  // Retrieve and store FED cabling, create FEC cabling
  edm::ESHandle<SiStripFedCabling> fed_cabling;
  setup.get<SiStripFedCablingRcd>().get( fed_cabling ); 
  fedCabling_ = const_cast<SiStripFedCabling*>( fed_cabling.product() ); 
  fecCabling_ = new SiStripFecCabling( *fed_cabling );
  
  // Debug
  stringstream ss;
  ss << "[SiStripCommissioningSource::" << __func__ << "]" << endl;
  fedCabling_->print(ss);
  /* LogDebug(mlCommSource_) */ cout << endl << ss.str();
  
  // Retrieve pointer to DQM back-end interface (and check for NULL pointer)
  dqm_ = edm::Service<DaqMonitorBEInterface>().operator->();
  dqm(__func__);
  dqm()->setVerbose(0);
  
  // Reset flags
  createTask_ = true;
  task_ = sistrip::UNDEFINED_TASK;
  
}

// -----------------------------------------------------------------------------
//
void SiStripCommissioningSource::endJob() {
  /* LogDebug(mlCommSource_) */ cout << endl 
    << "[SiStripCommissioningSource::" << __func__ << "]"
    << " Halting..." << endl;
  
  // Update histograms
  for ( TaskMap::iterator itask = tasks_.begin(); itask != tasks_.end(); itask++ ) { 
    if ( itask->second ) { itask->second->updateHistograms(); }
  }
  
  // Save histos to root file
  string name;
  if ( filename_.find(".root",0) == string::npos ) { name = filename_; }
  else { name = filename_.substr( 0, filename_.find(".root",0) ); }
  stringstream ss; ss << name << "_" << setfill('0') << setw(7) << run_ << ".root";
  dqm()->save( ss.str() ); 
  
  // Remove all MonitorElements in "SiStrip" dir and below
  dqm()->rmdir(sistrip::root_);
  
  // Delete commissioning task objects
  for ( TaskMap::iterator itask = tasks_.begin(); itask != tasks_.end(); itask++ ) { 
    if ( itask->second ) { delete itask->second; }
  }
  tasks_.clear();
}

// ----------------------------------------------------------------------------
//
void SiStripCommissioningSource::analyze( const edm::Event& event, 
					  const edm::EventSetup& setup ) {
  /* LogDebug(mlCommSource_) */ cout << endl 
    << "[SiStripCommissioningSource::" << __func__ << "]"
    << " Analyzing run/event "
    << event.id().run() << "/"
    << event.id().event();
  
  // Retrieve commissioning information from "event summary" 
  edm::Handle<SiStripEventSummary> summary;
  event.getByLabel( inputModuleLabel_, summary );
  summary->check();
  
  // Extract run number
  if ( event.id().run() != run_ ) { run_ = event.id().run(); }
  
  // Create commissioning task objects 
  if ( createTask_ ) { 
    createTask( summary.product() ); 
    createTask_ = false; 
  }

  stringstream ss;
  ss << "CommissioningTask: "
     << SiStripHistoNamingScheme::task( summary->task() )
     << " cablingTask_: " << cablingTask_;
  /* LogDebug(mlCommSource_) */ cout << endl << ss.str();
  
  // Retrieve raw digis
  edm::Handle< edm::DetSetVector<SiStripRawDigi> > raw;
  if ( task_ == sistrip::FED_CABLING ||
       task_ == sistrip::APV_TIMING ||
       task_ == sistrip::FED_TIMING ||
       task_ == sistrip::OPTO_SCAN ) { 
    event.getByLabel( inputModuleLabel_, "ScopeMode", raw );
  } else if ( task_ == sistrip::VPSP_SCAN ||
	      task_ == sistrip::PEDESTALS ) {
    event.getByLabel( inputModuleLabel_, "VirginRaw", raw );
  } else {
    stringstream ss;
    ss << "[SiStripCommissioningSource::" << __func__ << "]"
       << " Unknown CommissioningTask: " 
       << SiStripHistoNamingScheme::task( task_ )
       << " Unable to establish FED readout mode and retrieve digi container!"
       << " Check if SiStripEventSummary object is found/present in Event";
    edm::LogWarning(mlCommSource_) << ss.str();
    return;
  }
  // Check for NULL pointer to digi container
  if ( &(*raw) == 0 ) {
    stringstream ss;
    ss << "[SiStripCommissioningSource::" << __func__ << "]" << endl
       << " NULL pointer to DetSetVector!" << endl
       << " Unable to fill histograms!";
    edm::LogError(mlCommSource_) << ss.str();
    return;
  }
  
  // Generate FEC key (if FED cabling task)
  uint32_t fec_key = 0;
  if ( cablingTask_ ) {
    
    // Extract FEC key using DCU id and LLD channel from EventSummary
    uint32_t lld_chan = summary->deviceId() & 0x3;
    const SiStripModule& module = fecCabling_->module( summary->dcuId() );
    SiStripFecKey::Path fec_path = module.path();
    fec_path.channel_ = lld_chan;
    fec_key = SiStripFecKey::key( fec_path );
    
    stringstream ss;
    ss << "[SiStripCommissioningSource::" << __func__ << "]" << endl
       << " SiStripSummaryEvent info: DcuId: 0x" 
       << hex << setw(8) << setfill('0') << summary->dcuId() << dec
       << " LldChannel: " << lld_chan << endl
       << " FecKey: 0x"
       << hex << setw(8) << setfill('0') << summary->dcuId() << dec
       << " Crate/FEC/ring/CCU/module/channel: "
       << fec_path.fecCrate_ << "/"
       << fec_path.fecSlot_ << "/"
       << fec_path.fecRing_ << "/"
       << fec_path.ccuAddr_ << "/"
       << fec_path.ccuChan_ << "/"
       << fec_path.channel_
       << endl;
    /* LogDebug(mlCommSource_) */ cout << endl << ss.str();
    
  }    
  
  // Iterate through FED ids
  vector<uint16_t>::const_iterator ifed;
  for ( ifed = fedCabling_->feds().begin(); ifed != fedCabling_->feds().end(); ifed++ ) {

    /* LogDebug(mlCommSource_) */ cout << endl << " FedId: " << *ifed;
    
    // Container to hold median signal level for FED cabling task
    map<uint16_t,float> medians; medians.clear(); 
    
    // Iterate through FED channels
    for ( uint16_t ichan = 0; ichan < 96; ichan++ ) {
      
      // Create FED key and check if non-zero
      uint32_t fed_key = SiStripReadoutKey::key( *ifed, ichan );
      
      if ( fed_key ) { 

	// Retrieve digis for given FED key and check if found
	vector< edm::DetSet<SiStripRawDigi> >::const_iterator digis = raw->find( fed_key );
	if ( digis != raw->end() ) { 
      
	  if ( !cablingTask_ ) {
	    if ( tasks_.find(fed_key) != tasks_.end() ) { 
	      tasks_[fed_key]->fillHistograms( *summary, *digis );
	    } else {
	      SiStripReadoutKey::ReadoutPath path = SiStripReadoutKey::path( fed_key );
	      stringstream ss;
	      ss << "[SiStripCommissioningSource::" << __func__ << "]"
		 << " Unable to find CommissioningTask object with FED key " 
		 << hex << setfill('0') << setw(8) << fed_key << dec
		 << " and FED id/ch " 
		 << path.fedId_ << "/"
		 << path.fedCh_ 
		 << " Unable to fill histograms!";
	      edm::LogError(mlCommSource_) << ss.str();
	    }
	  } else { 

	    if ( digis->data.size() ) {
	      
	      if ( digis->data[0].adc() > 500 ) {
		stringstream ss;
		ss << " HIGH SIGNAL " << digis->data[0].adc() << " FOR"
		   << " FedKey: 0x" << hex << setw(8) << setfill('0') << fed_key << dec
		   << " FedId/Ch: " << *ifed << "/" << ichan;
		/* LogDebug(mlCommSource_) */ cout << endl << ss.str();
	      }
	      
	      Averages ave;
	      for ( uint16_t idigi = 0; idigi < digis->data.size(); idigi++ ) { 
		ave.add( static_cast<uint32_t>(digis->data[idigi].adc()) ); 
	      }
	      Averages::Params params;
	      ave.calc(params);
	      medians[ichan] = params.median_; // Store median signal level
	      
	      stringstream ss;
	      ss << "Channel Averages:" << endl
		 << "  nDigis: " << digis->data.size() << endl
		 << "  num/mean/MEDIAN/rms/max/min: "
		 << params.num_ << "/"
		 << params.mean_ << "/"
		 << params.median_ << "/"
		 << params.rms_ << "/"
		 << params.max_ << "/"
		 << params.min_ << endl;
	      /* LogDebug(mlCommSource_) */ cout << endl << ss.str();
	    }

#ifdef TEST 
	    // if test, overwrite median levels with test data
	    if ( *ifed == fed_ch.first && 
		 ichan == fed_ch.second ) {
	      medians[ichan] = 1000.;
	    } else { medians[ichan] = 100.; }
#endif
	    
	  }
	}
      }
      
    } // fed channel loop

      // If FED cabling task, identify channels with signal
    if ( cablingTask_ ) {

      // Calculate mean and spread on all (median) signal levels
      Averages average;
      map<uint16_t,float>::const_iterator ii = medians.begin();
      for ( ; ii != medians.end(); ii++ ) { average.add( ii->second ); }
      Averages::Params tmp;
      average.calc(tmp);
      
      stringstream ss;
      ss << "FED Averages:" << endl
	 << "  nChans: " << medians.size() << endl
	 << "  num/mean/median/rms/max/min: "
	 << tmp.num_ << "/"
	 << tmp.mean_ << "/"
	 << tmp.median_ << "/"
	 << tmp.rms_ << "/"
	 << tmp.max_ << "/"
	 << tmp.min_ << endl;
      /* LogDebug(mlCommSource_) */ cout << endl << ss.str();
      
      // Calculate mean and spread on "filtered" data
      Averages truncated;
      map<uint16_t,float>::const_iterator jj = medians.begin();
      for ( ; jj != medians.end(); jj++ ) { 
	if ( jj->second < tmp.median_+tmp.rms_ ) { 
	  truncated.add( jj->second ); 
	}
      }
      Averages::Params params;
      truncated.calc(params);
      
      stringstream ss1;
      ss1 << "Truncated Averages:" << endl
	  << "  nChans: " << medians.size() << endl
	  << "  num/mean/median/rms/max/min: "
	  << params.num_ << "/"
	  << params.mean_ << "/"
	  << params.median_ << "/"
	  << params.rms_ << "/"
	  << params.max_ << "/"
	  << params.min_ << endl;
      /* LogDebug(mlCommSource_) */ cout << endl << ss1.str();

      // Identify channels with signal
      stringstream ss2;
      ss2 << "Number of possible connections: " << medians.size()
	  << " channel/signal: ";
      map<uint16_t,float> channels;
      map<uint16_t,float>::const_iterator ichan = medians.begin();
      for ( ; ichan != medians.end(); ichan++ ) { 
	cout << " mean: " << params.mean_
	     << " rms: " << params.rms_
	     << " thresh: " << params.mean_ + 5.*params.rms_
	     << " value: " << ichan->second
	     << " strip: " << ichan->first << endl;
	if ( ichan->second > params.mean_ + 5.*params.rms_ ) { 
	  channels[ichan->first] = ichan->second;
	  ss2 << ichan->first << "/" << ichan->second << " ";
	}
      }
      ss2 << endl;
      /* LogDebug(mlCommSource_) */ cout << endl << ss2.str();
      
      // Fill cabling histograms
      if ( tasks_.find(fec_key) != tasks_.end() ) { 
	tasks_[fec_key]->fillHistograms( *summary, *ifed, channels );
      } else {
	SiStripControlKey::ControlPath path = SiStripControlKey::path( fec_key );
	stringstream ss;
	ss << "[SiStripCommissioningSource::" << __func__ << "]"
	   << " Unable to find CommissioningTask object with FEC key " 
	   << hex << setfill('0') << setw(8) << fec_key << dec
	   << " and FECcrate/FEC/ring/CCU/module/LLDchannel " 
	   << path.fecCrate_ << "/"
	   << path.fecSlot_ << "/"
	   << path.fecRing_ << "/"
	   << path.ccuAddr_ << "/"
	   << path.ccuChan_ << "/"
	   << path.channel_;
	edm::LogError(mlCommSource_) << ss.str();
      }
  
    } // if cabling task
    
  } // fed id loop
  
}

// -----------------------------------------------------------------------------
//
void SiStripCommissioningSource::createTask( const SiStripEventSummary* const summary ) {
  
  // Default value for commissioning task
  sistrip::Task task = sistrip::UNDEFINED_TASK;
  
  // Retrieve commissioning task from EventSummary
  if ( summary ) { task = summary->task(); } 
  else { 
    task = sistrip::UNKNOWN_TASK; 
    stringstream ss;
    ss << "[SiStripCommissioningSource::" << __func__ << "]"
       << " NULL pointer to SiStripEventSummary!" 
       << " Check SiStripEventSummary is found/present in Event";
    edm::LogError(mlCommSource_) << ss.str();
  } 
  
  // Override task with configurable (if set)
  sistrip::Task configurable = SiStripHistoNamingScheme::task( taskFromCfg_ );
  if ( configurable != sistrip::UNDEFINED_TASK ) { task = configurable; }
  
  // Create ME (string) that identifies commissioning task
  dqm()->setCurrentFolder( sistrip::root_ );
  string task_str = SiStripHistoNamingScheme::task( task );
  dqm()->bookString( sistrip::commissioningTask_ + sistrip::sep_ + task_str, task_str ); 
  
  // Set "commissioning task" private data member for this run
  task_ = task;
  
  // Check commissioning task is known / defined
  if ( task_ == sistrip::UNKNOWN_TASK ||
       task_ == sistrip::UNDEFINED_TASK ) {
    stringstream ss;
    ss << "[SiStripCommissioningSource::" << __func__ << "]"
       << " Unexpected CommissioningTask: " << SiStripHistoNamingScheme::task( task_ )
       << " Unexpected value found in SiStripEventSummary and/or cfg file"
       << " If SiStripEventSummary is not present in Event, check 'CommissioningTask' configurable in cfg file";
    edm::LogError(mlCommSource_) << ss.str();
    return; 
  } else {
    stringstream ss;
    ss << "[SiStripCommissioningSource::" << __func__ << "]"
       << " Identified CommissioningTask from EventSummary to be: " 
       << SiStripHistoNamingScheme::task( task_ );
    /* LogDebug(mlCommSource_) */ cout << endl << ss.str();
  }
  
  // Check if commissioning task is FED cabling 
  if ( task_ == sistrip::FED_CABLING ) { cablingTask_ = true; }
  else { cablingTask_ = false; }
  
  // Check FEC cabling object is populated
  if ( fecCabling_->crates().empty() ) {
    stringstream ss;
    ss << "[SiStripCommissioningSource::" << __func__ << "]"
       << " Empty vector returned by FEC cabling object!" 
       << " Check if database connection failed...";
    edm::LogError(mlCommSource_) << ss.str();
    return;
  }
  
  // Iterate through FEC cabling and create commissioning task objects
  for ( vector<SiStripFecCrate>::const_iterator icrate = fecCabling_->crates().begin(); icrate != fecCabling_->crates().end(); icrate++ ) {
    for ( vector<SiStripFec>::const_iterator ifec = icrate->fecs().begin(); ifec != icrate->fecs().end(); ifec++ ) {
      for ( vector<SiStripRing>::const_iterator iring = ifec->rings().begin(); iring != ifec->rings().end(); iring++ ) {
	for ( vector<SiStripCcu>::const_iterator iccu = iring->ccus().begin(); iccu != iring->ccus().end(); iccu++ ) {
	  for ( vector<SiStripModule>::const_iterator imodule = iccu->modules().begin(); imodule != iccu->modules().end(); imodule++ ) {
	    string dir = SiStripHistoNamingScheme::controlPath( icrate->fecCrate(), 
								ifec->fecSlot(), 
								iring->fecRing(), 
								iccu->ccuAddr(), 
								imodule->ccuChan() );
	    dqm()->setCurrentFolder( dir );

	    // Iterate through FED channels for this module
	    SiStripModule::FedCabling::const_iterator iconn;
	    for ( iconn = imodule->fedChannels().begin(); iconn != imodule->fedChannels().end(); iconn++ ) {
	      if ( !(iconn->second.first) ) { continue; } // if FED id, continue...

	      // Retrieve FED channel in order to create key for task map
	      FedChannelConnection conn = fedCabling_->connection( iconn->second.first,
								   iconn->second.second );
	      
	      // Define key (FEC for cabling task, FED for all other tasks) 
	      uint32_t key;
	      if ( cablingTask_ ) {
		key = SiStripControlKey::key( conn.fecCrate(),
					      conn.fecSlot(),
					      conn.fecRing(),
					      conn.ccuAddr(),
					      conn.ccuChan(),
					      conn.lldChannel() );
	      } else {
		key = SiStripReadoutKey::key( conn.fedId(), conn.fedCh() );
	      }
	      
	      // Check key in non zero
	      if ( !key ) { continue; }

	      // Create commissioning task objects
	      if ( tasks_.find( key ) == tasks_.end() ) {
		if      ( task_ == sistrip::FED_CABLING )    { tasks_[key] = new FedCablingTask( dqm(), conn ); }
		else if ( task_ == sistrip::PEDESTALS )      { tasks_[key] = new PedestalsTask( dqm(), conn ); }
		else if ( task_ == sistrip::APV_TIMING )     { tasks_[key] = new ApvTimingTask( dqm(), conn ); } 
		else if ( task_ == sistrip::OPTO_SCAN )      { tasks_[key] = new OptoScanTask( dqm(), conn ); }
		else if ( task_ == sistrip::VPSP_SCAN )      { tasks_[key] = new VpspScanTask( dqm(), conn ); }
		else if ( task_ == sistrip::FED_TIMING )     { tasks_[key] = new FedTimingTask( dqm(), conn ); }
		else if ( task_ == sistrip::UNDEFINED_TASK ) { 
		  edm::LogError(mlCommSource_)
		    << "[SiStripCommissioningSource::" << __func__ << "]"
		    << " Undefined CommissioningTask" 
		    << " Unable to create CommissioningTask object!";
		} else { 
		  edm::LogError(mlCommSource_)
		    << "[SiStripCommissioningSource::" << __func__ << "]"
		    << " Unknown CommissioningTask" 
		    << " Unable to create CommissioningTask object!";
		}
		
		// Check if key is found and, if so, book histos and set update freq
		if ( tasks_.find( key ) != tasks_.end() ) {
		  if ( tasks_[key] ) {
		    tasks_[key]->bookHistograms(); 
		    tasks_[key]->updateFreq( updateFreq_ ); 
		    stringstream ss;
		    ss << "[SiStripCommissioningSource::" << __func__ << "]"
		       << " Booking histograms for '" << tasks_[key]->myName()
		       << "' object for key 0x" << hex << setfill('0') << setw(8) << key << dec
		       << " in directory " << dir << endl;
		    /* LogDebug(mlCommSource_) */ cout << endl << ss.str();
		  } else {
		    stringstream ss;
		    ss << "[SiStripCommissioningSource::" << __func__ << "]"
		       << " NULL pointer to CommissioningTask for key 0x"
		       << hex << setfill('0') << setw(8) << key << dec
		       << " in directory " << dir 
		       << " Unable to book histograms!";
		    edm::LogError(mlCommSource_) << ss.str();
		  }
		} else {
		  stringstream ss;
		  ss << "[SiStripCommissioningSource::" << __func__ << "]"
		     << " Unable to find CommissioningTask for key 0x"
		     << hex << setfill('0') << setw(8) << key << dec
		     << " in directory " << dir
		     << " Unable to book histograms!";
		  edm::LogError(mlCommSource_) << ss.str();
		}
		
	      } else {
		stringstream ss;
		ss << "[SiStripCommissioningSource::" << __func__ << "]"
		   << " CommissioningTask object already exists for key 0x"
		   << hex << setfill('0') << setw(8) << key << dec
		   << " in directory " << dir 
		   << " Unable to create CommissioningTask object!";
		edm::LogError(mlCommSource_) << ss.str();
	      }
	      
	    }
	  }
	}
      }
    }
  }
  
  /* LogDebug(mlCommSource_) */ cout << endl
    << "[SiStripCommissioningSource::" << __func__ << "]"
    << " Number of CommissioningTask objects created: " 
    << tasks_.size();
  
}

