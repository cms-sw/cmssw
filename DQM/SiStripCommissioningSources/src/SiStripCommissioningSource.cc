#include "DQM/SiStripCommissioningSources/interface/SiStripCommissioningSource.h"
#include "CalibFormats/SiStripObjects/interface/SiStripFecCabling.h"
#include "CondFormats/DataRecord/interface/SiStripFedCablingRcd.h"
#include "CondFormats/SiStripObjects/interface/SiStripFedCabling.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/SiStripCommon/interface/SiStripEnumsAndStrings.h"
#include "DataFormats/SiStripCommon/interface/SiStripFecKey.h"
#include "DataFormats/SiStripCommon/interface/SiStripFedKey.h"
#include "DataFormats/SiStripCommon/interface/SiStripEventSummary.h"
#include "DQM/SiStripCommissioningSources/interface/Averages.h"
#include "DQM/SiStripCommissioningSources/interface/FedCablingTask.h"
#include "DQM/SiStripCommissioningSources/interface/ApvTimingTask.h"
#include "DQM/SiStripCommissioningSources/interface/FedTimingTask.h"
#include "DQM/SiStripCommissioningSources/interface/OptoScanTask.h"
#include "DQM/SiStripCommissioningSources/interface/VpspScanTask.h"
#include "DQM/SiStripCommissioningSources/interface/PedestalsTask.h"
#include "DQM/SiStripCommissioningSources/interface/DaqScopeModeTask.h"
#include "DQM/SiStripCommissioningSources/interface/FineDelayTask.h"
#include "DQMServices/Core/interface/DaqMonitorBEInterface.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/Exception.h"
#include <boost/cstdint.hpp>
#include <memory>
#include <iomanip>
#include <sstream>
#include <time.h>

using namespace sistrip;

// -----------------------------------------------------------------------------
//
SiStripCommissioningSource::SiStripCommissioningSource( const edm::ParameterSet& pset ) :
  dqm_(0),
  fedCabling_(0),
  fecCabling_(0),
  inputModuleLabel_( pset.getParameter<std::string>( "InputModuleLabel" ) ),
  filename_( pset.getUntrackedParameter<std::string>("RootFileName","Source") ),
  run_(0),
  time_(0),
  taskConfigurable_( pset.getUntrackedParameter<std::string>("CommissioningTask","UNDEFINED") ),
  task_(sistrip::UNDEFINED_RUN_TYPE),
  tasks_( 1024, VecOfTasks(96) ),
  cablingTasks_(),
  tasksExist_(false),
  cablingTask_(false),
  updateFreq_( pset.getUntrackedParameter<int>("HistoUpdateFreq",1) )
{
  LogTrace(mlDqmSource_)
    << "[SiStripCommissioningSource::" << __func__ << "]"
    << " Constructing object...";
}

// -----------------------------------------------------------------------------
//
SiStripCommissioningSource::~SiStripCommissioningSource() {
  LogTrace(mlDqmSource_)
    << "[SiStripCommissioningSource::" << __func__ << "]"
    << " Destructing object...";
}

// -----------------------------------------------------------------------------
//
DaqMonitorBEInterface* const SiStripCommissioningSource::dqm( std::string method ) const {
  if ( !dqm_ ) { 
    std::stringstream ss;
    if ( method != "" ) { ss << "[SiStripCommissioningSource::" << method << "]" << std::endl; }
    else { ss << "[SiStripCommissioningSource]" << std::endl; }
    ss << " NULL pointer to DaqMonitorBEInterface";
    edm::LogWarning(mlDqmSource_) << ss.str();
    return 0;
  } else { return dqm_; }
}

// -----------------------------------------------------------------------------
// Retrieve DQM interface, control cabling and "control view" utility
// class, create histogram directory structure and generate "reverse"
// control cabling.
void SiStripCommissioningSource::beginJob( const edm::EventSetup& setup ) {
  LogTrace(mlDqmSource_) 
    << "[SiStripCommissioningSource::" << __func__ << "]"
    << " Configuring..." << std::endl;
  
  // ---------- DQM back-end interface ----------

  dqm_ = edm::Service<DaqMonitorBEInterface>().operator->();
  dqm(__func__);
  dqm()->setVerbose(0);
  
  // ---------- FED and FEC cabling ----------
  
  edm::ESHandle<SiStripFedCabling> fed_cabling;
  setup.get<SiStripFedCablingRcd>().get( fed_cabling ); 
  fedCabling_ = const_cast<SiStripFedCabling*>( fed_cabling.product() ); 
  fecCabling_ = new SiStripFecCabling( *fed_cabling );
  if ( fecCabling_->crates().empty() ) {
    std::stringstream ss;
    ss << "[SiStripCommissioningSource::" << __func__ << "]"
       << " Empty std::vector returned by FEC cabling object!" 
       << " Check if database connection failed...";
    edm::LogWarning(mlDqmSource_) << ss.str();
  }

  // ---------- Reset ---------- 

  tasksExist_ = false;
  task_ = sistrip::UNDEFINED_RUN_TYPE;
  cablingTask_ = false;
  
  dqm()->rmdir(sistrip::root_);

  clearCablingTasks();
  clearTasks();
  
}

// -----------------------------------------------------------------------------
//
void SiStripCommissioningSource::endJob() {

  LogTrace(mlDqmSource_) 
    << "[SiStripCommissioningSource::" << __func__ << "]"
    << " Halting..." << std::endl;

  // ---------- Update histograms ----------
  
  // Cabling task
  for ( TaskMap::iterator itask = cablingTasks_.begin(); itask != cablingTasks_.end(); itask++ ) { 
    if ( itask->second ) { itask->second->updateHistograms(); }
  }
  
  // All tasks except cabling 
  uint16_t fed_id = 0;
  uint16_t fed_ch = 0;
  std::vector<uint16_t>::const_iterator ifed = fedCabling_->feds().begin(); 
  for ( ; ifed != fedCabling_->feds().end(); ifed++ ) { 
    const std::vector<FedChannelConnection>& conns = fedCabling_->connections(*ifed);
    std::vector<FedChannelConnection>::const_iterator iconn = conns.begin();
    for ( ; iconn != conns.end(); iconn++ ) {
      if ( !iconn->isConnected() ) { continue; }
      fed_id = iconn->fedId();
      fed_ch = iconn->fedCh();
      if ( tasks_[fed_id][fed_ch] ) { 
	tasks_[fed_id][fed_ch]->updateHistograms();
      }
    }
  }
  
  // ---------- Save histos to root file ----------

  std::string name;
  if ( filename_.find(".root",0) == std::string::npos ) { name = filename_; }
  else { name = filename_.substr( 0, filename_.find(".root",0) ); }
  std::stringstream ss; ss << name << "_" << std::setfill('0') << std::setw(7) << run_ << ".root";
  dqm()->save( ss.str() ); 
  // write std::map to root file here

  // ---------- Delete histograms ----------
  
  // Remove all MonitorElements in "SiStrip" dir and below
  // dqm()->rmdir(sistrip::root_);

  // Delete histogram objects
  // clearCablingTasks();
  // clearTasks();
  
  // ---------- Delete cabling ----------

  if ( fedCabling_ ) { fedCabling_ = 0; }
  if ( fecCabling_ ) { delete fecCabling_; fecCabling_ = 0; }
  
}

// ----------------------------------------------------------------------------
//
void SiStripCommissioningSource::analyze( const edm::Event& event, 
					  const edm::EventSetup& setup ) {

  // Retrieve commissioning information from "event summary" 
  edm::Handle<SiStripEventSummary> summary;
  //@@ BUG? why below added? why not initialized?
  inputModuleLabelSummary_ = inputModuleLabel_;
  event.getByLabel( inputModuleLabelSummary_, summary );

  // Check if EventSummary has info attached
  if ( ( summary->runType() == sistrip::UNDEFINED_RUN_TYPE ||
	 summary->runType() == sistrip::UNKNOWN_RUN_TYPE ) &&
       summary->nullParams() ) {
    edm::LogWarning(mlDqmSource_)
      << "[SiStripCommissioningSource::" << __func__ << "]"
      << " Unknown/undefined RunType and NULL parameter values!"
      << " It may be that the 'trigger FED' object was not found!"; 
 }
  
  // Extract run number
  if ( event.id().run() != run_ ) { run_ = event.id().run(); }

  // Coarse event rate counter
  if ( !(event.id().event()%updateFreq_) ) {
    std::stringstream ss;
    ss << "[SiStripCommissioningSource::" << __func__ << "]"
       << " The last " << updateFreq_ 
       << " events were processed at a rate of ";
    if ( time(NULL) == time_ ) { ss << ">" << updateFreq_ << " Hz"; }
    else { ss << (updateFreq_/(time(NULL)-time_)) << " Hz"; }
    edm::LogVerbatim(mlDqmSource_) << ss.str();
    time_ = time(NULL);
  }
  
  // Create commissioning task objects 
  if ( !tasksExist_ ) { createTask( summary.product() ); }
  
  // Retrieve raw digis
  edm::Handle< edm::DetSetVector<SiStripRawDigi> > raw;
  if ( task_ == sistrip::FED_CABLING ||
       task_ == sistrip::APV_TIMING ||
       task_ == sistrip::FED_TIMING ||
       task_ == sistrip::OPTO_SCAN ||
       task_ == sistrip::DAQ_SCOPE_MODE ) { 
    event.getByLabel( inputModuleLabel_, "ScopeMode", raw );
  } else if ( task_ == sistrip::VPSP_SCAN ||
	      task_ == sistrip::PEDESTALS ) {
    event.getByLabel( inputModuleLabel_, "VirginRaw", raw );
  } else if ( task_ == sistrip::FINE_DELAY ) {
    event.getByLabel( inputModuleLabel_, "FineDelaySelection", raw );
  } else {
    std::stringstream ss;
    ss << "[SiStripCommissioningSource::" << __func__ << "]"
       << " Unknown CommissioningTask: " 
       << SiStripEnumsAndStrings::runType( task_ )
       << " Unable to establish FED readout mode and retrieve digi container!"
       << " Check if SiStripEventSummary object is found/present in Event";
    edm::LogWarning(mlDqmSource_) << ss.str();
    return;
  }

  // Check for NULL pointer to digi container
  if ( &(*raw) == 0 ) {
    std::stringstream ss;
    ss << "[SiStripCommissioningSource::" << __func__ << "]" << std::endl
       << " NULL pointer to DetSetVector!" << std::endl
       << " Unable to fill histograms!";
    edm::LogWarning(mlDqmSource_) << ss.str();
    return;
  }
  
  if ( !cablingTask_ ) { fillHistos( summary.product(), *raw );  }
  else { fillCablingHistos( summary.product(), *raw ); }
  
}

// ----------------------------------------------------------------------------
//
void SiStripCommissioningSource::fillCablingHistos( const SiStripEventSummary* const summary,
						    const edm::DetSetVector<SiStripRawDigi>& raw ) {
  
  // Create FEC key using DCU id and LLD channel from SiStripEventSummary
  const SiStripModule& module = fecCabling_->module( summary->dcuId() );
  uint16_t lld_channel = summary->deviceId() & 0x3;
  SiStripFecKey key_object( module.key().fecCrate(),
			    module.key().fecSlot(),
			    module.key().fecRing(),
			    module.key().ccuAddr(),
			    module.key().ccuChan(),
			    lld_channel );
  uint32_t fec_key = key_object.key();
  
  // Check on whether DCU id is found
  if ( !key_object.fecCrate() &&
       !key_object.fecSlot() &&
       !key_object.ccuAddr() &&
       !key_object.ccuChan() &&
       !key_object.channel() ) {
    std::stringstream ss;
    ss << "[SiStripCommissioningSource::" << __func__ << "]" 
       << " DcuId 0x"
       << std::hex << std::setw(8) << std::setfill('0') << summary->dcuId() << std::dec 
       << " in 'DAQ register' field not found in cabling std::map!"
       << " (NULL values returned for FEC path)";
    edm::LogWarning(mlDqmSource_) << ss.str();
    return;
  }
    
  // Iterate through FED ids
  std::vector<uint16_t>::const_iterator ifed = fedCabling_->feds().begin(); 
  for ( ; ifed != fedCabling_->feds().end(); ifed++ ) {

    // Check if FedId is non-zero
    if ( *ifed == sistrip::invalid_ ) { continue; }
    
    // Container to hold median signal level for FED cabling task
    std::map<uint16_t,float> medians; medians.clear(); 
    
    // Iterate through FED channels
    for ( uint16_t ichan = 0; ichan < 96; ichan++ ) {
//       LogTrace(mlDqmSource_) << " FedCh: " << ichan;
      
      // Retrieve digis for given FED key
      uint32_t fed_key = SiStripFedKey( *ifed, 
					SiStripFedKey::feUnit(ichan),
					SiStripFedKey::feChan(ichan) ).key();
      std::vector< edm::DetSet<SiStripRawDigi> >::const_iterator digis = raw.find( fed_key );
      if ( digis != raw.end() ) { 
	if ( !digis->data.size() ) { continue; }
	
// 	if ( digis->data[0].adc() > 500 ) {
// 	  std::stringstream ss;
// 	  ss << " HIGH SIGNAL " << digis->data[0].adc() << " FOR"
// 	     << " FedKey: 0x" << std::hex << std::setw(8) << std::setfill('0') << fed_key << std::dec
// 	     << " FedId/Ch: " << *ifed << "/" << ichan;
// 	  LogTrace(mlDqmSource_) << ss.str();
// 	}
	
	Averages ave;
	for ( uint16_t idigi = 0; idigi < digis->data.size(); idigi++ ) { 
	  ave.add( static_cast<uint32_t>(digis->data[idigi].adc()) ); 
	}
	Averages::Params params;
	ave.calc(params);
	medians[ichan] = params.median_; // Store median signal level
	      
// 		std::stringstream ss;
// 		ss << "Channel Averages:" << std::endl
// 		   << "  nDigis: " << digis->data.size() << std::endl
// 		   << "  num/mean/MEDIAN/rms/max/min: "
// 		   << params.num_ << "/"
// 		   << params.mean_ << "/"
// 		   << params.median_ << "/"
// 		   << params.rms_ << "/"
// 		   << params.max_ << "/"
// 		   << params.min_ << std::endl;
// 		LogTrace(mlDqmSource_) << ss.str();

      }
      
    } // fed channel loop

    // Calculate mean and spread on all (median) signal levels
    Averages average;
    std::map<uint16_t,float>::const_iterator ii = medians.begin();
    for ( ; ii != medians.end(); ii++ ) { average.add( ii->second ); }
    Averages::Params tmp;
    average.calc(tmp);
      
//     std::stringstream ss;
//     ss << "FED Averages:" << std::endl
//        << "  nChans: " << medians.size() << std::endl
//        << "  num/mean/median/rms/max/min: "
//        << tmp.num_ << "/"
//        << tmp.mean_ << "/"
//        << tmp.median_ << "/"
//        << tmp.rms_ << "/"
//        << tmp.max_ << "/"
//        << tmp.min_ << std::endl;
//     LogTrace(mlDqmSource_) << ss.str();
      
    // Calculate mean and spread on "filtered" data
    Averages truncated;
    std::map<uint16_t,float>::const_iterator jj = medians.begin();
    for ( ; jj != medians.end(); jj++ ) { 
      if ( jj->second < tmp.median_+tmp.rms_ ) { 
	truncated.add( jj->second ); 
      }
    }
    Averages::Params params;
    truncated.calc(params);
      
//     std::stringstream ss1;
//     ss1 << "Truncated Averages:" << std::endl
// 	<< "  nChans: " << medians.size() << std::endl
// 	<< "  num/mean/median/rms/max/min: "
// 	<< params.num_ << "/"
// 	<< params.mean_ << "/"
// 	<< params.median_ << "/"
// 	<< params.rms_ << "/"
// 	<< params.max_ << "/"
// 	<< params.min_ << std::endl;
//     LogTrace(mlDqmSource_) << ss1.str();

    // Identify channels with signal
//     std::stringstream ss2;
//     ss2 << "Number of possible connections: " << medians.size()
// 	<< " channel/signal: ";
    std::map<uint16_t,float> channels;
    std::map<uint16_t,float>::const_iterator ichan = medians.begin();
    for ( ; ichan != medians.end(); ichan++ ) { 
//             cout << " mean: " << params.mean_
//       	   << " rms: " << params.rms_
//       	   << " thresh: " << params.mean_ + 5.*params.rms_
//       	   << " value: " << ichan->second
//       	   << " strip: " << ichan->first << std::endl;
      if ( ichan->second > params.mean_ + 5.*params.rms_ ) { 
 	channels[ichan->first] = ichan->second;
//  	ss2 << ichan->first << "/" << ichan->second << " ";
      }
    }
//     ss2 << std::endl;
//     LogTrace(mlDqmSource_) << ss2.str();

//     LogTrace(mlDqmSource_)
//       << "[FedCablingTask::" << __func__ << "]"
//       << " Found candidate connection between device: 0x"
//       << std::setfill('0') << std::setw(8) << std::hex << summary.deviceId() << std::dec
//       << " with Crate/FEC/Ring/CCU/Module/LLDchannel: " 
//       << connection().fecCrate() << "/"
//       << connection().fecSlot() << "/"
//       << connection().fecRing() << "/"
//       << connection().ccuAddr() << "/"
//       << connection().ccuChan() << "/"
//       << connection().lldChannel()
//       << " and FedId/Ch: "
//       << fed_id << "/" << ichan->first
//       << " with signal " << ichan->second 
//       << " [adc] over background " << "XXX +/- YYY [adc]" 
//       << " (S/N = " << "ZZZ" << ")";

    
    // Fill cabling histograms
    if ( cablingTasks_.find(fec_key) != cablingTasks_.end() ) { 
      cablingTasks_[fec_key]->fillHistograms( *summary, *ifed, channels );
    } else {
      SiStripFecKey path( fec_key );
      std::stringstream ss;
      ss << "[SiStripCommissioningSource::" << __func__ << "]"
	 << " Unable to find CommissioningTask object with FecKey: " 
	 << std::hex << std::setfill('0') << std::setw(8) << fec_key << std::dec
	 << " and Crate/FEC/ring/CCU/module/LLDchan: " 
	 << path.fecCrate() << "/"
	 << path.fecSlot() << "/"
	 << path.fecRing() << "/"
	 << path.ccuAddr() << "/"
	 << path.ccuChan() << "/"
	 << path.channel();
      edm::LogWarning(mlDqmSource_) << ss.str();
    }
  
  } // fed id loop
  
}

// ----------------------------------------------------------------------------
//
void SiStripCommissioningSource::fillHistos( const SiStripEventSummary* const summary, 
					     const edm::DetSetVector<SiStripRawDigi>& raw ) {

  // Iterate through FED ids and channels
  std::vector<uint16_t>::const_iterator ifed = fedCabling_->feds().begin();
  for ( ; ifed != fedCabling_->feds().end(); ifed++ ) {

    // Iterate through connected FED channels
    const std::vector<FedChannelConnection>& conns = fedCabling_->connections(*ifed);
    std::vector<FedChannelConnection>::const_iterator iconn = conns.begin();
    for ( ; iconn != conns.end(); iconn++ ) {
      
      // Create FED key and check if non-zero
      uint32_t fed_key = SiStripFedKey( iconn->fedId(), 
					SiStripFedKey::feUnit(iconn->fedCh()),
					SiStripFedKey::feUnit(iconn->fedCh()) ).key();
      if ( !(iconn->fedId()) ) { continue; }

      if ( task_ != sistrip::FINE_DELAY ) {
       // Retrieve digis for given FED key and check if found
       std::vector< edm::DetSet<SiStripRawDigi> >::const_iterator digis = raw.find( fed_key ); 
       if ( digis != raw.end() ) { 
 	if ( tasks_[iconn->fedId()][iconn->fedCh()] ) { 
	  tasks_[iconn->fedId()][iconn->fedCh()]->fillHistograms( *summary, *digis );
	} else {
	  std::stringstream ss;
	  ss << "[SiStripCommissioningSource::" << __func__ << "]"
	     << " Unable to find CommissioningTask object with FED key " 
	     << std::hex << std::setfill('0') << std::setw(8) << fed_key << std::dec
	     << " and FED id/ch " 
	     << iconn->fedId() << "/"
	     << iconn->fedCh()
	     << " Unable to fill histograms!";
	  edm::LogWarning(mlDqmSource_) << ss.str();
	}
       }
      } else {
        // for a fine delay task, the detset key is the detid
        // we start by checking that there is a task for the fedid/fedch pair since in fine delay this is often not the case
        if ( tasks_[iconn->fedId()][iconn->fedCh()] ) {
          // Retrieve digis for given detid and check if found
          std::vector< edm::DetSet<SiStripRawDigi> >::const_iterator digis = raw.find( iconn->detId() );
          if ( digis != raw.end() ) {
            tasks_[iconn->fedId()][iconn->fedCh()]->fillHistograms( *summary, *digis );
          }
        }
      } // fine delay task
    } // fed channel loop
  } // fed id loop

}

// -----------------------------------------------------------------------------
//
void SiStripCommissioningSource::createTask( const SiStripEventSummary* const summary ) {
  
  // Set commissioning task to default ("undefined") value
  task_ = sistrip::UNDEFINED_RUN_TYPE;
  
  // Retrieve commissioning task from EventSummary
  if ( summary ) { task_ = summary->runType(); } 
  else { 
    task_ = sistrip::UNKNOWN_RUN_TYPE; 
    std::stringstream ss;
    ss << "[SiStripCommissioningSource::" << __func__ << "]"
       << " NULL pointer to SiStripEventSummary!" 
       << " Check SiStripEventSummary is found/present in Event";
    edm::LogWarning(mlDqmSource_) << ss.str();
  } 
  
  // Override task with ParameterSet configurable (if defined)
  sistrip::RunType configurable = SiStripEnumsAndStrings::runType( taskConfigurable_ );
  if ( configurable != sistrip::UNDEFINED_RUN_TYPE &&
       configurable != sistrip::UNKNOWN_RUN_TYPE ) { task_ = configurable; }
  
  // Create ME (std::string) that identifies commissioning task
  dqm()->setCurrentFolder( sistrip::root_ );
  std::string task_str = SiStripEnumsAndStrings::runType( task_ );
  dqm()->bookString( sistrip::taskId_ + sistrip::sep_ + task_str, task_str ); 
  
  // Check commissioning task is known / defined
  if ( task_ == sistrip::UNKNOWN_RUN_TYPE ||
       task_ == sistrip::UNDEFINED_RUN_TYPE ) {
    std::stringstream ss;
    ss << "[SiStripCommissioningSource::" << __func__ << "]"
       << " Unexpected CommissioningTask: " << SiStripEnumsAndStrings::runType( task_ )
       << " Unexpected value found in SiStripEventSummary and/or cfg file"
       << " If SiStripEventSummary is not present in Event, check 'CommissioningTask' configurable in cfg file";
    edm::LogWarning(mlDqmSource_) << ss.str();
    return; 
  } else {
    std::stringstream ss;
    ss << "[SiStripCommissioningSource::" << __func__ << "]"
       << " Identified CommissioningTask from EventSummary to be: " 
       << SiStripEnumsAndStrings::runType( task_ );
    LogTrace(mlDqmSource_) << ss.str();
  }
  
  // Check if commissioning task is FED cabling 
  if ( task_ == sistrip::FED_CABLING ) { cablingTask_ = true; }
  else { cablingTask_ = false; }

  std::stringstream ss;
  ss << "[SiStripCommissioningSource::" << __func__ << "]"
     << " CommissioningTask: "
     << SiStripEnumsAndStrings::runType( summary->runType() );
  LogTrace(mlDqmSource_) << ss.str();

  if ( !cablingTask_ ) { createTasks(); }
  else { createCablingTasks(); }
  tasksExist_ = true;

}

// -----------------------------------------------------------------------------
//
void SiStripCommissioningSource::createCablingTasks() {
  
  // Iterate through FEC cabling and create commissioning task objects
  for ( std::vector<SiStripFecCrate>::const_iterator icrate = fecCabling_->crates().begin(); icrate != fecCabling_->crates().end(); icrate++ ) {
    for ( std::vector<SiStripFec>::const_iterator ifec = icrate->fecs().begin(); ifec != icrate->fecs().end(); ifec++ ) {
      for ( std::vector<SiStripRing>::const_iterator iring = ifec->rings().begin(); iring != ifec->rings().end(); iring++ ) {
	for ( std::vector<SiStripCcu>::const_iterator iccu = iring->ccus().begin(); iccu != iring->ccus().end(); iccu++ ) {
	  for ( std::vector<SiStripModule>::const_iterator imodule = iccu->modules().begin(); imodule != iccu->modules().end(); imodule++ ) {
	      
	    // Set working directory prior to booking histograms
	    SiStripFecKey path( icrate->fecCrate(), 
				ifec->fecSlot(), 
				iring->fecRing(), 
				iccu->ccuAddr(), 
				imodule->ccuChan() );
	    std::string dir = path.path();
	    dqm()->setCurrentFolder( dir );
	      
	    // Iterate through all APV pairs for this module
	    for ( uint16_t ipair = 0; ipair < imodule->nApvPairs(); ipair++ ) {
		
	      // Retrieve active APV devices
	      SiStripModule::PairOfU16 apvs = imodule->activeApvPair( imodule->lldChannel(ipair) );
		
	      // Create connection object to hold all relevant info
	      FedChannelConnection conn( icrate->fecCrate(), 
					 ifec->fecSlot(), 
					 iring->fecRing(), 
					 iccu->ccuAddr(), 
					 imodule->ccuChan(),
					 apvs.first,
					 apvs.second,
					 imodule->dcuId(),
					 imodule->detId(),
					 imodule->nApvPairs() );
		
	      // Define key encoding control path 
	      uint32_t key = SiStripFecKey( icrate->fecCrate(), 
					    ifec->fecSlot(), 
					    iring->fecRing(), 
					    iccu->ccuAddr(), 
					    imodule->ccuChan(),
					    imodule->lldChannel(ipair) ).key();
		
	      // Check key is non zero
	      if ( !key ) { 
		edm::LogWarning(mlDqmSource_)
		  << "[SiStripCommissioningSource::" << __func__ << "]"
		  << " Unexpected NULL value for FEC key!";
		continue; 
	      }
		
	      // Create cabling task objects if not already existing
	      if ( cablingTasks_.find( key ) == cablingTasks_.end() ) {
		
                if ( task_ == sistrip::FED_CABLING ) { 
		  cablingTasks_[key] = new FedCablingTask( dqm(), conn ); 
		} else if ( task_ == sistrip::UNDEFINED_RUN_TYPE ) { 
                  edm::LogWarning(mlDqmSource_)
		    << "[SiStripCommissioningSource::" << __func__ << "]"
		    << " Undefined CommissioningTask" 
		    << " Unable to create FedCablingTask object!";
		} else if ( task_ == sistrip::UNKNOWN_RUN_TYPE ) { 
                  edm::LogWarning(mlDqmSource_)
		    << "[SiStripCommissioningSource::" << __func__ << "]"
		    << " Unknown CommissioningTask" 
		    << " Unable to create FedCablingTask object!";
                } else { 
                  edm::LogWarning(mlDqmSource_)
		    << "[SiStripCommissioningSource::" << __func__ << "]"
		    << " Unexpected CommissioningTask: " 
		    << SiStripEnumsAndStrings::runType( task_ )
		    << " Unable to create FedCablingTask object!";
                }
		
		// Check if key is found and, if so, book histos and set update freq
		if ( cablingTasks_.find( key ) != cablingTasks_.end() ) {
		  if ( cablingTasks_[key] ) {
		    cablingTasks_[key]->bookHistograms(); 
		    cablingTasks_[key]->updateFreq(1); //@@ hardwired to update every event!!! 
		    std::stringstream ss;
		    ss << "[SiStripCommissioningSource::" << __func__ << "]"
		       << " Booking histograms for '" << cablingTasks_[key]->myName()
		       << "' object with key 0x" << std::hex << std::setfill('0') << std::setw(8) << key << std::dec
		       << " in directory " << dir;
		    LogTrace(mlDqmSource_) << ss.str();
		  } else {
		    std::stringstream ss;
		    ss << "[SiStripCommissioningSource::" << __func__ << "]"
		       << " NULL pointer to CommissioningTask for key 0x"
		       << std::hex << std::setfill('0') << std::setw(8) << key << std::dec
		       << " in directory " << dir 
		       << " Unable to book histograms!";
		    edm::LogWarning(mlDqmSource_) << ss.str();
		  }
		} else {
		  std::stringstream ss;
		  ss << "[SiStripCommissioningSource::" << __func__ << "]"
		     << " Unable to find CommissioningTask for key 0x"
		     << std::hex << std::setfill('0') << std::setw(8) << key << std::dec
		     << " in directory " << dir
		     << " Unable to book histograms!";
		  edm::LogWarning(mlDqmSource_) << ss.str();
		}
	      
	      } else {
		std::stringstream ss;
		ss << "[SiStripCommissioningSource::" << __func__ << "]"
		   << " CommissioningTask object already exists for key 0x"
		   << std::hex << std::setfill('0') << std::setw(8) << key << std::dec
		   << " in directory " << dir 
		   << " Unable to create FedCablingTask object!";
		edm::LogWarning(mlDqmSource_) << ss.str();
	      }
	      
	    } // loop through apv pairs
	  } // loop through modules
	} // loop through ccus
      } // loop through rings
    } // loop through fecs
  } // loop through crates
  
}

// -----------------------------------------------------------------------------
//
void SiStripCommissioningSource::createTasks() {
  // list of already used detids
  std::map<uint32_t,bool> detids;
  // Iterate through FED ids and channels 
  std::vector<uint16_t>::const_iterator ifed = fedCabling_->feds().begin();
  for ( ; ifed != fedCabling_->feds().end(); ifed++ ) {
    
    // Iterate through connected FED channels
    const std::vector<FedChannelConnection>& conns = fedCabling_->connections(*ifed);
    std::vector<FedChannelConnection>::const_iterator iconn = conns.begin();
    for ( ; iconn != conns.end(); iconn++ ) {
      
      // Create FED key and check if non-zero
      uint32_t fed_key = SiStripFedKey( iconn->fedId(), 
					SiStripFedKey::feUnit(iconn->fedCh()),
					SiStripFedKey::feChan(iconn->fedCh()) ).key();
      if ( !iconn->isConnected() ) { continue; }
      
      // Set working directory prior to booking histograms 
      SiStripFecKey path( iconn->fecCrate(), 
			  iconn->fecSlot(), 
			  iconn->fecRing(), 
			  iconn->ccuAddr(), 
			  iconn->ccuChan() );
      std::string dir = path.path();
      dqm()->setCurrentFolder( dir );
      
      // Create commissioning task objects
      if ( !tasks_[iconn->fedId()][iconn->fedCh()] ) { 
	if ( task_ == sistrip::APV_TIMING ) { tasks_[iconn->fedId()][iconn->fedCh()] = new ApvTimingTask( dqm(), *iconn ); } 
	else if ( task_ == sistrip::FED_TIMING ) { tasks_[iconn->fedId()][iconn->fedCh()] = new FedTimingTask( dqm(), *iconn ); }
	else if ( task_ == sistrip::OPTO_SCAN ) { tasks_[iconn->fedId()][iconn->fedCh()] = new OptoScanTask( dqm(), *iconn ); }
	else if ( task_ == sistrip::VPSP_SCAN ) { tasks_[iconn->fedId()][iconn->fedCh()] = new VpspScanTask( dqm(), *iconn ); }
	else if ( task_ == sistrip::PEDESTALS ) { tasks_[iconn->fedId()][iconn->fedCh()] = new PedestalsTask( dqm(), *iconn ); }
	else if ( task_ == sistrip::DAQ_SCOPE_MODE ) { tasks_[iconn->fedId()][iconn->fedCh()] = new DaqScopeModeTask( dqm(), *iconn ); }
        else if ( task_ == sistrip::FINE_DELAY ) {
          //only create one task per module
          //it would be simpler if tasks were stored in a map, but here we reuse the vector of vectors.
          //a task is created for the first fedid/fedch pair of each module
          if(detids.find(iconn->detId())==detids.end()) {
            detids[iconn->detId()] = 1;
            tasks_[iconn->fedId()][iconn->fedCh()] = new FineDelayTask( dqm(), *iconn );
          } else tasks_[iconn->fedId()][iconn->fedCh()] = 0;
        }
	else if ( task_ == sistrip::UNDEFINED_RUN_TYPE ) { 
	  edm::LogWarning(mlDqmSource_)  
	    << "[SiStripCommissioningSource::" << __func__ << "]"
	    << " Undefined CommissioningTask" 
	    << " Unable to create CommissioningTask object!";
	} else { 
	  edm::LogWarning(mlDqmSource_)
	    << "[SiStripCommissioningSource::" << __func__ << "]"
	    << " Unknown CommissioningTask" 
	    << " Unable to create CommissioningTask object!";
	}
	
	// Check if fed_key is found and, if so, book histos and set update freq
	if ( tasks_[iconn->fedId()][iconn->fedCh()] ) {
	  tasks_[iconn->fedId()][iconn->fedCh()]->bookHistograms(); 
	  tasks_[iconn->fedId()][iconn->fedCh()]->updateFreq( updateFreq_ ); 
	  std::stringstream ss;
	  ss << "[SiStripCommissioningSource::" << __func__ << "]"
	     << " Booking histograms for '" << tasks_[iconn->fedId()][iconn->fedCh()]->myName()
	     << "' object with key 0x" << std::hex << std::setfill('0') << std::setw(8) << fed_key << std::dec
	     << " in directory " << dir;
	  LogTrace(mlDqmSource_) << ss.str();
	} else {
	  std::stringstream ss;
	  ss << "[SiStripCommissioningSource::" << __func__ << "]"
	     << " NULL pointer to CommissioningTask for key 0x"
	     << std::hex << std::setfill('0') << std::setw(8) << fed_key << std::dec
	     << " in directory " << dir 
	     << " Unable to book histograms!";
	  edm::LogWarning(mlDqmSource_) << ss.str();
	}
	
      } else {
	std::stringstream ss;
	ss << "[SiStripCommissioningSource::" << __func__ << "]"
	   << " CommissioningTask object already exists for key 0x"
	   << std::hex << std::setfill('0') << std::setw(8) << fed_key << std::dec
	   << " in directory " << dir 
	   << " Unable to create CommissioningTask object!";
	edm::LogWarning(mlDqmSource_) << ss.str();
      }
      
    }
  }
  
  LogTrace(mlDqmSource_) 
       << "[SiStripCommissioningSource::" << __func__ << "]"
       << " Number of CommissioningTask objects created: " 
       << tasks_.size();
  
}

// ----------------------------------------------------------------------------
//
void SiStripCommissioningSource::clearCablingTasks() {

  for ( TaskMap::iterator itask = cablingTasks_.begin(); itask != cablingTasks_.end(); itask++ ) { 
    if ( itask->second ) { delete itask->second; }
  }
  cablingTasks_.clear();

}

// ----------------------------------------------------------------------------
//
void SiStripCommissioningSource::clearTasks() {
  
  uint16_t length = 1024;
  tasks_.resize(length);
  for ( uint16_t ii = 0; ii < length; ii++ ) { tasks_[ii].resize(96); }
  
  std::vector<uint16_t>::const_iterator ifed = fedCabling_->feds().begin(); 
  for ( ; ifed != fedCabling_->feds().end(); ifed++ ) { 
    const std::vector<FedChannelConnection>& conns = fedCabling_->connections(*ifed);
    std::vector<FedChannelConnection>::const_iterator iconn = conns.begin();
    for ( ; iconn != conns.end(); iconn++ ) {
      static uint16_t fed_id = iconn->fedId();
      static uint16_t fed_ch = iconn->fedCh();
      if ( !fed_id && !fed_ch ) { continue; }
      if ( tasks_[fed_id][fed_ch] ) { 
	delete tasks_[fed_id][fed_ch];
	tasks_[fed_id][fed_ch] = 0;
      }
    }
  }

}
