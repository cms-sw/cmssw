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
#include "DQM/SiStripCommissioningSources/interface/Averages.h"
#include "DQM/SiStripCommissioningSources/interface/FedCablingTask.h"
#include "DQM/SiStripCommissioningSources/interface/ApvTimingTask.h"
#include "DQM/SiStripCommissioningSources/interface/FedTimingTask.h"
#include "DQM/SiStripCommissioningSources/interface/OptoScanTask.h"
#include "DQM/SiStripCommissioningSources/interface/VpspScanTask.h"
#include "DQM/SiStripCommissioningSources/interface/PedestalsTask.h"
#include "DQM/SiStripCommissioningSources/interface/DaqScopeModeTask.h"
// conditions
#include "CondFormats/DataRecord/interface/SiStripFedCablingRcd.h"
#include "CondFormats/SiStripObjects/interface/SiStripFedCabling.h"
// calibrations
#include "CalibFormats/SiStripObjects/interface/SiStripFecCabling.h"
// data formats
#include "DataFormats/SiStripCommon/interface/SiStripConstants.h"
#include "DataFormats/SiStripCommon/interface/SiStripFecKey.h"
#include "DataFormats/SiStripCommon/interface/SiStripFedKey.h"
#include "DataFormats/SiStripDigi/interface/SiStripEventSummary.h"
// std, utilities
#include <boost/cstdint.hpp>
#include <memory>
#include <vector>
#include <iomanip>
#include <sstream>
#include <time.h>

using namespace std;
using namespace sistrip;

// -----------------------------------------------------------------------------
//
SiStripCommissioningSource::SiStripCommissioningSource( const edm::ParameterSet& pset ) :
  dqm_(0),
  fedCabling_(0),
  fecCabling_(0),
  inputModuleLabel_( pset.getParameter<string>( "InputModuleLabel" ) ),
  filename_( pset.getUntrackedParameter<string>("RootFileName","Source") ),
  run_(0),
  time_(0),
  taskConfigurable_( pset.getUntrackedParameter<string>("CommissioningTask","UNDEFINED") ),
  task_(sistrip::UNDEFINED_TASK),
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
DaqMonitorBEInterface* const SiStripCommissioningSource::dqm( string method ) const {
  if ( !dqm_ ) { 
    stringstream ss;
    if ( method != "" ) { ss << "[SiStripCommissioningSource::" << method << "]" << endl; }
    else { ss << "[SiStripCommissioningSource]" << endl; }
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
    << " Configuring..." << endl;
  
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
    stringstream ss;
    ss << "[SiStripCommissioningSource::" << __func__ << "]"
       << " Empty vector returned by FEC cabling object!" 
       << " Check if database connection failed...";
    edm::LogWarning(mlDqmSource_) << ss.str();
  }
  
  // ---------- Reset ---------- 

  tasksExist_ = false;
  task_ = sistrip::UNDEFINED_TASK;
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
    << " Halting..." << endl;

  // ---------- Update histograms ----------
  
  // Cabling task
  for ( TaskMap::iterator itask = cablingTasks_.begin(); itask != cablingTasks_.end(); itask++ ) { 
    if ( itask->second ) { itask->second->updateHistograms(); }
  }
  
  // All tasks except cabling 
  uint16_t fed_id = 0;
  uint16_t fed_ch = 0;
  vector<uint16_t>::const_iterator ifed = fedCabling_->feds().begin(); 
  for ( ; ifed != fedCabling_->feds().end(); ifed++ ) { 
    const vector<FedChannelConnection>& conns = fedCabling_->connections(*ifed);
    vector<FedChannelConnection>::const_iterator iconn = conns.begin();
    for ( ; iconn != conns.end(); iconn++ ) {
      fed_id = iconn->fedId();
      fed_ch = iconn->fedCh();
      if ( !fed_id ) { continue; }
      if ( tasks_[fed_id][fed_ch] ) { 
	tasks_[fed_id][fed_ch]->updateHistograms();
      }
    }
  }
  
  // ---------- Save histos to root file ----------

  string name;
  if ( filename_.find(".root",0) == string::npos ) { name = filename_; }
  else { name = filename_.substr( 0, filename_.find(".root",0) ); }
  stringstream ss; ss << name << "_" << setfill('0') << setw(7) << run_ << ".root";
  dqm()->save( ss.str() ); 
  // write map to root file here

  // ---------- Delete histograms ----------
  
  // Remove all MonitorElements in "SiStrip" dir and below
  dqm()->rmdir(sistrip::root_);

  // Delete histogram objects
  clearCablingTasks();
  clearTasks();
  
  // ---------- Delete cabling ----------

  if ( fedCabling_ ) { fedCabling_ = 0; }
  if ( fecCabling_ ) { delete fecCabling_; fecCabling_ = 0; }
  
}

// ----------------------------------------------------------------------------
//
void SiStripCommissioningSource::analyze( const edm::Event& event, 
					  const edm::EventSetup& setup ) {
   LogTrace(mlDqmSource_) 
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

  // Coarse event rate counter
  if ( !(event.id().event()%updateFreq_) ) {
    float rate = 
      static_cast<float>( updateFreq_ ) / 
      static_cast<float>( time(NULL) - time_ );
    rate = static_cast<int>( 10 * rate );
    rate /= 10.;
    stringstream ss;
    ss << "[SiStripCommissioningSource::" << __func__ << "]"
       << " Last " << updateFreq_ 
       << " events processed at a rate of "
       << rate << " Hz";
    edm::LogVerbatim(mlDqmSource_) << ss.str();
    time_ = time(NULL);
  }
  
  // Create commissioning task objects 
  if ( !tasksExist_ ) { createTask( summary.product() ); }
  
  stringstream ss;
  ss << "[SiStripCommissioningSource::" << __func__ << "]"
     << " CommissioningTask: "
     << SiStripHistoNamingScheme::task( summary->task() )
     << " cablingTask_: " << cablingTask_;
  LogTrace(mlDqmSource_) << ss.str();
  
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
  } else {
    stringstream ss;
    ss << "[SiStripCommissioningSource::" << __func__ << "]"
       << " Unknown CommissioningTask: " 
       << SiStripHistoNamingScheme::task( task_ )
       << " Unable to establish FED readout mode and retrieve digi container!"
       << " Check if SiStripEventSummary object is found/present in Event";
    edm::LogWarning(mlDqmSource_) << ss.str();
    return;
  }
  // Check for NULL pointer to digi container
  if ( &(*raw) == 0 ) {
    stringstream ss;
    ss << "[SiStripCommissioningSource::" << __func__ << "]" << endl
       << " NULL pointer to DetSetVector!" << endl
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
  LogTrace(mlDqmSource_) << "[SiStripCommissioningSource::" << __func__ << "]";
    
  // Create FEC key using DCU id and LLD channel from SiStripEventSummary
  const SiStripModule& module = fecCabling_->module( summary->dcuId() );
  SiStripFecKey::Path fec_path = module.path();
  fec_path.channel_ = summary->deviceId() & 0x3;
  uint32_t fec_key = SiStripFecKey::key( fec_path );
  
  stringstream ss;
  ss << "[SiStripCommissioningSource::" << __func__ << "]" 
     << " SiStripSummaryEvent info:" 
     << "  DcuId: 0x" << hex << setw(8) << setfill('0') << summary->dcuId() << dec 
     << " LldChannel: " << fec_path.channel_ 
     << "  FecKey: 0x" << hex << setw(8) << setfill('0') << fec_key << dec
     << " Crate/FEC/ring/CCU/module/LLDchan: "
     << fec_path.fecCrate_ << "/"
     << fec_path.fecSlot_ << "/"
     << fec_path.fecRing_ << "/"
     << fec_path.ccuAddr_ << "/"
     << fec_path.ccuChan_ << "/"
     << fec_path.channel_;
  LogTrace(mlDqmSource_) << ss.str();

  // Check on whether DCU id is found
  if ( !fec_path.fecCrate_ &&
       !fec_path.fecSlot_ &&
       !fec_path.ccuAddr_ &&
       !fec_path.ccuChan_ &&
       !fec_path.channel_ ) {
    stringstream ss;
    ss << "[SiStripCommissioningSource::" << __func__ << "]" 
       << " DcuId 0x"
       << hex << setw(8) << setfill('0') << summary->dcuId() << dec 
       << " in 'DAQ register' field not found in cabling map!"
       << " (NULL values returned for FEC path)";
    edm::LogWarning(mlDqmSource_) << ss.str();
    return;
  }
    
  // Iterate through FED ids
  vector<uint16_t>::const_iterator ifed = fedCabling_->feds().begin(); 
  for ( ; ifed != fedCabling_->feds().end(); ifed++ ) {
    LogTrace(mlDqmSource_) << " FedId: " << *ifed;

    // Check if FedId is non-zero
    if ( !(*ifed) ) { continue; }
    
    // Container to hold median signal level for FED cabling task
    map<uint16_t,float> medians; medians.clear(); 
    
    // Iterate through FED channels
    for ( uint16_t ichan = 0; ichan < 96; ichan++ ) {
      LogTrace(mlDqmSource_) << " FedCh: " << ichan;
      
      // Retrieve digis for given FED key
      uint32_t fed_key = SiStripFedKey::key( *ifed, ichan );
      vector< edm::DetSet<SiStripRawDigi> >::const_iterator digis = raw.find( fed_key );
      if ( digis != raw.end() ) { 
	if ( !digis->data.size() ) { continue; }
	
	if ( digis->data[0].adc() > 500 ) {
	  stringstream ss;
	  ss << " HIGH SIGNAL " << digis->data[0].adc() << " FOR"
	     << " FedKey: 0x" << hex << setw(8) << setfill('0') << fed_key << dec
	     << " FedId/Ch: " << *ifed << "/" << ichan;
	  LogTrace(mlDqmSource_) << ss.str();
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
		LogTrace(mlDqmSource_) << ss.str();

      }
      
    } // fed channel loop

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
    LogTrace(mlDqmSource_) << ss.str();
      
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
    LogTrace(mlDqmSource_) << ss1.str();

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
    LogTrace(mlDqmSource_) << ss2.str();

//     LogTrace(mlDqmSource_)
//       << "[FedCablingTask::" << __func__ << "]"
//       << " Found candidate connection between device: 0x"
//       << setfill('0') << setw(8) << hex << summary.deviceId() << dec
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
      SiStripFecKey::Path path = SiStripFecKey::path( fec_key );
      stringstream ss;
      ss << "[SiStripCommissioningSource::" << __func__ << "]"
	 << " Unable to find CommissioningTask object with FecKey: " 
	 << hex << setfill('0') << setw(8) << fec_key << dec
	 << " and Crate/FEC/ring/CCU/module/LLDchan: " 
	 << path.fecCrate_ << "/"
	 << path.fecSlot_ << "/"
	 << path.fecRing_ << "/"
	 << path.ccuAddr_ << "/"
	 << path.ccuChan_ << "/"
	 << path.channel_;
      edm::LogWarning(mlDqmSource_) << ss.str();
    }
  
  } // fed id loop
  
}

// ----------------------------------------------------------------------------
//
void SiStripCommissioningSource::fillHistos( const SiStripEventSummary* const summary, 
					     const edm::DetSetVector<SiStripRawDigi>& raw ) {

  // Iterate through FED ids and channels
  vector<uint16_t>::const_iterator ifed = fedCabling_->feds().begin();
  for ( ; ifed != fedCabling_->feds().end(); ifed++ ) {

    // Iterate through connected FED channels
    const vector<FedChannelConnection>& conns = fedCabling_->connections(*ifed);
    vector<FedChannelConnection>::const_iterator iconn = conns.begin();
    for ( ; iconn != conns.end(); iconn++ ) {
      
      // Create FED key and check if non-zero
      uint32_t fed_key = SiStripFedKey::key( iconn->fedId(), iconn->fedCh() );
      if ( !(iconn->fedId()) ) { continue; }

      // Retrieve digis for given FED key and check if found
      vector< edm::DetSet<SiStripRawDigi> >::const_iterator digis = raw.find( fed_key ); 
      if ( digis != raw.end() ) { 
	if ( tasks_[iconn->fedId()][iconn->fedCh()] ) { 
	  tasks_[iconn->fedId()][iconn->fedCh()]->fillHistograms( *summary, *digis );
	} else {
	  stringstream ss;
	  ss << "[SiStripCommissioningSource::" << __func__ << "]"
	     << " Unable to find CommissioningTask object with FED key " 
	     << hex << setfill('0') << setw(8) << fed_key << dec
	     << " and FED id/ch " 
	     << iconn->fedId() << "/"
	     << iconn->fedCh()
	     << " Unable to fill histograms!";
	  edm::LogWarning(mlDqmSource_) << ss.str();
	}
      }
      
    } // fed channel loop
  } // fed id loop

}

// -----------------------------------------------------------------------------
//
void SiStripCommissioningSource::createTask( const SiStripEventSummary* const summary ) {
  
  // Set commissioning task to default ("undefined") value
  task_ = sistrip::UNDEFINED_TASK;
  
  // Retrieve commissioning task from EventSummary
  if ( summary ) { task_ = summary->task(); } 
  else { 
    task_ = sistrip::UNKNOWN_TASK; 
    stringstream ss;
    ss << "[SiStripCommissioningSource::" << __func__ << "]"
       << " NULL pointer to SiStripEventSummary!" 
       << " Check SiStripEventSummary is found/present in Event";
    edm::LogWarning(mlDqmSource_) << ss.str();
  } 
  
  // Override task with ParameterSet configurable (if defined)
  sistrip::Task configurable = SiStripHistoNamingScheme::task( taskConfigurable_ );
  if ( configurable != sistrip::UNDEFINED_TASK &&
       configurable != sistrip::UNKNOWN_TASK ) { task_ = configurable; }
  
  // Create ME (string) that identifies commissioning task
  dqm()->setCurrentFolder( sistrip::root_ );
  string task_str = SiStripHistoNamingScheme::task( task_ );
  dqm()->bookString( sistrip::commissioningTask_ + sistrip::sep_ + task_str, task_str ); 
  
  // Check commissioning task is known / defined
  if ( task_ == sistrip::UNKNOWN_TASK ||
       task_ == sistrip::UNDEFINED_TASK ) {
    stringstream ss;
    ss << "[SiStripCommissioningSource::" << __func__ << "]"
       << " Unexpected CommissioningTask: " << SiStripHistoNamingScheme::task( task_ )
       << " Unexpected value found in SiStripEventSummary and/or cfg file"
       << " If SiStripEventSummary is not present in Event, check 'CommissioningTask' configurable in cfg file";
    edm::LogWarning(mlDqmSource_) << ss.str();
    return; 
  } else {
    stringstream ss;
    ss << "[SiStripCommissioningSource::" << __func__ << "]"
       << " Identified CommissioningTask from EventSummary to be: " 
       << SiStripHistoNamingScheme::task( task_ );
    LogTrace(mlDqmSource_) << ss.str();
  }
  
  // Check if commissioning task is FED cabling 
  if ( task_ == sistrip::FED_CABLING ) { cablingTask_ = true; }
  else { cablingTask_ = false; }

  if ( !cablingTask_ ) { createTasks(); }
  else { createCablingTasks(); }
  tasksExist_ = true;

}

// -----------------------------------------------------------------------------
//
void SiStripCommissioningSource::createCablingTasks() {
  
  // Iterate through FEC cabling and create commissioning task objects
  for ( vector<SiStripFecCrate>::const_iterator icrate = fecCabling_->crates().begin(); icrate != fecCabling_->crates().end(); icrate++ ) {
    for ( vector<SiStripFec>::const_iterator ifec = icrate->fecs().begin(); ifec != icrate->fecs().end(); ifec++ ) {
      for ( vector<SiStripRing>::const_iterator iring = ifec->rings().begin(); iring != ifec->rings().end(); iring++ ) {
	for ( vector<SiStripCcu>::const_iterator iccu = iring->ccus().begin(); iccu != iring->ccus().end(); iccu++ ) {
	  for ( vector<SiStripModule>::const_iterator imodule = iccu->modules().begin(); imodule != iccu->modules().end(); imodule++ ) {
	      
	    // Set working directory prior to booking histograms
	    SiStripFecKey::Path path( icrate->fecCrate(), 
				      ifec->fecSlot(), 
				      iring->fecRing(), 
				      iccu->ccuAddr(), 
				      imodule->ccuChan() );
	    string dir = SiStripHistoNamingScheme::controlPath( path );
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
	      uint32_t key = SiStripFecKey::key( icrate->fecCrate(), 
						 ifec->fecSlot(), 
						 iring->fecRing(), 
						 iccu->ccuAddr(), 
						 imodule->ccuChan(),
						 imodule->lldChannel(ipair) );
		
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
		} else if ( task_ == sistrip::UNDEFINED_TASK ) { 
                  edm::LogWarning(mlDqmSource_)
		    << "[SiStripCommissioningSource::" << __func__ << "]"
		    << " Undefined CommissioningTask" 
		    << " Unable to create FedCablingTask object!";
		} else if ( task_ == sistrip::UNKNOWN_TASK ) { 
                  edm::LogWarning(mlDqmSource_)
		    << "[SiStripCommissioningSource::" << __func__ << "]"
		    << " Unknown CommissioningTask" 
		    << " Unable to create FedCablingTask object!";
                } else { 
                  edm::LogWarning(mlDqmSource_)
		    << "[SiStripCommissioningSource::" << __func__ << "]"
		    << " Unexpected CommissioningTask: " 
		    << SiStripHistoNamingScheme::task( task_ )
		    << " Unable to create FedCablingTask object!";
                }
		
		// Check if key is found and, if so, book histos and set update freq
		if ( cablingTasks_.find( key ) != cablingTasks_.end() ) {
		  if ( cablingTasks_[key] ) {
		    cablingTasks_[key]->bookHistograms(); 
		    cablingTasks_[key]->updateFreq(1); //@@ hardwired to update every event!!! 
		    stringstream ss;
		    ss << "[SiStripCommissioningSource::" << __func__ << "]"
		       << " Booking histograms for '" << cablingTasks_[key]->myName()
		       << "' object with key 0x" << hex << setfill('0') << setw(8) << key << dec
		       << " in directory " << dir;
		    LogTrace(mlDqmSource_) << ss.str();
		  } else {
		    stringstream ss;
		    ss << "[SiStripCommissioningSource::" << __func__ << "]"
		       << " NULL pointer to CommissioningTask for key 0x"
		       << hex << setfill('0') << setw(8) << key << dec
		       << " in directory " << dir 
		       << " Unable to book histograms!";
		    edm::LogWarning(mlDqmSource_) << ss.str();
		  }
		} else {
		  stringstream ss;
		  ss << "[SiStripCommissioningSource::" << __func__ << "]"
		     << " Unable to find CommissioningTask for key 0x"
		     << hex << setfill('0') << setw(8) << key << dec
		     << " in directory " << dir
		     << " Unable to book histograms!";
		  edm::LogWarning(mlDqmSource_) << ss.str();
		}
	      
	      } else {
		stringstream ss;
		ss << "[SiStripCommissioningSource::" << __func__ << "]"
		   << " CommissioningTask object already exists for key 0x"
		   << hex << setfill('0') << setw(8) << key << dec
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

  // Iterate through FED ids and channels 
  vector<uint16_t>::const_iterator ifed = fedCabling_->feds().begin();
  for ( ; ifed != fedCabling_->feds().end(); ifed++ ) {
    
    // Iterate through connected FED channels
    const vector<FedChannelConnection>& conns = fedCabling_->connections(*ifed);
    vector<FedChannelConnection>::const_iterator iconn = conns.begin();
    for ( ; iconn != conns.end(); iconn++ ) {
      
      // Create FED key and check if non-zero
      uint32_t fed_key = SiStripFedKey::key( iconn->fedId(), iconn->fedCh() );
      if ( !(iconn->fedId()) ) { continue; }
      
      // Set working directory prior to booking histograms 
      SiStripFecKey::Path path( iconn->fecCrate(), 
				iconn->fecSlot(), 
				iconn->fecRing(), 
				iconn->ccuAddr(), 
				iconn->ccuChan() );
      string dir = SiStripHistoNamingScheme::controlPath( path );
      dqm()->setCurrentFolder( dir );
      
      // Create commissioning task objects
      if ( !tasks_[iconn->fedId()][iconn->fedCh()] ) { 
	if ( task_ == sistrip::APV_TIMING ) { tasks_[iconn->fedId()][iconn->fedCh()] = new ApvTimingTask( dqm(), *iconn ); } 
	else if ( task_ == sistrip::FED_TIMING ) { tasks_[iconn->fedId()][iconn->fedCh()] = new FedTimingTask( dqm(), *iconn ); }
	else if ( task_ == sistrip::OPTO_SCAN ) { tasks_[iconn->fedId()][iconn->fedCh()] = new OptoScanTask( dqm(), *iconn ); }
	else if ( task_ == sistrip::VPSP_SCAN ) { tasks_[iconn->fedId()][iconn->fedCh()] = new VpspScanTask( dqm(), *iconn ); }
	else if ( task_ == sistrip::PEDESTALS ) { tasks_[iconn->fedId()][iconn->fedCh()] = new PedestalsTask( dqm(), *iconn ); }
	else if ( task_ == sistrip::DAQ_SCOPE_MODE ) { tasks_[iconn->fedId()][iconn->fedCh()] = new DaqScopeModeTask( dqm(), *iconn ); }
	else if ( task_ == sistrip::UNDEFINED_TASK ) { 
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
	  stringstream ss;
	  ss << "[SiStripCommissioningSource::" << __func__ << "]"
	     << " Booking histograms for '" << tasks_[iconn->fedId()][iconn->fedCh()]->myName()
	     << "' object with key 0x" << hex << setfill('0') << setw(8) << fed_key << dec
	     << " in directory " << dir;
	  LogTrace(mlDqmSource_) << ss.str();
	} else {
	  stringstream ss;
	  ss << "[SiStripCommissioningSource::" << __func__ << "]"
	     << " NULL pointer to CommissioningTask for key 0x"
	     << hex << setfill('0') << setw(8) << fed_key << dec
	     << " in directory " << dir 
	     << " Unable to book histograms!";
	  edm::LogWarning(mlDqmSource_) << ss.str();
	}
	
      } else {
	stringstream ss;
	ss << "[SiStripCommissioningSource::" << __func__ << "]"
	   << " CommissioningTask object already exists for key 0x"
	   << hex << setfill('0') << setw(8) << fed_key << dec
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
  
  vector<uint16_t>::const_iterator ifed = fedCabling_->feds().begin(); 
  for ( ; ifed != fedCabling_->feds().end(); ifed++ ) { 
    const vector<FedChannelConnection>& conns = fedCabling_->connections(*ifed);
    vector<FedChannelConnection>::const_iterator iconn = conns.begin();
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
