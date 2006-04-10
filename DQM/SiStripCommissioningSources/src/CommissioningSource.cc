#include "DQM/SiStripCommissioningSources/interface/CommissioningSource.h"
// edm
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
// dqm
#include "DQMServices/Core/interface/DaqMonitorBEInterface.h"
#include "DQM/SiStripCommon/interface/SiStripHistoNamingScheme.h"
#include "DQM/SiStripCommon/interface/SiStripGenerateKey.h"
// conditions
#include "CondFormats/DataRecord/interface/SiStripFedCablingRcd.h"
#include "CondFormats/SiStripObjects/interface/SiStripFedCabling.h"
// calibrations
#include "CalibFormats/SiStripObjects/interface/SiStripFecCabling.h"
// data formats
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/SiStripDigi/interface/SiStripDigi.h"
// tasks
//#include "DQM/SiStripCommissioningSources/interface/PhysicsTask.h"
#include "DQM/SiStripCommissioningSources/interface/PedestalsTask.h"
#include "DQM/SiStripCommissioningSources/interface/ApvTimingTask.h"
#include "DQM/SiStripCommissioningSources/interface/OptoBiasAndGainScanTask.h"
//#include "DQM/SiStripCommissioningSources/interface/VpspScanTask.h"
// std, utilities
#include <boost/cstdint.hpp>
#include <memory>
#include <vector>

// -----------------------------------------------------------------------------
//
CommissioningSource::CommissioningSource( const edm::ParameterSet& pset ) :
  inputModuleLabel_( pset.getParameter<string>( "InputModuleLabel" ) ),
  dqm_(0),
  task_( pset.getUntrackedParameter<string>("CommissioningTask","UNKNOWN") ),
  tasks_(),
  updateFreq_( pset.getUntrackedParameter<int>("HistoUpdateFreq",100) )
{
  edm::LogInfo("CommissioningSource") << "[CommissioningSource::CommissioningSource] Constructing object...";
}

// -----------------------------------------------------------------------------
//
CommissioningSource::~CommissioningSource() {
  edm::LogInfo("CommissioningSource") << "[CommissioningSource::~CommissioningSource] Destructing object...";
}

// -----------------------------------------------------------------------------
// Retrieve DQM interface, control cabling and "control view" utility
// class, create histogram directory structure and generate "reverse"
// control cabling.
void CommissioningSource::beginJob( const edm::EventSetup& setup ) {
  edm::LogInfo("Commissioning") << "[CommissioningSource::beginJob]";
  createTask( setup );
}

// -----------------------------------------------------------------------------
//
void CommissioningSource::endJob() {
  edm::LogInfo("Commissioning") << "[CommissioningSource::endJob]";
  for ( TaskMap::iterator itask = tasks_.begin(); itask != tasks_.end(); itask++ ) { 
    if ( itask->second ) { itask->second->updateHistograms(); }
  }
  //if ( dqm_ ) { dqm_->showDirStructure(); }
  if ( dqm_ ) { dqm_->save("test.root"); }
  for ( TaskMap::iterator itask = tasks_.begin(); itask != tasks_.end(); itask++ ) { 
    if ( itask->second ) { delete itask->second; }
  }
  task_.clear();
}

// -----------------------------------------------------------------------------
//
void CommissioningSource::analyze( const edm::Event& event, 
				   const edm::EventSetup& setup ) {
  LogDebug("Commissioning") << "[CommissioningSource::analyze]";

  edm::ESHandle<SiStripFedCabling> fed_cabling;
  setup.get<SiStripFedCablingRcd>().get( fed_cabling );
  
  edm::Handle<SiStripEventSummary> summary;
  event.getByLabel( inputModuleLabel_, summary );
  //createTask( setup, summary->task() );

  edm::Handle< edm::DetSetVector<SiStripRawDigi> > raw;
  //edm::Handle< edm::DetSetVector<SiStripDigi> > zs;
  
  if ( summary->fedReadoutMode() == SiStripEventSummary::VIRGIN_RAW ) {
    event.getByLabel( inputModuleLabel_, "VirginRaw", raw );
  } else if ( summary->fedReadoutMode() == SiStripEventSummary::SCOPE_MODE ) {
    event.getByLabel( inputModuleLabel_, "ScopeMode", raw );
  } else if ( summary->fedReadoutMode() == SiStripEventSummary::ZERO_SUPPR ) {
    //event.getByLabel( inputModuleLabel_, "ZeroSuppr", zs );
  } else if ( summary->fedReadoutMode() == SiStripEventSummary::PROC_RAW ) {
    event.getByLabel( inputModuleLabel_, "ProcRaw", raw );
  } else {
    edm::LogError("CommissioningSource") << "[CommissioningSource::analyze]"
					 << " Unknown FED readout mode!";
  }
  
  vector<uint16_t>::const_iterator ifed;
  for ( ifed = fed_cabling->feds().begin(); ifed != fed_cabling->feds().end(); ifed++ ) {
    for ( uint16_t ichan = 0; ichan < 96; ichan++ ) {
      uint32_t fed_key = SiStripGenerateKey::fed( *ifed, ichan );
      if ( fed_key ) { 
	if ( tasks_.find(fed_key) != tasks_.end() ) { 
	  vector< edm::DetSet<SiStripRawDigi> >::const_iterator digis = raw->find( fed_key );
	  if ( digis != raw->end() ) { 
	    tasks_[fed_key]->fillHistograms( *summary, *digis );
	  }
	}
      }
    }
  }
  
}

// -----------------------------------------------------------------------------
//
bool CommissioningSource::createTask( const edm::EventSetup& setup,
				      SiStripEventSummary::Task task ) {
  LogDebug("Commissioning") << "[CommissioningSource::createTask]";
  
  // Check DQM service is available
  dqm_ = edm::Service<DaqMonitorBEInterface>().operator->();
  if ( !dqm_ ) { 
    edm::LogError("Commissioning") << "[CommissioningSource::createTask] Null pointer to DQM interface!"; 
    return false; 
  }

  // Check commissioning task is known
  if ( task == SiStripEventSummary::UNKNOWN_TASK && task_ == "UNKNOWN" ) {
    edm::LogError("Commissioning") << "[CommissioningSource::createTask] Unknown commissioning task!"; 
    return false; 
  }

  // Retrieve FED cabling
  edm::ESHandle<SiStripFedCabling> fed_cabling;
  setup.get<SiStripFedCablingRcd>().get( fed_cabling );

  // Iterate through FEC cabling and create commissioning task objects
  SiStripFecCabling* fec_cabling = new SiStripFecCabling( *fed_cabling );
  for ( vector<SiStripFec>::const_iterator ifec = fec_cabling->fecs().begin(); ifec != fec_cabling->fecs().end(); ifec++ ) {
    for ( vector<SiStripRing>::const_iterator iring = (*ifec).rings().begin(); iring != (*ifec).rings().end(); iring++ ) {
      for ( vector<SiStripCcu>::const_iterator iccu = (*iring).ccus().begin(); iccu != (*iring).ccus().end(); iccu++ ) {
	for ( vector<SiStripModule>::const_iterator imodule = (*iccu).modules().begin(); imodule != (*iccu).modules().end(); imodule++ ) {
	  string dir = SiStripHistoNamingScheme::controlPath( 0, // FEC crate 
							      (*ifec).fecSlot(),
							      (*iring).fecRing(),
							      (*iccu).ccuAddr(),
							      (*imodule).ccuChan() );
// 	  SiStripHistoNamingScheme::ControlPath path = SiStripHistoNamingScheme::controlPath( dir );
// 	  edm::LogInfo("rob") << dir << " " 
// 			      << path.fecCrate_ << " " 
// 			      << path.fecSlot_ << " " 
// 			      << path.fecRing_ << " " 
// 			      << path.ccuAddr_ << " "  
// 			      << path.ccuChan_; 
	  dqm_->setCurrentFolder( dir );
	  map< uint16_t, pair<uint16_t,uint16_t> >::const_iterator iconn;
	  for ( iconn = imodule->fedChannels().begin(); iconn != imodule->fedChannels().end(); iconn++ ) {
	    if ( !(iconn->second.first) ) { continue; } 
	    // Retrieve FED id/ch in order to create key for task map
	    FedChannelConnection conn = fed_cabling->connection( iconn->second.first,
								 iconn->second.second );
	    uint32_t fed_key = SiStripGenerateKey::fed( conn.fedId(), conn.fedCh() );
	    // Create commissioning task objects
	    if ( tasks_.find( fed_key ) == tasks_.end() ) {
	      if      ( task_ == "PEDESTALS" )  { tasks_[fed_key] = new PedestalsTask( dqm_, conn ); }
	      else if ( task_ == "APV_TIMING" ) { tasks_[fed_key] = new ApvTimingTask( dqm_, conn ); }
	      else if ( task_ == "OPTO_SCAN" )  { tasks_[fed_key] = new OptoBiasAndGainScanTask( dqm_, conn ); }
	      //	      else if ( task_ == "VPSP_SCAN" )  { tasks_[fed_key] = new VpspScanTask( dqm_, conn ); }
	      //else if ( task_ == "PHYSICS" )    { tasks_[fed_key] = new PhysicsTask( dqm_, conn ); }
	      else if ( task_ != "UNKNOWN" ) {
		//  Use data stream to determine which task objects are created!
		if      ( task == SiStripEventSummary::PEDESTALS )  { tasks_[fed_key] = new PedestalsTask( dqm_, conn ); }
		else if ( task == SiStripEventSummary::APV_TIMING ) { tasks_[fed_key] = new ApvTimingTask( dqm_, conn ); }
		else if ( task == SiStripEventSummary::OPTO_SCAN )  { tasks_[fed_key] = new OptoBiasAndGainScanTask( dqm_, conn ); }
		//		else if ( task == SiStripEventSummary::VPSP_SCAN )  { tasks_[fed_key] = new VpspScanTask( dqm_, conn ); }
		//else if ( task == SiStripEventSummary::PHYSICS )    { tasks_[fed_key] = new PhysicsTask( dqm_, conn ); }
		else if ( task == SiStripEventSummary::UNKNOWN_TASK ) {
		  edm::LogError("Commissioning") << "[CommissioningSource::createTask]"
						 << " Unknown commissioning task in data stream! " << task_;
		}
	      } else {
		edm::LogError("Commissioning") << "[CommissioningSource::createTask]"
					       << " Unknown commissioning task in .cfg file! " << task_;
	      }
	      // Check if FED key found and, if so, book histos and set update freq
	      if ( tasks_.find( fed_key ) != tasks_.end() ) {
		tasks_[fed_key]->bookHistograms(); 
		tasks_[fed_key]->updateFreq( updateFreq_ ); 
	      }
	    } else {
	      edm::LogError("Commissioning") << "[CommissioningSource::createTask]"
					     << " PhysicsTask already exists for FED id/channel "
					     << conn.fedId() << "/" << conn.fedCh(); 
	    }
	  }
	}
      }
    }
  }
  edm::LogInfo("Commissioning") << "[CommissioningSource]"
				<< " Number of task objects created: " << tasks_.size();
  return true;
}

