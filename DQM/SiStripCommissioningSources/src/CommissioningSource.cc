#include "DQM/SiStripCommissioningSources/interface/CommissioningSource.h"
// edm
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
// dqm
#include "DQMServices/Core/interface/DaqMonitorBEInterface.h"
#include "DQM/SiStripCommon/interface/SiStripControlDirPath.h"
#include "DQM/SiStripCommon/interface/SiStripGenerateKey.h"
// conditions
#include "CondFormats/DataRecord/interface/SiStripFedCablingRcd.h"
#include "CondFormats/SiStripObjects/interface/SiStripFedCabling.h"
// calibrations
#include "CalibFormats/SiStripObjects/interface/SiStripFecCabling.h"
// data formats
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/SiStripDigi/interface/SiStripDigi.h"
#include "DataFormats/SiStripDigi/interface/SiStripEventSummary.h"
// tasks
#include "DQM/SiStripCommissioningSources/interface/PhysicsTask.h"
#include "DQM/SiStripCommissioningSources/interface/PedestalsTask.h"
// std, utilities
#include <boost/cstdint.hpp>
#include <memory>
#include <vector>

// -----------------------------------------------------------------------------
//
CommissioningSource::CommissioningSource( const edm::ParameterSet& pset ) :
  dqm_(0),
  task_( pset.getUntrackedParameter<string>("CommissioningTask","PHYSICS") ),
  tasks_(),
  updateFreq_( pset.getUntrackedParameter<int>("HistoUpdateFreq",100) )
{
  cout << "[CommissioningSource::CommissioningSource]" 
       << " Constructing object..." << endl;
}

// -----------------------------------------------------------------------------
//
CommissioningSource::~CommissioningSource() {
  cout << "[CommissioningSource::~CommissioningSource]"
       << " Destructing object..." << endl;
}

// -----------------------------------------------------------------------------
// Retrieve DQM interface, control cabling and "control view" utility
// class, create histogram directory structure and generate "reverse"
// control cabling.
void CommissioningSource::beginJob( const edm::EventSetup& setup ) {
  cout << "[CommissioningSource::beginJob]" << endl;

  dqm_ = edm::Service<DaqMonitorBEInterface>().operator->();

  edm::ESHandle<SiStripFedCabling> fed_cabling;
  setup.get<SiStripFedCablingRcd>().get( fed_cabling );

  SiStripControlDirPath directory;
  
  SiStripFecCabling* fec_cabling = new SiStripFecCabling( *fed_cabling );
  for ( vector<SiStripFec>::const_iterator ifec = fec_cabling->fecs().begin(); ifec != fec_cabling->fecs().end(); ifec++ ) {
    for ( vector<SiStripRing>::const_iterator iring = (*ifec).rings().begin(); iring != (*ifec).rings().end(); iring++ ) {
      for ( vector<SiStripCcu>::const_iterator iccu = (*iring).ccus().begin(); iccu != (*iring).ccus().end(); iccu++ ) {
	for ( vector<SiStripModule>::const_iterator imodule = (*iccu).modules().begin(); imodule != (*iccu).modules().end(); imodule++ ) {
	  string dir = directory.path( (*ifec).fecSlot(),
				       (*iring).fecRing(),
				       (*iccu).ccuAddr(),
				       (*imodule).ccuChan() );
	  dqm_->setCurrentFolder( dir );
	  map< uint16_t, pair<uint16_t,uint16_t> >::const_iterator iconn;
	  for ( iconn = imodule->fedChannels().begin(); iconn != imodule->fedChannels().end(); iconn++ ) {
	    if ( !(iconn->second.first) ) { continue; } 
	    FedChannelConnection conn = fed_cabling->connection( iconn->second.first,
								 iconn->second.second );
	    CommissioningTask* task = createTask( conn );
	    if ( task ) { task->updateFreq( updateFreq_ ); }
	  }
	}
      }
    }
  }
  
}

// -----------------------------------------------------------------------------
//
CommissioningTask* CommissioningSource::createTask( const FedChannelConnection& conn ) {
  cout << "[CommissioningSource::createTask]" << endl;
  uint32_t fed_key = SiStripGenerateKey::fed( conn.fedId(), conn.fedCh() );
  if ( task_ == "PHYSICS" ) {
    if ( tasks_.find( fed_key ) == tasks_.end() ) { 
      tasks_[fed_key] = new PhysicsTask( dqm_, conn );
    } else {
      cerr << "[CommissioningSource::createTask]"
	   << " PhysicsTask already exists for FED id/channel "
	   << fed_key << endl; //@@ extract FED id/ch
    }
  } else if ( task_ == "PEDESTALS" ) {
    if ( tasks_.find( fed_key ) == tasks_.end() ) { 
      tasks_[fed_key] = new PedestalsTask( dqm_, conn );
    } else {
      cerr << "[CommissioningSource::createTask]"
	   << " PedestalsTask already exists for FED id/channel "
	   << fed_key << endl; //@@ extract FED id/ch
    }
  } else {
    cerr << "[CommissioningSource::createTask]"
	 << " Unknown commissioning task! " << task_ << endl;
    return 0;
  }
  return tasks_[fed_key];
}

// -----------------------------------------------------------------------------
//
void CommissioningSource::endJob() {
  cout << "[CommissioningSource::endJob]" << endl;

  TaskMap::iterator iter;
  for ( iter = tasks_.begin(); iter != tasks_.end(); iter++ ) {
    delete (*iter).second;
  }

}

// -----------------------------------------------------------------------------
//
void CommissioningSource::analyze( const edm::Event& event, 
				   const edm::EventSetup& setup ) {
  cout << "[CommissioningSource::analyze]" << endl;

  edm::ESHandle<SiStripFedCabling> fed_cabling;
  setup.get<SiStripFedCablingRcd>().get( fed_cabling );
  
  edm::Handle<SiStripEventSummary> summary;
  event.getByType( summary );

  edm::Handle< edm::DetSetVector<SiStripRawDigi> > collection;
  event.getByType( collection );
  
  vector<uint16_t>::const_iterator ifed;
  for ( ifed = fed_cabling->feds().begin(); ifed != fed_cabling->feds().end(); ifed++ ) {
    for ( uint16_t ichan = 0; ichan < 96; ichan++ ) {
      uint32_t fed_key = SiStripGenerateKey::fed( *ifed, ichan );
      if ( fed_key ) {
	if ( tasks_.find(fed_key) != tasks_.end() ) {
	  vector< edm::DetSet<SiStripRawDigi> >::const_iterator digis = collection->find( fed_key );
	  tasks_[fed_key]->fillHistograms( *summary, *digis );
	}
      }
    }
  }
  
}

