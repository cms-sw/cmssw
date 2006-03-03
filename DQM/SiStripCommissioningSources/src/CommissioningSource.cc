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
// conditions
#include "CondFormats/DataRecord/interface/SiStripFedCablingRcd.h"
#include "CondFormats/SiStripObjects/interface/SiStripFedCabling.h"
// calibrations
#include "CalibFormats/SiStripObjects/interface/SiStripFecCabling.h"
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
// Retrieve DQM interface, control cabling and "control view" utlity
// class, create histogram directory structure and generate "reverse"
// control cabling.
void CommissioningSource::beginJob( const edm::EventSetup& setup ) {
  cout << "[CommissioningSource::beginJob]" << endl;

  dqm_ = edm::Service<DaqMonitorBEInterface>().operator->();

  edm::ESHandle<SiStripFedCabling> fed_cabling;
  setup.get<SiStripFedCablingRcd>().get( fed_cabling );

  SiStripControlDirPath directory;
  
  SiStripFecCabling* fec_cabling = new SiStripFecCabling( *fed_cabling );
  const vector<SiStripFec>& fecs = fec_cabling->fecs();
  for ( vector<SiStripFec>::const_iterator ifec = fecs.begin(); ifec != fecs.end(); ifec++ ) {
    const vector<SiStripRing>& rings = (*ifec).rings();
    for ( vector<SiStripRing>::const_iterator iring = rings.begin(); iring != rings.end(); iring++ ) {
      const vector<SiStripCcu>& ccus = (*iring).ccus();
      for ( vector<SiStripCcu>::const_iterator iccu = ccus.begin(); iccu != ccus.end(); iccu++ ) {
	const vector<SiStripModule>& modules = (*iccu).modules();
	for ( vector<SiStripModule>::const_iterator imodule = modules.begin(); imodule != modules.end(); imodule++ ) {
	  string dir = directory.path( (*ifec).fecSlot(),
				       (*iring).fecRing(),
				       (*iccu).ccuAddr(),
				       (*imodule).ccuChan() );
	  dqm_->setCurrentFolder( dir );
	  CommissioningTask* task = createTask( *imodule );
	  if ( task ) { task->updateFreq( updateFreq_ ); }
	} 
      }
    }
  }
  
}

// -----------------------------------------------------------------------------
//
CommissioningTask* CommissioningSource::createTask( const SiStripModule& module ) {
  cout << "[CommissioningSource::createTask]" << endl;

  if ( task_ == "PHYSICS" ) {
    if ( tasks_.find( module.dcuId() ) == tasks_.end() ) { 
      tasks_[module.dcuId()] = new PhysicsTask( dqm_, module );
    } else {
      cerr << "[CommissioningSource::createTask]"
	   << " PhysicsTask already exists for DcuId "
	   << module.dcuId() << endl;
    }
  } else if ( task_ == "PEDESTALS" ) {
    if ( tasks_.find( module.dcuId() ) == tasks_.end() ) { 
      tasks_[module.dcuId()] = new PedestalsTask( dqm_, module );
    } else {
      cerr << "[CommissioningSource::createTask]"
	   << " PedestalsTask already exists for DcuId "
	   << module.dcuId() << endl;
    }
  } else {
    cerr << "[CommissioningSource::createTask]"
	 << " Unknown commissioning task! " << task_ << endl;
    return 0;
  }

  return tasks_[module.dcuId()];

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
  
  edm::Handle<StripDigiCollection> collection;
  event.getByType( collection );

//   edm::Handle<SiStripEventInfo> info;
//   event.getByType( info );
  
  vector<unsigned int> ids;
  (*collection).detIDs( ids ); //@@ method name incorrect! 
  
  for ( vector<unsigned int>::iterator id = ids.begin(); id != ids.end(); id++ ) {
    unsigned short fed_id = ((*id)>>16) & 0xFFFFFFFF;
    unsigned short fed_ch = (*id) & 0xFFFFFFFF;
    unsigned int dcu_id = fed_cabling->connection( fed_id, fed_ch ).dcuId();
    if ( tasks_.find(dcu_id) != tasks_.end() ) {
      vector<StripDigi> digis;
      (*collection).digis( *id, digis );
      tasks_[dcu_id]->fillHistograms( digis/*, info*/ );
    }
  }
  
}

// -----------------------------------------------------------------------------
// Define plug-in
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(CommissioningSource)
