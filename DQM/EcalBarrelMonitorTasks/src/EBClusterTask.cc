/*
 * \file EBClusterTask.cc
 *
 * $Date: 2006/11/01 07:59:16 $
 * $Revision: 1.4 $
 * \author G. Della Ricca
 *
*/

#include <DQM/EcalBarrelMonitorTasks/interface/EBClusterTask.h>

EBClusterTask::EBClusterTask(const ParameterSet& ps){

  init_ = false;

  for (int i = 0; i < 36 ; i++) {
    mePedMapG12_[i] = 0;
  }

}

EBClusterTask::~EBClusterTask(){

}

void EBClusterTask::beginJob(const EventSetup& c){

  ievt_ = 0;

  DaqMonitorBEInterface* dbe = 0;

  // get hold of back-end interface
  dbe = Service<DaqMonitorBEInterface>().operator->();

  if ( dbe ) {
    dbe->setCurrentFolder("EcalBarrel/EBClusterTask");
    dbe->rmdir("EcalBarrel/EBClusterTask");
  }

}

void EBClusterTask::setup(void){

  init_ = true;

  Char_t histo[200];

  DaqMonitorBEInterface* dbe = 0;

  // get hold of back-end interface
  dbe = Service<DaqMonitorBEInterface>().operator->();

  if ( dbe ) {
    dbe->setCurrentFolder("EcalBarrel/EBClusterTask");

    for (int i = 0; i < 36 ; i++) {
      sprintf(histo, "EBPOT pedestal SM%02d G12", i+1);
      mePedMapG12_[i] = dbe->bookProfile2D(histo, histo, 85, 0., 85., 20, 0., 20., 4096, 0., 4096., "s");
    }

  }

}

void EBClusterTask::cleanup(void){

  DaqMonitorBEInterface* dbe = 0;

  // get hold of back-end interface
  dbe = Service<DaqMonitorBEInterface>().operator->();

  if ( dbe ) {
    dbe->setCurrentFolder("EcalBarrel/EBClusterTask");

    for ( int i = 0; i < 36; i++ ) {
      if ( mePedMapG12_[i] ) dbe->removeElement( mePedMapG12_[i]->getName() );
      mePedMapG12_[i] = 0;
    }

  }

  init_ = false;

}

void EBClusterTask::endJob(void){

  LogInfo("EBClusterTask") << "analyzed " << ievt_ << " events";

  if ( init_ ) this->cleanup();

}

void EBClusterTask::analyze(const Event& e, const EventSetup& c){

  if ( ! init_ ) this->setup();

  ievt_++;

  try {

    Handle<BasicClusterCollection> bclusters;
    e.getByLabel("islandBasicClusterProducer", "islandBarrelBasicClusterCollection", bclusters);

    int nbcc = bclusters->size();
    LogDebug("EBClusterTask") << "event " << ievt_ << " basic cluster collection size " << nbcc;

    for ( BasicClusterCollection::const_iterator bclusterItr = bclusters->begin(); bclusterItr != bclusters->end(); ++bclusterItr ) {

      BasicCluster bcluster = *(bclusterItr);

      cout << "basic cluster" << endl;
      cout << "number of crystals " << bcluster.getHitsByDetId().size() << endl;
      cout << "energy " << bcluster.energy() << endl;
      cout << "eta " << bcluster.eta() << endl;
      cout << "phi " << bcluster.phi() << endl;

    }

  } catch ( std::exception& ex ) {
    LogDebug("EBClusterTask") << " BasicClusterCollection not in event.";
  }

  try {

    Handle<SuperClusterCollection> sclusters;
    e.getByLabel("islandSuperClusterProducer", "islandBarrelSuperClusterCollection", sclusters);

    int nscc = sclusters->size();
    LogDebug("EBClusterTask") << "event " << ievt_ << " super cluster collection size " << nscc; 

    for ( SuperClusterCollection::const_iterator sclusterItr = sclusters->begin(); sclusterItr != sclusters->end(); ++sclusterItr ) {
  
      SuperCluster scluster = *(sclusterItr);

      cout << "super cluster" << endl;
      cout << "number of crystals " << scluster.getHitsByDetId().size() << endl;
      cout << "number of basic clusters " << (1+scluster.clustersSize()) << endl;
      cout << "energy " << scluster.energy() << endl;
      cout << "eta " << scluster.eta() << endl;
      cout << "phi " << scluster.phi() << endl;
//      cout << "eMax " << scluster.eMax() << endl;
//      cout << "e2x2 " << scluster.e2x2() << endl;
//      cout << "e3x3 " << scluster.e3x3() << endl;
//      cout << "e5x5 " << scluster.e5x5() << endl;

    }

  } catch ( std::exception& ex ) {
    LogDebug("EBClusterTask") << " SuperClusterCollection not in event.";
  }

}

