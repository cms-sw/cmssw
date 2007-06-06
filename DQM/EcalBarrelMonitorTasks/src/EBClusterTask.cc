/*
 * \file EBClusterTask.cc
 *
 * $Date: 2006/11/03 10:07:17 $
 * $Revision: 1.7 $
 * \author G. Della Ricca
 *
*/

#include <DQM/EcalBarrelMonitorTasks/interface/EBClusterTask.h>

using namespace cms;
using namespace edm;
using namespace std;
using namespace reco;

EBClusterTask::EBClusterTask(const ParameterSet& ps){

  init_ = false;

  meBEne_ = 0;
  meBNum_ = 0;
  meBCry_ = 0;

  meSEneMap_ = 0;
  meSNumMap_ = 0;

  meSEne_ = 0;
  meSNum_ = 0;

  meSSiz_ = 0;

  meSEneMap_ = 0;
  meSNumMap_ = 0;

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

    sprintf(histo, "EBCLT basic cluster energy");
    meBEne_ = dbe->book1D(histo, histo, 100, 0., 500.);

    sprintf(histo, "EBCLT basic cluster number");
    meBNum_ = dbe->book1D(histo, histo, 100, 0., 100.);

    sprintf(histo, "EBCLT basic cluster crystals");
    meBCry_ = dbe->book1D(histo, histo, 100, 0., 100.);

    sprintf(histo, "EBCLT basic cluster energy map");
    meBEneMap_ = dbe->bookProfile2D(histo, histo, 170, -1.479, 1.479, 360, 0., 2*M_PI, 100, 0., 500., "s");

    sprintf(histo, "EBCLT basic cluster number map");
    meBNumMap_ = dbe->book2D(histo, histo, 170, -1.479, 1.479, 360, 0., 2*M_PI);

    sprintf(histo, "EBCLT super cluster energy");
    meSEne_ = dbe->book1D(histo, histo, 100, 0., 500.);

    sprintf(histo, "EBCLT super cluster number");
    meSNum_ = dbe->book1D(histo, histo, 100, 0., 100.);

    sprintf(histo, "EBCLT super cluster size");
    meSSiz_ = dbe->book1D(histo, histo, 20, 0., 20.);

    sprintf(histo, "EBCLT super cluster energy map");
    meSEneMap_ = dbe->bookProfile2D(histo, histo, 170, -1.479, 1.479, 360, 0., 2*M_PI, 100, 0., 500., "s");

    sprintf(histo, "EBCLT super cluster number map");
    meSNumMap_ = dbe->book2D(histo, histo, 170, -1.479, 1.479, 360, 0., 2*M_PI);

  }

}

void EBClusterTask::cleanup(void){

  DaqMonitorBEInterface* dbe = 0;

  // get hold of back-end interface
  dbe = Service<DaqMonitorBEInterface>().operator->();

  if ( dbe ) {
    dbe->setCurrentFolder("EcalBarrel/EBClusterTask");

    if ( meBEne_ ) dbe->removeElement( meBEne_->getName() );
    meBEne_ = 0;

    if ( meBNum_ ) dbe->removeElement( meBNum_->getName() );
    meBNum_ = 0;

    if ( meBEneMap_ ) dbe->removeElement( meBEneMap_->getName() );
    meBEneMap_ = 0;

    if ( meBNumMap_ ) dbe->removeElement( meBNumMap_->getName() );
    meBNumMap_ = 0;

    if ( meBCry_ ) dbe->removeElement( meBCry_->getName() );
    meBCry_ = 0;

    if ( meSEne_ ) dbe->removeElement( meSEne_->getName() );
    meSEne_ = 0;

    if ( meSNum_ ) dbe->removeElement( meSNum_->getName() );
    meSNum_ = 0;

    if ( meSSiz_ ) dbe->removeElement( meSSiz_->getName() );
    meSSiz_ = 0;

    if ( meSEneMap_ ) dbe->removeElement( meSEneMap_->getName() );
    meSEneMap_ = 0;

    if ( meSNumMap_ ) dbe->removeElement( meSNumMap_->getName() );
    meSNumMap_ = 0;

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

    meBNum_->Fill(float(nbcc));

    for ( BasicClusterCollection::const_iterator bclusterItr = bclusters->begin(); bclusterItr != bclusters->end(); ++bclusterItr ) {

      BasicCluster bcluster = *(bclusterItr);

      meBEne_->Fill(bcluster.energy());
      meBCry_->Fill(float(bcluster.getHitsByDetId().size()));

      meBEneMap_->Fill(bcluster.eta(), bcluster.phi(), bcluster.energy());
      meBNumMap_->Fill(bcluster.eta(), bcluster.phi());

    }

  } catch ( std::exception& ex ) {
    LogDebug("EBClusterTask") << " BasicClusterCollection not in event.";
  }

  try {

    Handle<SuperClusterCollection> sclusters;
    e.getByLabel("islandSuperClusterProducer", "islandBarrelSuperClusterCollection", sclusters);

    int nscc = sclusters->size();
    LogDebug("EBClusterTask") << "event " << ievt_ << " super cluster collection size " << nscc; 

    meSNum_->Fill(float(nscc));

    for ( SuperClusterCollection::const_iterator sclusterItr = sclusters->begin(); sclusterItr != sclusters->end(); ++sclusterItr ) {
  
      SuperCluster scluster = *(sclusterItr);

      meSEne_->Fill(scluster.energy());
      meSSiz_->Fill(float(1+scluster.clustersSize()));

      meSEneMap_->Fill(scluster.eta(), scluster.phi(), scluster.energy());
      meSNumMap_->Fill(scluster.eta(), scluster.phi());

    }

  } catch ( std::exception& ex ) {
    LogDebug("EBClusterTask") << " SuperClusterCollection not in event.";
  }

}

