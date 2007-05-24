/*
 * \file EEClusterTask.cc
 *
 * $Date: 2007/05/22 15:08:13 $
 * $Revision: 1.7 $
 * \author G. Della Ricca
 * \author E. Di Marco
 *
*/

#include <iostream>
#include <fstream>
#include <vector>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DQMServices/Core/interface/DaqMonitorBEInterface.h"
#include "DQMServices/Daemon/interface/MonitorDaemon.h"

#include "DataFormats/EcalRawData/interface/EcalRawDataCollections.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDigi/interface/EBDataFrame.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "DataFormats/EcalRecHit/interface/EcalUncalibratedRecHit.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/EgammaReco/interface/BasicCluster.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/Math/interface/Point3D.h"

#include <DQM/EcalEndcapMonitorTasks/interface/EEClusterTask.h>

#include <TLorentzVector.h>

using namespace cms;
using namespace edm;
using namespace std;
using namespace reco;

EEClusterTask::EEClusterTask(const ParameterSet& ps){

  init_ = false;

  // get hold of back-end interface
  dbe_ = Service<DaqMonitorBEInterface>().operator->();

  enableCleanup_ = ps.getUntrackedParameter<bool>("enableCleanup", true);

  // parameters...
  islandEndcapBasicClusterCollection_ = ps.getParameter<edm::InputTag>("islandEndcapBasicClusterCollection");
  islandEndcapSuperClusterCollection_ = ps.getParameter<edm::InputTag>("islandEndcapSuperClusterCollection");

  // histograms...
  meEne_ = 0;
  meNum_ = 0;
  meSiz_ = 0;

  meEneBasic_ = 0;
  meNumBasic_ = 0;
  meSizBasic_ = 0;

  meEneFwdMap_ = 0;
  meNumFwdMap_ = 0;
  meEneFwdPolarMap_ = 0;
  meNumFwdPolarMap_ = 0;

  meEneBwdMap_ = 0;
  meNumBwdMap_ = 0;
  meEneBwdPolarMap_ = 0;
  meNumBwdPolarMap_ = 0;

  meEneFwdMapBasic_ = 0;
  meNumFwdMapBasic_ = 0;
  meEneFwdPolarMapBasic_ = 0;
  meNumFwdPolarMapBasic_ = 0;

  meEneBwdMapBasic_ = 0;
  meNumBwdMapBasic_ = 0;
  meEneBwdPolarMapBasic_ = 0;
  meNumBwdPolarMapBasic_ = 0;

  meInvMass_ = 0;

}

EEClusterTask::~EEClusterTask(){

}

void EEClusterTask::beginJob(const EventSetup& c){

  ievt_ = 0;

  if ( dbe_ ) {
    dbe_->setCurrentFolder("EcalEndcap/EEClusterTask");
    dbe_->rmdir("EcalEndcap/EEClusterTask");
  }

}

void EEClusterTask::setup(void){

  init_ = true;

  Char_t histo[200];

  if ( dbe_ ) {
    dbe_->setCurrentFolder("EcalEndcap/EEClusterTask");

    sprintf(histo, "EECLT SC energy");
    meEne_ = dbe_->book1D(histo, histo, 100, 0., 150.);

    sprintf(histo, "EECLT SC number");
    meNum_ = dbe_->book1D(histo, histo, 50, 0., 50.);

    sprintf(histo, "EECLT SC size");
    meSiz_ = dbe_->book1D(histo, histo, 10, 0., 10.);

    sprintf(histo, "EECLT BC energy");
    meEneBasic_ = dbe_->book1D(histo, histo, 100, 0., 20.);

    sprintf(histo, "EECLT BC number");
    meNumBasic_ = dbe_->book1D(histo, histo, 100, 100., 1200.);

    sprintf(histo, "EECLT BC size");
    meSizBasic_ = dbe_->book1D(histo, histo, 10, 0., 10.);

    sprintf(histo, "EECLT SC energy map EE +");
    meEneFwdMap_ = dbe_->bookProfile2D(histo, histo, 144, -171.1, 171.1, 144, -171.1, 171.1, 100, 0., 500., "s");

    sprintf(histo, "EECLT SC number map EE +");
    meNumFwdMap_ = dbe_->book2D(histo, histo, 144, -171.1, 171.1, 144, -171.1, 171.1);

    sprintf(histo, "EECLT SC energy polar map EE +");
    meEneFwdPolarMap_ = dbe_->bookProfile2D(histo, histo, 144, 0., 171.1, 180, -M_PI, M_PI, 100, 0., 500., "s");

    sprintf(histo, "EECLT SC number polar map EE +");
    meNumFwdPolarMap_ = dbe_->book2D(histo, histo, 144, 0., 171.1, 180, -M_PI, M_PI);

    sprintf(histo, "EECLT SC energy map EE -");
    meEneBwdMap_ = dbe_->bookProfile2D(histo, histo, 144, -171.1, 171.1, 144, -171.1, 171.1, 100, 0., 500., "s");

    sprintf(histo, "EECLT SC number map EE -");
    meNumBwdMap_ = dbe_->book2D(histo, histo, 144, -171.1, 171.1, 144, -171.1, 171.1);

    sprintf(histo, "EECLT SC energy polar map EE -");
    meEneBwdPolarMap_ = dbe_->bookProfile2D(histo, histo, 144, 0., 171.1, 180, -M_PI, M_PI, 100, 0., 500., "s");

    sprintf(histo, "EECLT SC number polar map EE -");
    meNumBwdPolarMap_ = dbe_->book2D(histo, histo, 144, 0., 171.1, 180, -M_PI, M_PI);

    sprintf(histo, "EECLT BC energy map EE +");
    meEneFwdMapBasic_ = dbe_->bookProfile2D(histo, histo, 144, -171.1, 171.1, 144, -171.1, 171.1, 100, 0., 500., "s");

    sprintf(histo, "EECLT BC number map EE +");
    meNumFwdMapBasic_ = dbe_->book2D(histo, histo, 144, -171.1, 171.1, 144, -171.1, 171.1);

    sprintf(histo, "EECLT BC energy polar map EE +");
    meEneFwdPolarMapBasic_ = dbe_->bookProfile2D(histo, histo, 144, 0., 171.1, 180, -M_PI, M_PI, 100, 0., 500., "s");

    sprintf(histo, "EECLT BC number polar map EE +");
    meNumFwdPolarMapBasic_ = dbe_->book2D(histo, histo, 144, 0., 171.1, 180, -M_PI, M_PI);

    sprintf(histo, "EECLT BC energy map EE -");
    meEneBwdMapBasic_ = dbe_->bookProfile2D(histo, histo, 144, -171.1, 171.1, 144, -171.1, 171.1, 100, 0., 500., "s");

    sprintf(histo, "EECLT BC number map EE -");
    meNumBwdMapBasic_ = dbe_->book2D(histo, histo, 144, -171.1, 171.1, 144, -171.1, 171.1);

    sprintf(histo, "EECLT BC energy polar map EE -");
    meEneBwdPolarMapBasic_ = dbe_->bookProfile2D(histo, histo, 144, 0., 171.1, 180, -M_PI, M_PI, 100, 0., 500., "s");

    sprintf(histo, "EECLT BC number polar map EE -");
    meNumBwdPolarMapBasic_ = dbe_->book2D(histo, histo, 144, 0., 171.1, 180, -M_PI, M_PI);

    sprintf(histo, "EECLT dicluster invariant mass");
    meInvMass_ = dbe_->book1D(histo, histo, 100, 0., 200.);

  }

}

void EEClusterTask::cleanup(void){

  if ( ! enableCleanup_ ) return;

  if ( dbe_ ) {
    dbe_->setCurrentFolder("EcalEndcap/EEClusterTask");

    if ( meEne_ ) dbe_->removeElement( meEne_->getName() );
    meEne_ = 0;

    if ( meNum_ ) dbe_->removeElement( meNum_->getName() );
    meNum_ = 0;

    if ( meSiz_ ) dbe_->removeElement( meSiz_->getName() );
    meSiz_ = 0;

    if ( meEneBasic_ ) dbe_->removeElement( meEneBasic_->getName() );
    meEneBasic_ = 0;

    if ( meNumBasic_ ) dbe_->removeElement( meNumBasic_->getName() );
    meNumBasic_ = 0;

    if ( meSizBasic_ ) dbe_->removeElement( meSizBasic_->getName() );
    meSizBasic_ = 0;

    if ( meEneFwdMap_ ) dbe_->removeElement( meEneFwdMap_->getName() );
    meEneFwdMap_ = 0;

    if ( meNumFwdMap_ ) dbe_->removeElement( meNumFwdMap_->getName() );
    meNumFwdMap_ = 0;

    if ( meEneFwdPolarMap_ ) dbe_->removeElement( meEneFwdPolarMap_->getName() );
    meEneFwdPolarMap_ = 0;

    if ( meNumFwdPolarMap_ ) dbe_->removeElement( meNumFwdPolarMap_->getName() );
    meNumFwdPolarMap_ = 0;

    if ( meEneBwdMap_ ) dbe_->removeElement( meEneBwdMap_->getName() );
    meEneBwdMap_ = 0;

    if ( meNumBwdMap_ ) dbe_->removeElement( meNumBwdMap_->getName() );
    meNumBwdMap_ = 0;

    if ( meEneBwdPolarMap_ ) dbe_->removeElement( meEneBwdPolarMap_->getName() );
    meEneBwdPolarMap_ = 0;

    if ( meNumBwdPolarMap_ ) dbe_->removeElement( meNumBwdPolarMap_->getName() );
    meNumBwdPolarMap_ = 0;

    if ( meEneFwdMapBasic_ ) dbe_->removeElement( meEneFwdMapBasic_->getName() );
    meEneFwdMapBasic_ = 0;

    if ( meNumFwdMapBasic_ ) dbe_->removeElement( meNumFwdMapBasic_->getName() );
    meNumFwdMapBasic_ = 0;

    if ( meEneFwdPolarMapBasic_ ) dbe_->removeElement( meEneFwdPolarMapBasic_->getName() );
    meEneFwdPolarMapBasic_ = 0;

    if ( meNumFwdPolarMapBasic_ ) dbe_->removeElement( meNumFwdPolarMapBasic_->getName() );
    meNumFwdPolarMapBasic_ = 0;

    if ( meEneBwdMapBasic_ ) dbe_->removeElement( meEneBwdMapBasic_->getName() );
    meEneBwdMapBasic_ = 0;

    if ( meNumBwdMapBasic_ ) dbe_->removeElement( meNumBwdMapBasic_->getName() );
    meNumBwdMapBasic_ = 0;

    if ( meEneBwdPolarMapBasic_ ) dbe_->removeElement( meEneBwdPolarMapBasic_->getName() );
    meEneBwdPolarMapBasic_ = 0;

    if ( meNumBwdPolarMapBasic_ ) dbe_->removeElement( meNumBwdPolarMapBasic_->getName() );
    meNumBwdPolarMapBasic_ = 0;

    if ( meInvMass_ ) dbe_->removeElement( meInvMass_->getName() );
    meInvMass_ = 0;

  }

  init_ = false;

}

void EEClusterTask::endJob(void){

  LogInfo("EEClusterTask") << "analyzed " << ievt_ << " events";

  if ( init_ ) this->cleanup();

}

void EEClusterTask::analyze(const Event& e, const EventSetup& c){

  if ( ! init_ ) this->setup();

  ievt_++;

  // --- Endcap "Island" Basic Clusters ---
  try {

    Handle<BasicClusterCollection> pIslandEndcapBasicClusters;
    e.getByLabel(islandEndcapBasicClusterCollection_, pIslandEndcapBasicClusters);

    meNumBasic_->Fill(float(pIslandEndcapBasicClusters->size()));

    BasicClusterCollection::const_iterator bCluster;
    for ( bCluster = pIslandEndcapBasicClusters->begin(); bCluster != pIslandEndcapBasicClusters->end(); bCluster++ ) {

      meEneBasic_->Fill(bCluster->energy());
      meSizBasic_->Fill(float(bCluster->getHitsByDetId().size()));

      if(bCluster->eta()>0) {
	meEneFwdMapBasic_->Fill(bCluster->x(), bCluster->y(), bCluster->energy());
	meNumFwdMapBasic_->Fill(bCluster->x(), bCluster->y());
	meEneFwdPolarMapBasic_->Fill(sqrt(pow(bCluster->x(),2)+pow(bCluster->y(),2)), bCluster->phi(), bCluster->energy());
	meNumFwdPolarMapBasic_->Fill(sqrt(pow(bCluster->x(),2)+pow(bCluster->y(),2)), bCluster->phi());
      }
      else {
	meEneBwdMapBasic_->Fill(bCluster->x(), bCluster->y(), bCluster->energy());
	meNumBwdMapBasic_->Fill(bCluster->x(), bCluster->y());
	meEneBwdPolarMapBasic_->Fill(sqrt(pow(bCluster->x(),2)+pow(bCluster->y(),2)), bCluster->phi(), bCluster->energy());
	meNumBwdPolarMapBasic_->Fill(sqrt(pow(bCluster->x(),2)+pow(bCluster->y(),2)), bCluster->phi());
      }

    }

  } catch ( exception& ex ) {
    LogWarning("EEClusterTask") << " BasicClusterCollection: " << islandEndcapBasicClusterCollection_ << " not in event.";
  }

  // --- Endcap "Island" Super Clusters ----
  try {

    Handle<SuperClusterCollection> pIslandEndcapSuperClusters;
    e.getByLabel(islandEndcapSuperClusterCollection_, pIslandEndcapSuperClusters);

    Int_t nscc = pIslandEndcapSuperClusters->size();
    meNum_->Fill(float(nscc));

    TLorentzVector sc1_p(0,0,0,0);
    TLorentzVector sc2_p(0,0,0,0);

    SuperClusterCollection::const_iterator sCluster;
    for ( sCluster = pIslandEndcapSuperClusters->begin(); sCluster != pIslandEndcapSuperClusters->end(); sCluster++ ) {

      meEne_->Fill(sCluster->energy());
      meSiz_->Fill(float(sCluster->clustersSize()));

      if(sCluster->eta()>0) {
	meEneFwdMap_->Fill(sCluster->x(), sCluster->y(), sCluster->energy());
	meNumFwdMap_->Fill(sCluster->x(), sCluster->y());
	meEneFwdPolarMap_->Fill(sqrt(pow(sCluster->x(),2)+pow(sCluster->y(),2)), sCluster->phi(), sCluster->energy());
	meNumFwdPolarMap_->Fill(sqrt(pow(sCluster->x(),2)+pow(sCluster->y(),2)), sCluster->phi());
      }
      else {
	meEneBwdMap_->Fill(sCluster->x(), sCluster->y(), sCluster->energy());
	meNumBwdMap_->Fill(sCluster->x(), sCluster->y());
	meEneBwdPolarMap_->Fill(sqrt(pow(sCluster->x(),2)+pow(sCluster->y(),2)), sCluster->phi(), sCluster->energy());
	meNumBwdPolarMap_->Fill(sqrt(pow(sCluster->x(),2)+pow(sCluster->y(),2)), sCluster->phi());
      }

      // look for the two most energetic super clusters
      if (sCluster->energy()>sc1_p.Energy()) {
	sc2_p=sc1_p;
	sc1_p.SetPtEtaPhiE(sCluster->energy()*sin(sCluster->position().theta()),
			   sCluster->eta(), sCluster->phi(), sCluster->energy());
      }
      else if (sCluster->energy()>sc2_p.Energy()) {
	sc2_p.SetPtEtaPhiE(sCluster->energy()*sin(sCluster->position().theta()),
			   sCluster->eta(), sCluster->phi(), sCluster->energy());
      }
      
    }
    // Get the invariant mass of the two most energetic super clusters
    if (nscc>1) {
      TLorentzVector sum = sc1_p+sc2_p;
      meInvMass_->Fill(sum.M());
    }


  } catch ( exception& ex ) {
    LogWarning("EEClusterTask") << " SuperClusterCollection: " << islandEndcapSuperClusterCollection_ << " not in event.";
  }

}

