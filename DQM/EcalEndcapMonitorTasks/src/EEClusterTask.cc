/*
 * \file EEClusterTask.cc
 *
 * $Date: 2007/10/18 17:09:49 $
 * $Revision: 1.15 $
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
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"
#include "DataFormats/EgammaReco/interface/ClusterShape.h"
#include "DataFormats/EgammaReco/interface/BasicClusterShapeAssociation.h"
#include "DataFormats/Math/interface/Point3D.h"

#include <DQM/EcalCommon/interface/Numbers.h>

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
  BasicClusterCollection_ = ps.getParameter<edm::InputTag>("BasicClusterCollection");
  SuperClusterCollection_ = ps.getParameter<edm::InputTag>("SuperClusterCollection");
  ClusterShapeAssociation_ = ps.getParameter<edm::InputTag>("ClusterShapeAssociation");

  // histograms...
  meBCEne_ = 0;
  meBCNum_ = 0;
  meBCSiz_ = 0;

  meBCEneFwdMap_ = 0;
  meBCNumFwdMap_ = 0;
  meBCETFwdMap_ = 0;
  meBCSizFwdMap_ = 0;

  meBCEneFwdMapProjR_ = 0;
  meBCEneFwdMapProjPhi_ = 0;
  meBCNumFwdMapProjR_ = 0;
  meBCNumFwdMapProjPhi_ = 0;
  meBCETFwdMapProjR_ = 0;
  meBCETFwdMapProjPhi_ = 0;
  meBCSizFwdMapProjR_ = 0;
  meBCSizFwdMapProjPhi_ = 0;

  meBCEneBwdMap_ = 0;
  meBCNumBwdMap_ = 0;
  meBCETBwdMap_ = 0;
  meBCSizBwdMap_ = 0;

  meBCEneBwdMapProjR_ = 0;
  meBCEneBwdMapProjPhi_ = 0;
  meBCNumBwdMapProjR_ = 0;
  meBCNumBwdMapProjPhi_ = 0;
  meBCETBwdMapProjR_ = 0;
  meBCETBwdMapProjPhi_ = 0;
  meBCSizBwdMapProjR_ = 0;
  meBCSizBwdMapProjPhi_ = 0;

  meSCEne_ = 0;
  meSCNum_ = 0;
  meSCSiz_ = 0;

  mes1s9_ = 0;
  mes9s25_ = 0;
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

    sprintf(histo, "EECLT BC energy");
    meBCEne_ = dbe_->book1D(histo, histo, 100, 0., 150.);

    sprintf(histo, "EECLT BC size");
    meBCSiz_ = dbe_->book1D(histo, histo, 10, 0., 10.);

    sprintf(histo, "EECLT BC number");
    meBCNum_ = dbe_->book1D(histo, histo, 100, 0., 200.);

    sprintf(histo, "EECLT BC energy map EE +");
    meBCEneFwdMap_ = dbe_->bookProfile2D(histo, histo, 20, -150.0, 150.0, 20, -150.0, 150.0, 100, 0., 500., "s");

    sprintf(histo, "EECLT BC number map EE +");
    meBCNumFwdMap_ = dbe_->book2D(histo, histo, 20, -150.0, 150.0, 20, -150.0, 150.0);

    sprintf(histo, "EECLT BC ET map EE +");
    meBCETFwdMap_ = dbe_->bookProfile2D(histo, histo, 20, -150.0, 150.0, 20, -150.0, 150.0, 100, 0., 500., "s");

    sprintf(histo, "EECLT BC size map EE +");
    meBCSizFwdMap_ = dbe_->bookProfile2D(histo, histo, 20, -150.0, 150.0, 20, -150.0, 150.0, 100, 0., 100., "s");

    sprintf(histo, "EECLT BC energy projection R EE +");
    meBCEneFwdMapProjR_ = dbe_->bookProfile(histo, histo, 20, 0., 150.0, 100, 0., 500., "s");

    sprintf(histo, "EECLT BC energy projection phi EE +");
    meBCEneFwdMapProjPhi_ = dbe_->bookProfile(histo, histo, 50, -M_PI, M_PI, 100, 0., 500., "s");

    sprintf(histo, "EECLT BC number projection R EE +");
    meBCNumFwdMapProjR_ = dbe_->book1D(histo, histo, 20, 0., 150.0);

    sprintf(histo, "EECLT BC number projection phi EE +");
    meBCNumFwdMapProjPhi_ = dbe_->book1D(histo, histo, 50, -M_PI, M_PI);

    sprintf(histo, "EECLT BC ET projection R EE +");
    meBCETFwdMapProjR_ = dbe_->bookProfile(histo, histo, 20, 0., 150.0, 100, 0., 500., "s");

    sprintf(histo, "EECLT BC ET projection phi EE +");
    meBCETFwdMapProjPhi_ = dbe_->bookProfile(histo, histo, 50, -M_PI, M_PI, 100, 0., 500., "s");

    sprintf(histo, "EECLT BC size projection R EE +");
    meBCSizFwdMapProjR_ = dbe_->bookProfile(histo, histo, 20, 0., 150.0, 100, 0., 100., "s");

    sprintf(histo, "EECLT BC size projection phi EE +");
    meBCSizFwdMapProjPhi_ = dbe_->bookProfile(histo, histo, 50, -M_PI, M_PI, 100, 0., 100., "s");

    sprintf(histo, "EECLT BC energy map EE -");
    meBCEneBwdMap_ = dbe_->bookProfile2D(histo, histo, 20, -150.0, 150.0, 20, -150.0, 150.0, 100, 0., 500., "s");

    sprintf(histo, "EECLT BC number map EE -");
    meBCNumBwdMap_ = dbe_->book2D(histo, histo, 20, -150.0, 150.0, 20, -150.0, 150.0);

    sprintf(histo, "EECLT BC ET map EE -");
    meBCETBwdMap_ = dbe_->bookProfile2D(histo, histo, 20, -150.0, 150.0, 20, -150.0, 150.0, 100, 0., 500., "s");

    sprintf(histo, "EECLT BC size map EE -");
    meBCSizBwdMap_ = dbe_->bookProfile2D(histo, histo, 20, -150.0, 150.0, 20, -150.0, 150.0, 100, 0., 100., "s");

    sprintf(histo, "EECLT BC energy projection R EE -");
    meBCEneBwdMapProjR_ = dbe_->bookProfile(histo, histo, 20, 0., 150.0, 100, 0., 500., "s");

    sprintf(histo, "EECLT BC energy projection phi EE -");
    meBCEneBwdMapProjPhi_ = dbe_->bookProfile(histo, histo, 50, -M_PI, M_PI, 100, 0., 500., "s");

    sprintf(histo, "EECLT BC number projection R EE -");
    meBCNumBwdMapProjR_ = dbe_->book1D(histo, histo, 20, 0., 150.0);

    sprintf(histo, "EECLT BC number projection phi EE -");
    meBCNumBwdMapProjPhi_ = dbe_->book1D(histo, histo, 50, -M_PI, M_PI);

    sprintf(histo, "EECLT BC ET projection R EE -");
    meBCETBwdMapProjR_ = dbe_->bookProfile(histo, histo, 20, 0., 150.0, 100, 0., 500., "s");

    sprintf(histo, "EECLT BC ET projection phi EE -");
    meBCETBwdMapProjPhi_ = dbe_->bookProfile(histo, histo, 50, -M_PI, M_PI, 100, 0., 500., "s");

    sprintf(histo, "EECLT BC size projection R EE -");
    meBCSizBwdMapProjR_ = dbe_->bookProfile(histo, histo, 20, 0., 150.0, 100, 0., 100., "s");

    sprintf(histo, "EECLT BC size projection phi EE -");
    meBCSizBwdMapProjPhi_ = dbe_->bookProfile(histo, histo, 50, -M_PI, M_PI, 100, 0., 100., "s");

    sprintf(histo, "EECLT SC energy");
    meSCEne_ = dbe_->book1D(histo, histo, 100, 0., 150.);

    sprintf(histo, "EECLT SC size");
    meSCSiz_ = dbe_->book1D(histo, histo, 10, 0., 10.);

    sprintf(histo, "EECLT SC number");
    meSCNum_ = dbe_->book1D(histo, histo, 100, 0., 200.);

    sprintf(histo, "EECLT island s1s9");
    mes1s9_ = dbe_->book1D(histo, histo, 50, 0., 1.);

    sprintf(histo, "EECLT island s9s25");
    mes9s25_ = dbe_->book1D(histo, histo, 75, 0., 1.5);

    sprintf(histo, "EECLT dicluster invariant mass");
    meInvMass_ = dbe_->book1D(histo, histo, 100, 0., 200.);

  }

}

void EEClusterTask::cleanup(void){

  if ( ! enableCleanup_ ) return;

  if ( dbe_ ) {
    dbe_->setCurrentFolder("EcalEndcap/EEClusterTask");

    if ( meBCEne_ ) dbe_->removeElement( meBCEne_->getName() );
    meBCEne_ = 0;

    if ( meBCNum_ ) dbe_->removeElement( meBCNum_->getName() );
    meBCNum_ = 0;

    if ( meBCSiz_ ) dbe_->removeElement( meBCSiz_->getName() );
    meBCSiz_ = 0;

    if ( meBCEneFwdMap_ ) dbe_->removeElement( meBCEneFwdMap_->getName() );
    meBCEneFwdMap_ = 0;

    if ( meBCNumFwdMap_ ) dbe_->removeElement( meBCNumFwdMap_->getName() );
    meBCNumFwdMap_ = 0;

    if ( meBCETFwdMap_ ) dbe_->removeElement( meBCETFwdMap_->getName() );
    meBCETFwdMap_ = 0;

    if ( meBCSizFwdMap_ ) dbe_->removeElement( meBCSizFwdMap_->getName() );
    meBCSizFwdMap_ = 0;

    if ( meBCEneFwdMapProjR_ ) dbe_->removeElement( meBCEneFwdMapProjR_->getName() );
    meBCEneFwdMapProjR_ = 0;

    if ( meBCEneFwdMapProjPhi_ ) dbe_->removeElement( meBCEneFwdMapProjPhi_->getName() );
    meBCEneFwdMapProjPhi_ = 0;

    if ( meBCNumFwdMapProjR_ ) dbe_->removeElement( meBCNumFwdMapProjR_->getName() );
    meBCNumFwdMapProjR_ = 0;

    if ( meBCNumFwdMapProjPhi_ ) dbe_->removeElement( meBCNumFwdMapProjPhi_->getName() );
    meBCNumFwdMapProjPhi_ = 0;

    if ( meBCETFwdMapProjR_ ) dbe_->removeElement( meBCETFwdMapProjR_->getName() );
    meBCETFwdMapProjR_ = 0;

    if ( meBCETFwdMapProjPhi_ ) dbe_->removeElement( meBCETFwdMapProjPhi_->getName() );
    meBCETFwdMapProjPhi_ = 0;

    if ( meBCSizFwdMapProjR_ ) dbe_->removeElement( meBCSizFwdMapProjR_->getName() );
    meBCSizFwdMapProjR_ = 0;

    if ( meBCSizFwdMapProjPhi_ ) dbe_->removeElement( meBCSizFwdMapProjPhi_->getName() );
    meBCSizFwdMapProjPhi_ = 0;

    if ( meBCEneBwdMap_ ) dbe_->removeElement( meBCEneBwdMap_->getName() );
    meBCEneBwdMap_ = 0;

    if ( meBCNumBwdMap_ ) dbe_->removeElement( meBCNumBwdMap_->getName() );
    meBCNumBwdMap_ = 0;

    if ( meBCETBwdMap_ ) dbe_->removeElement( meBCETBwdMap_->getName() );
    meBCETBwdMap_ = 0;

    if ( meBCSizBwdMap_ ) dbe_->removeElement( meBCSizBwdMap_->getName() );
    meBCSizBwdMap_ = 0;

    if ( meBCEneBwdMapProjR_ ) dbe_->removeElement( meBCEneBwdMapProjR_->getName() );
    meBCEneBwdMapProjR_ = 0;

    if ( meBCEneBwdMapProjPhi_ ) dbe_->removeElement( meBCEneBwdMapProjPhi_->getName() );
    meBCEneBwdMapProjPhi_ = 0;

    if ( meBCNumBwdMapProjR_ ) dbe_->removeElement( meBCNumBwdMapProjR_->getName() );
    meBCNumBwdMapProjR_ = 0;

    if ( meBCNumBwdMapProjPhi_ ) dbe_->removeElement( meBCNumBwdMapProjPhi_->getName() );
    meBCNumBwdMapProjPhi_ = 0;

    if ( meBCETBwdMapProjR_ ) dbe_->removeElement( meBCETBwdMapProjR_->getName() );
    meBCETBwdMapProjR_ = 0;

    if ( meBCETBwdMapProjPhi_ ) dbe_->removeElement( meBCETBwdMapProjPhi_->getName() );
    meBCETBwdMapProjPhi_ = 0;

    if ( meBCSizBwdMapProjR_ ) dbe_->removeElement( meBCSizBwdMapProjR_->getName() );
    meBCSizBwdMapProjR_ = 0;

    if ( meBCSizBwdMapProjPhi_ ) dbe_->removeElement( meBCSizBwdMapProjPhi_->getName() );
    meBCSizBwdMapProjPhi_ = 0;

    if ( meSCEne_ ) dbe_->removeElement( meSCEne_->getName() );
    meSCEne_ = 0;

    if ( meSCNum_ ) dbe_->removeElement( meSCNum_->getName() );
    meSCNum_ = 0;

    if ( meSCSiz_ ) dbe_->removeElement( meSCSiz_->getName() );
    meSCSiz_ = 0;

    if ( mes1s9_ ) dbe_->removeElement( mes1s9_->getName() );
    mes1s9_ = 0;

    if ( mes9s25_ ) dbe_->removeElement( mes9s25_->getName() );
    mes9s25_ = 0;

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

  Numbers::initGeometry(c);

  if ( ! init_ ) this->setup();

  ievt_++;

  // --- Endcap "Island" Basic Clusters ---
  try {

    Handle<BasicClusterCollection> pIslandEndcapBasicClusters;
    e.getByLabel(BasicClusterCollection_, pIslandEndcapBasicClusters);

    meBCNum_->Fill(float(pIslandEndcapBasicClusters->size()));

    BasicClusterCollection::const_iterator bCluster;
    for ( bCluster = pIslandEndcapBasicClusters->begin(); bCluster != pIslandEndcapBasicClusters->end(); bCluster++ ) {

      meBCEne_->Fill(bCluster->energy());
      meBCSiz_->Fill(float(bCluster->getHitsByDetId().size()));

      if(bCluster->eta()>0) {
	meBCEneFwdMap_->Fill(bCluster->x(), bCluster->y(), bCluster->energy());
	meBCEneFwdMapProjR_->Fill( sqrt(pow(bCluster->x(),2)+pow(bCluster->y(),2)), bCluster->energy() );
	meBCEneFwdMapProjPhi_->Fill( bCluster->phi(), bCluster->energy() );

	meBCNumFwdMap_->Fill(bCluster->x(), bCluster->y());
	meBCNumFwdMapProjR_->Fill(sqrt(pow(bCluster->x(),2)+pow(bCluster->y(),2)));
	meBCNumFwdMapProjPhi_->Fill( bCluster->phi() );

	meBCETFwdMap_->Fill(bCluster->x(), bCluster->y(),  bCluster->energy() * sin(bCluster->position().theta()) );
	meBCETFwdMapProjR_->Fill( sqrt(pow(bCluster->x(),2)+pow(bCluster->y(),2)), bCluster->energy() * sin(bCluster->position().theta()) );
	meBCETFwdMapProjPhi_->Fill( bCluster->phi(), bCluster->energy() * sin(bCluster->position().theta()) );

	meBCSizFwdMap_->Fill(bCluster->x(), bCluster->y(), float(bCluster->getHitsByDetId().size()) );
	meBCSizFwdMapProjR_->Fill( sqrt(pow(bCluster->x(),2)+pow(bCluster->y(),2)), float(bCluster->getHitsByDetId().size()) );
        meBCSizFwdMapProjPhi_->Fill( bCluster->phi(), float(bCluster->getHitsByDetId().size()) );
      }
      else {
	meBCEneBwdMap_->Fill(bCluster->x(), bCluster->y(), bCluster->energy());
	meBCEneBwdMapProjR_->Fill( sqrt(pow(bCluster->x(),2)+pow(bCluster->y(),2)), bCluster->energy() );
	meBCEneBwdMapProjPhi_->Fill( bCluster->phi(), bCluster->energy() );

	meBCNumBwdMap_->Fill(bCluster->x(), bCluster->y());
	meBCNumBwdMapProjR_->Fill(sqrt(pow(bCluster->x(),2)+pow(bCluster->y(),2)));
	meBCNumBwdMapProjPhi_->Fill( bCluster->phi() );

	meBCETBwdMap_->Fill(bCluster->x(), bCluster->y(),  bCluster->energy() * sin(bCluster->position().theta()) );
	meBCETBwdMapProjR_->Fill( sqrt(pow(bCluster->x(),2)+pow(bCluster->y(),2)), bCluster->energy() * sin(bCluster->position().theta()) );
	meBCETBwdMapProjPhi_->Fill( bCluster->phi(), bCluster->energy() * sin(bCluster->position().theta()) );

	meBCSizBwdMap_->Fill(bCluster->x(), bCluster->y(), float(bCluster->getHitsByDetId().size()) );
	meBCSizBwdMapProjR_->Fill( sqrt(pow(bCluster->x(),2)+pow(bCluster->y(),2)), float(bCluster->getHitsByDetId().size()) );
        meBCSizBwdMapProjPhi_->Fill( bCluster->phi(), float(bCluster->getHitsByDetId().size()) );
      }

    }

  } catch ( exception& ex ) {
    LogWarning("EEClusterTask") << " BasicClusterCollection: " << BasicClusterCollection_ << " not in event.";
  }

  // --- Endcap "Island" Super Clusters ----
  try {

    Handle<SuperClusterCollection> pIslandEndcapSuperClusters;
    e.getByLabel(SuperClusterCollection_, pIslandEndcapSuperClusters);

    Int_t nscc = pIslandEndcapSuperClusters->size();
    meSCNum_->Fill(float(nscc));

    Handle<BasicClusterShapeAssociationCollection> pClusterShapeAssociation;
    try	{
      e.getByLabel(ClusterShapeAssociation_, pClusterShapeAssociation);
    }	catch ( cms::Exception& ex )	{
      LogWarning("EEClusterTask") << "Can't get collection with label "   << ClusterShapeAssociation_.label();
    }

    TLorentzVector sc1_p(0,0,0,0);
    TLorentzVector sc2_p(0,0,0,0);

    SuperClusterCollection::const_iterator sCluster;
    for ( sCluster = pIslandEndcapSuperClusters->begin(); sCluster != pIslandEndcapSuperClusters->end(); sCluster++ ) {

      // energy, size 
      meSCEne_->Fill(sCluster->energy());
      meSCSiz_->Fill(float(sCluster->clustersSize()));

      // seed and shapes
      const ClusterShapeRef& shape = pClusterShapeAssociation->find(sCluster->seed())->val;
      mes1s9_->Fill(shape->eMax()/shape->e3x3());
      mes9s25_->Fill(shape->e3x3()/shape->e5x5());

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
    LogWarning("EEClusterTask") << " SuperClusterCollection: " << SuperClusterCollection_ << " not in event.";
  }

}

