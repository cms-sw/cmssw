/*
 * \file EBClusterTask.cc
 *
 * $Date: 2008/03/14 14:38:56 $
 * $Revision: 1.52 $
 * \author G. Della Ricca
 * \author E. Di Marco
 *
*/

#include <iostream>
#include <fstream>
#include <vector>
#include <math.h>

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DQMServices/Core/interface/MonitorElement.h"

#include "DQMServices/Core/interface/DQMStore.h"

#include "DataFormats/EgammaReco/interface/BasicCluster.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"
#include "DataFormats/EgammaReco/interface/ClusterShape.h"
#include "DataFormats/EgammaReco/interface/BasicClusterShapeAssociation.h"
#include "DataFormats/Math/interface/Point3D.h"

#include <DQM/EcalCommon/interface/Numbers.h>

#include <DQM/EcalBarrelMonitorTasks/interface/EBClusterTask.h>

#include "TLorentzVector.h"

using namespace cms;
using namespace edm;
using namespace std;
using namespace reco;

EBClusterTask::EBClusterTask(const ParameterSet& ps){

  init_ = false;

  // get hold of back-end interface
  dbe_ = Service<DQMStore>().operator->();

  enableCleanup_ = ps.getUntrackedParameter<bool>("enableCleanup", false);

  // parameters...
  BasicClusterCollection_ = ps.getParameter<edm::InputTag>("BasicClusterCollection");
  SuperClusterCollection_ = ps.getParameter<edm::InputTag>("SuperClusterCollection");
  ClusterShapeAssociation_ = ps.getParameter<edm::InputTag>("ClusterShapeAssociation");

  // histograms...
  meBCEne_ = 0;
  meBCNum_ = 0;
  meBCSiz_ = 0;

  meBCEneMap_ = 0;
  meBCNumMap_ = 0;
  meBCETMap_  = 0;
  meBCSizMap_ = 0;

  meBCEneMapProjEta_ = 0;
  meBCNumMapProjEta_ = 0;
  meBCETMapProjEta_  = 0;
  meBCSizMapProjEta_ = 0;

  meBCEneMapProjPhi_ = 0;
  meBCNumMapProjPhi_ = 0;
  meBCETMapProjPhi_  = 0;
  meBCSizMapProjPhi_ = 0;

  meSCEne_ = 0;
  meSCNum_ = 0;
  meSCSiz_ = 0;

  mes1s9_  = 0;
  mes9s25_  = 0;
  meInvMass_ = 0;

}

EBClusterTask::~EBClusterTask(){

}

void EBClusterTask::beginJob(const EventSetup& c){

  ievt_ = 0;

  if ( dbe_ ) {
    dbe_->setCurrentFolder("EcalBarrel/EBClusterTask");
    dbe_->rmdir("EcalBarrel/EBClusterTask");
  }

  Numbers::initGeometry(c);

}

void EBClusterTask::setup(void){

  init_ = true;

  char histo[200];

  if ( dbe_ ) {
    dbe_->setCurrentFolder("EcalBarrel/EBClusterTask");

    sprintf(histo, "EBCLT BC energy");
    meBCEne_ = dbe_->book1D(histo, histo, 100, 0., 150.);
    meBCEne_->setAxisTitle("energy (GeV)", 1);

    sprintf(histo, "EBCLT BC number");
    meBCNum_ = dbe_->book1D(histo, histo, 100, 0., 100.);
    meBCNum_->setAxisTitle("number of clusters", 1);

    sprintf(histo, "EBCLT BC size");
    meBCSiz_ = dbe_->book1D(histo, histo, 100, 0., 100.);
    meBCSiz_->setAxisTitle("cluster size", 1);

    sprintf(histo, "EBCLT BC energy map");
    meBCEneMap_ = dbe_->bookProfile2D(histo, histo, 72, -M_PI*(9+1.5)/9, M_PI*(9-1.5)/9, 34, -1.479, 1.479, 100, 0., 500., "s");
    meBCEneMap_->setAxisTitle("phi", 1);
    meBCEneMap_->setAxisTitle("eta", 2);

    sprintf(histo, "EBCLT BC number map");
    meBCNumMap_ = dbe_->book2D(histo, histo, 72, -M_PI*(9+1.5)/9, M_PI*(9-1.5)/9, 34, -1.479, 1.479);
    meBCNumMap_->setAxisTitle("phi", 1);
    meBCNumMap_->setAxisTitle("eta", 2);

    sprintf(histo, "EBCLT BC ET map");
    meBCETMap_ = dbe_->bookProfile2D(histo, histo, 72, -M_PI*(9+1.5)/9, M_PI*(9-1.5)/9, 34, -1.479, 1.479, 100, 0., 500., "s");
    meBCETMap_->setAxisTitle("phi", 1);
    meBCETMap_->setAxisTitle("eta", 2);

    sprintf(histo, "EBCLT BC size map");
    meBCSizMap_ = dbe_->bookProfile2D(histo, histo, 72, -M_PI*(9+1.5)/9, M_PI*(9-1.5)/9, 34, -1.479, 1.479, 100, 0., 100., "s");
    meBCSizMap_->setAxisTitle("phi", 1);
    meBCSizMap_->setAxisTitle("eta", 2);

    sprintf(histo, "EBCLT BC energy projection eta");
    meBCEneMapProjEta_ = dbe_->bookProfile(histo, histo, 34, -1.479, 1.479, 100, 0., 500., "s");
    meBCEneMapProjEta_->setAxisTitle("eta", 1);
    meBCEneMapProjEta_->setAxisTitle("energy (GeV)", 2);

    sprintf(histo, "EBCLT BC energy projection phi");
    meBCEneMapProjPhi_ = dbe_->bookProfile(histo, histo, 72, -M_PI*(9+1.5)/9, M_PI*(9-1.5)/9, 100, 0., 500., "s");
    meBCEneMapProjPhi_->setAxisTitle("phi", 1);
    meBCEneMapProjPhi_->setAxisTitle("energy (GeV)", 2);

    sprintf(histo, "EBCLT BC number projection eta");
    meBCNumMapProjEta_ = dbe_->book1D(histo, histo, 34, -1.479, 1.479);
    meBCNumMapProjEta_->setAxisTitle("eta", 1);
    meBCNumMapProjEta_->setAxisTitle("number of clusters", 2);

    sprintf(histo, "EBCLT BC number projection phi");
    meBCNumMapProjPhi_ = dbe_->book1D(histo, histo, 72, -M_PI*(9+1.5)/9, M_PI*(9-1.5)/9);
    meBCNumMapProjPhi_->setAxisTitle("phi", 1);
    meBCNumMapProjPhi_->setAxisTitle("number of clusters", 2);

    sprintf(histo, "EBCLT BC ET projection eta");
    meBCETMapProjEta_ = dbe_->bookProfile(histo, histo, 34, -1.479, 1.479, 100, 0., 500., "s");
    meBCETMapProjEta_->setAxisTitle("eta", 1);
    meBCETMapProjEta_->setAxisTitle("transverse energy (GeV)", 2);

    sprintf(histo, "EBCLT BC ET projection phi");
    meBCETMapProjPhi_ = dbe_->bookProfile(histo, histo, 72, -M_PI*(9+1.5)/9, M_PI*(9-1.5)/9, 100, 0., 500., "s");
    meBCETMapProjPhi_->setAxisTitle("phi", 1);
    meBCETMapProjPhi_->setAxisTitle("transverse energy (GeV)", 2);

    sprintf(histo, "EBCLT BC size projection eta");
    meBCSizMapProjEta_ = dbe_->bookProfile(histo, histo, 34, -1.479, 1.479, 100, 0., 100., "s");
    meBCSizMapProjEta_->setAxisTitle("eta", 1);
    meBCSizMapProjEta_->setAxisTitle("cluster size", 2);

    sprintf(histo, "EBCLT BC size projection phi");
    meBCSizMapProjPhi_ = dbe_->bookProfile(histo, histo, 72, -M_PI*(9+1.5)/9, M_PI*(9-1.5)/9, 100, 0., 100., "s");
    meBCSizMapProjPhi_->setAxisTitle("phi", 1);
    meBCSizMapProjPhi_->setAxisTitle("cluster size", 2);

    sprintf(histo, "EBCLT SC energy");
    meSCEne_ = dbe_->book1D(histo, histo, 100, 0., 150.);
    meSCEne_->setAxisTitle("energy (GeV)", 1);

    sprintf(histo, "EBCLT SC number");
    meSCNum_ = dbe_->book1D(histo, histo, 50, 0., 50.);
    meSCNum_->setAxisTitle("number of clusters", 1);

    sprintf(histo, "EBCLT SC size");
    meSCSiz_ = dbe_->book1D(histo, histo, 50, 0., 50.);
    meSCSiz_->setAxisTitle("cluster size", 1);

    sprintf(histo, "EBCLT s1s9");
    mes1s9_ = dbe_->book1D(histo, histo, 50, 0., 1.5);
    mes1s9_->setAxisTitle("s1/s9", 1);

    sprintf(histo, "EBCLT s9s25");
    mes9s25_ = dbe_->book1D(histo, histo, 75, 0., 1.5);
    mes9s25_->setAxisTitle("s9/s25", 1);

    sprintf(histo, "EBCLT dicluster invariant mass");
    meInvMass_ = dbe_->book1D(histo, histo, 100, 0., 200.);
    meInvMass_->setAxisTitle("mass (GeV)", 1);

  }

}

void EBClusterTask::cleanup(void){

  if ( ! enableCleanup_ ) return;

  if ( dbe_ ) {
    dbe_->setCurrentFolder("EcalBarrel/EBClusterTask");

    if ( meBCEne_ ) dbe_->removeElement( meBCEne_->getName() );
    meBCEne_ = 0;

    if ( meBCNum_ ) dbe_->removeElement( meBCNum_->getName() );
    meBCNum_ = 0;

    if ( meBCSiz_ ) dbe_->removeElement( meBCSiz_->getName() );
    meBCSiz_ = 0;

    if ( meBCEneMap_ ) dbe_->removeElement( meBCEneMap_->getName() );
    meBCEneMap_ = 0;

    if ( meBCNumMap_ ) dbe_->removeElement( meBCNumMap_->getName() );
    meBCNumMap_ = 0;

    if ( meBCETMap_ ) dbe_->removeElement( meBCETMap_->getName() );
    meBCETMap_ = 0;

    if ( meBCSizMap_ ) dbe_->removeElement( meBCSizMap_->getName() );
    meBCSizMap_ = 0;

    if ( meBCEneMapProjEta_ ) dbe_->removeElement( meBCEneMapProjEta_->getName() );
    meBCEneMapProjEta_ = 0;

    if ( meBCEneMapProjPhi_ ) dbe_->removeElement( meBCEneMapProjPhi_->getName() );
    meBCEneMapProjPhi_ = 0;

    if ( meBCNumMapProjEta_ ) dbe_->removeElement( meBCNumMapProjEta_->getName() );
    meBCNumMapProjEta_ = 0;

    if ( meBCNumMapProjPhi_ ) dbe_->removeElement( meBCNumMapProjPhi_->getName() );
    meBCNumMapProjPhi_ = 0;

    if ( meBCETMapProjEta_ ) dbe_->removeElement( meBCETMapProjEta_->getName() );
    meBCETMapProjEta_ = 0;

    if ( meBCETMapProjPhi_ ) dbe_->removeElement( meBCETMapProjPhi_->getName() );
    meBCETMapProjPhi_ = 0;

    if ( meBCSizMapProjEta_ ) dbe_->removeElement( meBCSizMapProjEta_->getName() );
    meBCSizMapProjEta_ = 0;

    if ( meBCSizMapProjPhi_ ) dbe_->removeElement( meBCSizMapProjPhi_->getName() );
    meBCSizMapProjPhi_ = 0;

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

void EBClusterTask::endJob(void){

  LogInfo("EBClusterTask") << "analyzed " << ievt_ << " events";

  if ( init_ ) this->cleanup();

}

void EBClusterTask::analyze(const Event& e, const EventSetup& c){

  if ( ! init_ ) this->setup();

  ievt_++;

  // --- Barrel Basic Clusters ---

  Handle<BasicClusterCollection> pBasicClusters;

  if ( e.getByLabel(BasicClusterCollection_, pBasicClusters) ) {

    int nbcc = pBasicClusters->size();
    if ( nbcc > 0 ) meBCNum_->Fill(float(nbcc));

    for ( BasicClusterCollection::const_iterator bCluster = pBasicClusters->begin(); bCluster != pBasicClusters->end(); ++bCluster ) {

      meBCEne_->Fill(bCluster->energy());
      meBCSiz_->Fill(float(bCluster->getHitsByDetId().size()));

      float xphi = bCluster->phi();
      if ( xphi > M_PI*(9-1.5)/9 ) xphi = xphi - M_PI*2;

      meBCEneMap_->Fill(xphi, bCluster->eta(), bCluster->energy());
      meBCEneMapProjEta_->Fill(bCluster->eta(), bCluster->energy());
      meBCEneMapProjPhi_->Fill(xphi, bCluster->energy());

      meBCNumMap_->Fill(xphi, bCluster->eta());
      meBCNumMapProjEta_->Fill(bCluster->eta());
      meBCNumMapProjPhi_->Fill(xphi);

      meBCSizMap_->Fill(xphi, bCluster->eta(), float(bCluster->getHitsByDetId().size()));
      meBCSizMapProjEta_->Fill(bCluster->eta(), float(bCluster->getHitsByDetId().size()));
      meBCSizMapProjPhi_->Fill(xphi, float(bCluster->getHitsByDetId().size()));

      meBCETMap_->Fill(xphi, bCluster->eta(), float(bCluster->energy()) * sin(bCluster->position().theta()));
      meBCETMapProjEta_->Fill(bCluster->eta(), float(bCluster->energy()) * sin(bCluster->position().theta()));
      meBCETMapProjPhi_->Fill(xphi, float(bCluster->energy()) * sin(bCluster->position().theta()));

    }

  } else {

    LogWarning("EBClusterTask") << BasicClusterCollection_ << " not available";

  }


  // --- Barrel Super Clusters ---

  Handle<SuperClusterCollection> pSuperClusters;

  if ( e.getByLabel(SuperClusterCollection_, pSuperClusters) ) {

    int nscc = pSuperClusters->size();
    if ( nscc > 0 ) meSCNum_->Fill(float(nscc));

    Handle<BasicClusterShapeAssociationCollection> pClusterShapeAssociation;

    if ( ! e.getByLabel(ClusterShapeAssociation_, pClusterShapeAssociation) ) {
      LogWarning("EBClusterTask") << "Can't get collection with label "   << ClusterShapeAssociation_.label();
    }

    TLorentzVector sc1_p(0,0,0,0);
    TLorentzVector sc2_p(0,0,0,0);

    for ( SuperClusterCollection::const_iterator sCluster = pSuperClusters->begin(); sCluster != pSuperClusters->end(); ++sCluster ) {

      // energy, size
      meSCEne_->Fill( sCluster->energy() );
      meSCSiz_->Fill( float(sCluster->clustersSize()) );

      // seed and shapes
      if ( pClusterShapeAssociation.isValid() ) {
        const ClusterShapeRef& shape = pClusterShapeAssociation->find(sCluster->seed())->val;
        mes1s9_->Fill(shape->eMax()/shape->e3x3());
        mes9s25_->Fill(shape->e3x3()/shape->e5x5());
      }

      // look for the two most energetic super clusters
      if ( nscc >= 2 ) {
	if ( sCluster->energy() > sc1_p.Energy() ) {
	  sc2_p=sc1_p;
	  sc1_p.SetPtEtaPhiE(sCluster->energy()*sin(sCluster->position().theta()),
			     sCluster->eta(), sCluster->phi(), sCluster->energy());
	} else if ( sCluster->energy() > sc2_p.Energy() ) {
	  sc2_p.SetPtEtaPhiE(sCluster->energy()*sin(sCluster->position().theta()),
			     sCluster->eta(), sCluster->phi(), sCluster->energy());
	}
      }
    }
    // Get the invariant mass of the two most energetic super clusters
    if ( nscc >= 2 ) {
      TLorentzVector sum = sc1_p+sc2_p;
      meInvMass_->Fill(sum.M());
    }

  } else {

    LogWarning("EBClusterTask") << SuperClusterCollection_ << " not available";

  }

}
