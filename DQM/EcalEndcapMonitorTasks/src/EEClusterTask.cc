/*
 * \file EEClusterTask.cc
 *
 * $Date: 2008/04/08 18:11:28 $
 * $Revision: 1.45 $
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

#include <DQM/EcalEndcapMonitorTasks/interface/EEClusterTask.h>

#include "TLorentzVector.h"

using namespace cms;
using namespace edm;
using namespace std;
using namespace reco;

EEClusterTask::EEClusterTask(const ParameterSet& ps){

  init_ = false;

  dqmStore_ = Service<DQMStore>().operator->();

  prefixME_ = ps.getUntrackedParameter<string>("prefixME", "");

  enableCleanup_ = ps.getUntrackedParameter<bool>("enableCleanup", false);

  mergeRuns_ = ps.getUntrackedParameter<bool>("mergeRuns", false);

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

  if ( dqmStore_ ) {
    dqmStore_->setCurrentFolder(prefixME_ + "/EEClusterTask");
    dqmStore_->rmdir(prefixME_ + "/EEClusterTask");
  }

  Numbers::initGeometry(c, false);

}

void EEClusterTask::beginRun(const Run& r, const EventSetup& c) {

  if ( ! mergeRuns_ ) this->reset();

}

void EEClusterTask::endRun(const Run& r, const EventSetup& c) {

}

void EEClusterTask::reset(void) {

  if ( meBCEne_ ) meBCEne_->Reset();

  if ( meBCNum_ ) meBCNum_->Reset();

  if ( meBCSiz_ ) meBCSiz_->Reset();

  if ( meBCEneFwdMap_ ) meBCEneFwdMap_->Reset();

  if ( meBCNumFwdMap_ ) meBCNumFwdMap_->Reset();

  if ( meBCETFwdMap_ ) meBCETFwdMap_->Reset();

  if ( meBCSizFwdMap_ ) meBCSizFwdMap_->Reset();

  if ( meBCEneFwdMapProjR_ ) meBCEneFwdMapProjR_->Reset();

  if ( meBCEneFwdMapProjPhi_ ) meBCEneFwdMapProjPhi_->Reset();

  if ( meBCNumFwdMapProjR_ ) meBCNumFwdMapProjR_->Reset();

  if ( meBCNumFwdMapProjPhi_ ) meBCNumFwdMapProjPhi_->Reset();

  if ( meBCETFwdMapProjR_ ) meBCETFwdMapProjR_->Reset();

  if ( meBCETFwdMapProjPhi_ ) meBCETFwdMapProjPhi_->Reset();

  if ( meBCSizFwdMapProjR_ ) meBCSizFwdMapProjR_->Reset();

  if ( meBCSizFwdMapProjPhi_ ) meBCSizFwdMapProjPhi_->Reset();

  if ( meBCEneBwdMap_ ) meBCEneBwdMap_->Reset();

  if ( meBCNumBwdMap_ ) meBCNumBwdMap_->Reset();

  if ( meBCETBwdMap_ ) meBCETBwdMap_->Reset();

  if ( meBCSizBwdMap_ ) meBCSizBwdMap_->Reset();

  if ( meBCEneBwdMapProjR_ ) meBCEneBwdMapProjR_->Reset();

  if ( meBCEneBwdMapProjPhi_ ) meBCEneBwdMapProjPhi_->Reset();

  if ( meBCNumBwdMapProjR_ ) meBCNumBwdMapProjR_->Reset();

  if ( meBCNumBwdMapProjPhi_ ) meBCNumBwdMapProjPhi_->Reset();

  if ( meBCETBwdMapProjR_ ) meBCETBwdMapProjR_->Reset();

  if ( meBCETBwdMapProjPhi_ ) meBCETBwdMapProjPhi_->Reset();

  if ( meBCSizBwdMapProjR_ ) meBCSizBwdMapProjR_->Reset();

  if ( meBCSizBwdMapProjPhi_ ) meBCSizBwdMapProjPhi_->Reset();

  if ( meSCEne_ ) meSCEne_->Reset();

  if ( meSCNum_ ) meSCNum_->Reset();

  if ( meSCSiz_ ) meSCSiz_->Reset();

  if ( mes1s9_ ) mes1s9_->Reset();

  if ( mes9s25_ ) mes9s25_->Reset();

  if ( meInvMass_ ) meInvMass_->Reset();
}

void EEClusterTask::setup(void){

  init_ = true;

  char histo[200];

  if ( dqmStore_ ) {
    dqmStore_->setCurrentFolder(prefixME_ + "/EEClusterTask");

    sprintf(histo, "EECLT BC energy");
    meBCEne_ = dqmStore_->book1D(histo, histo, 100, 0., 150.);
    meBCEne_->setAxisTitle("energy (GeV)", 1);

    sprintf(histo, "EECLT BC number");
    meBCNum_ = dqmStore_->book1D(histo, histo, 100, 0., 150.);
    meBCNum_->setAxisTitle("number of clusters", 1);

    sprintf(histo, "EECLT BC size");
    meBCSiz_ = dqmStore_->book1D(histo, histo, 100, 0., 150.);
    meBCSiz_->setAxisTitle("cluster size", 1);

    sprintf(histo, "EECLT BC energy map EE +");
    meBCEneFwdMap_ = dqmStore_->bookProfile2D(histo, histo, 20, -150., 150., 20, -150., 150., 100, 0., 500., "s");
    meBCEneFwdMap_->setAxisTitle("x", 1);
    meBCEneFwdMap_->setAxisTitle("y", 2);

    sprintf(histo, "EECLT BC number map EE +");
    meBCNumFwdMap_ = dqmStore_->book2D(histo, histo, 20, -150., 150., 20, -150., 150.);
    meBCNumFwdMap_->setAxisTitle("x", 1);
    meBCNumFwdMap_->setAxisTitle("y", 2);

    sprintf(histo, "EECLT BC ET map EE +");
    meBCETFwdMap_ = dqmStore_->bookProfile2D(histo, histo, 20, -150., 150., 20, -150., 150., 100, 0., 500., "s");
    meBCETFwdMap_->setAxisTitle("x", 1);
    meBCETFwdMap_->setAxisTitle("y", 2);

    sprintf(histo, "EECLT BC size map EE +");
    meBCSizFwdMap_ = dqmStore_->bookProfile2D(histo, histo, 20, -150., 150., 20, -150., 150., 100, 0., 100., "s");
    meBCSizFwdMap_->setAxisTitle("x", 1);
    meBCSizFwdMap_->setAxisTitle("y", 2);

    sprintf(histo, "EECLT BC energy projection R EE +");
    meBCEneFwdMapProjR_ = dqmStore_->bookProfile(histo, histo, 20, 0., 150., 100, 0., 500., "s");
    meBCEneFwdMapProjR_->setAxisTitle("r", 1);
    meBCEneFwdMapProjR_->setAxisTitle("energy (GeV)", 2);

    sprintf(histo, "EECLT BC energy projection phi EE +");
    meBCEneFwdMapProjPhi_ = dqmStore_->bookProfile(histo, histo, 50, -M_PI, M_PI, 100, 0., 500., "s");
    meBCEneFwdMapProjPhi_->setAxisTitle("phi", 1);
    meBCEneFwdMapProjPhi_->setAxisTitle("energy (GeV)", 2);

    sprintf(histo, "EECLT BC number projection R EE +");
    meBCNumFwdMapProjR_ = dqmStore_->book1D(histo, histo, 20, 0., 150.);
    meBCNumFwdMapProjR_->setAxisTitle("r", 1);
    meBCNumFwdMapProjR_->setAxisTitle("number of clusters", 2);

    sprintf(histo, "EECLT BC number projection phi EE +");
    meBCNumFwdMapProjPhi_ = dqmStore_->book1D(histo, histo, 50, -M_PI, M_PI);
    meBCNumFwdMapProjPhi_->setAxisTitle("phi", 1);
    meBCNumFwdMapProjPhi_->setAxisTitle("number of clusters", 2);

    sprintf(histo, "EECLT BC ET projection R EE +");
    meBCETFwdMapProjR_ = dqmStore_->bookProfile(histo, histo, 20, 0., 150., 100, 0., 500., "s");
    meBCETFwdMapProjR_->setAxisTitle("r", 1);
    meBCETFwdMapProjR_->setAxisTitle("transverse energy (GeV)", 2);

    sprintf(histo, "EECLT BC ET projection phi EE +");
    meBCETFwdMapProjPhi_ = dqmStore_->bookProfile(histo, histo, 50, -M_PI, M_PI, 100, 0., 500., "s");
    meBCETFwdMapProjPhi_->setAxisTitle("phi", 1);
    meBCETFwdMapProjPhi_->setAxisTitle("transverse energy (GeV)", 2);

    sprintf(histo, "EECLT BC size projection R EE +");
    meBCSizFwdMapProjR_ = dqmStore_->bookProfile(histo, histo, 20, 0., 150., 100, 0., 100., "s");
    meBCSizFwdMapProjR_->setAxisTitle("r", 1);
    meBCSizFwdMapProjR_->setAxisTitle("cluster size", 2);

    sprintf(histo, "EECLT BC size projection phi EE +");
    meBCSizFwdMapProjPhi_ = dqmStore_->bookProfile(histo, histo, 50, -M_PI, M_PI, 100, 0., 100., "s");
    meBCSizFwdMapProjPhi_->setAxisTitle("phi", 1);
    meBCSizFwdMapProjPhi_->setAxisTitle("cluster size", 2);

    sprintf(histo, "EECLT BC energy map EE -");
    meBCEneBwdMap_ = dqmStore_->bookProfile2D(histo, histo, 20, -150., 150., 20, -150., 150., 100, 0., 500., "s");
    meBCEneBwdMap_->setAxisTitle("x", 1);
    meBCEneBwdMap_->setAxisTitle("y", 2);

    sprintf(histo, "EECLT BC number map EE -");
    meBCNumBwdMap_ = dqmStore_->book2D(histo, histo, 20, -150., 150., 20, -150., 150.);
    meBCNumBwdMap_->setAxisTitle("x", 1);
    meBCNumBwdMap_->setAxisTitle("y", 2);

    sprintf(histo, "EECLT BC ET map EE -");
    meBCETBwdMap_ = dqmStore_->bookProfile2D(histo, histo, 20, -150., 150., 20, -150., 150., 100, 0., 500., "s");
    meBCETBwdMap_->setAxisTitle("x", 1);
    meBCETBwdMap_->setAxisTitle("y", 2);

    sprintf(histo, "EECLT BC size map EE -");
    meBCSizBwdMap_ = dqmStore_->bookProfile2D(histo, histo, 20, -150., 150., 20, -150., 150., 100, 0., 100., "s");
    meBCSizBwdMap_->setAxisTitle("x", 1);
    meBCSizBwdMap_->setAxisTitle("y", 2);

    sprintf(histo, "EECLT BC energy projection R EE -");
    meBCEneBwdMapProjR_ = dqmStore_->bookProfile(histo, histo, 20, 0., 150., 100, 0., 500., "s");
    meBCEneBwdMapProjR_->setAxisTitle("r", 1);
    meBCEneBwdMapProjR_->setAxisTitle("energy (GeV)", 2);

    sprintf(histo, "EECLT BC energy projection phi EE -");
    meBCEneBwdMapProjPhi_ = dqmStore_->bookProfile(histo, histo, 50, -M_PI, M_PI, 100, 0., 500., "s");
    meBCEneBwdMapProjPhi_->setAxisTitle("phi", 1);
    meBCEneBwdMapProjPhi_->setAxisTitle("energy (GeV)", 2);

    sprintf(histo, "EECLT BC number projection R EE -");
    meBCNumBwdMapProjR_ = dqmStore_->book1D(histo, histo, 20, 0., 150.);
    meBCNumBwdMapProjR_->setAxisTitle("r", 1);
    meBCNumBwdMapProjR_->setAxisTitle("number of clusters", 2);

    sprintf(histo, "EECLT BC number projection phi EE -");
    meBCNumBwdMapProjPhi_ = dqmStore_->book1D(histo, histo, 50, -M_PI, M_PI);
    meBCNumBwdMapProjPhi_->setAxisTitle("phi", 1);
    meBCNumBwdMapProjPhi_->setAxisTitle("number of clusters", 2);

    sprintf(histo, "EECLT BC ET projection R EE -");
    meBCETBwdMapProjR_ = dqmStore_->bookProfile(histo, histo, 20, 0., 150., 100, 0., 500., "s");
    meBCETBwdMapProjR_->setAxisTitle("r", 1);
    meBCETBwdMapProjR_->setAxisTitle("transverse energy (GeV)", 2);

    sprintf(histo, "EECLT BC ET projection phi EE -");
    meBCETBwdMapProjPhi_ = dqmStore_->bookProfile(histo, histo, 50, -M_PI, M_PI, 100, 0., 500., "s");
    meBCETBwdMapProjPhi_->setAxisTitle("phi", 1);
    meBCETBwdMapProjPhi_->setAxisTitle("transverse energy (GeV)", 2);

    sprintf(histo, "EECLT BC size projection R EE -");
    meBCSizBwdMapProjR_ = dqmStore_->bookProfile(histo, histo, 20, 0., 150., 100, 0., 100., "s");
    meBCSizBwdMapProjR_->setAxisTitle("r", 1);
    meBCSizBwdMapProjR_->setAxisTitle("cluster size", 2);

    sprintf(histo, "EECLT BC size projection phi EE -");
    meBCSizBwdMapProjPhi_ = dqmStore_->bookProfile(histo, histo, 50, -M_PI, M_PI, 100, 0., 100., "s");
    meBCSizBwdMapProjPhi_->setAxisTitle("phi", 1);
    meBCSizBwdMapProjPhi_->setAxisTitle("cluster size", 2);

    sprintf(histo, "EECLT SC energy");
    meSCEne_ = dqmStore_->book1D(histo, histo, 100, 0., 150.);
    meSCEne_->setAxisTitle("energy (GeV)", 1);

    sprintf(histo, "EECLT SC number");
    meSCNum_ = dqmStore_->book1D(histo, histo, 50, 0., 50.);
    meSCNum_->setAxisTitle("number of clusters", 1);

    sprintf(histo, "EECLT SC size");
    meSCSiz_ = dqmStore_->book1D(histo, histo, 50, 0., 50.);
    meSCSiz_->setAxisTitle("cluster size", 1);

    sprintf(histo, "EECLT s1s9");
    mes1s9_ = dqmStore_->book1D(histo, histo, 50, 0., 1.5);
    mes1s9_->setAxisTitle("s1/s9", 1);

    sprintf(histo, "EECLT s9s25");
    mes9s25_ = dqmStore_->book1D(histo, histo, 75, 0., 1.5);
    mes9s25_->setAxisTitle("s9/s25", 1);

    sprintf(histo, "EECLT dicluster invariant mass");
    meInvMass_ = dqmStore_->book1D(histo, histo, 100, 0., 200.);
    meInvMass_->setAxisTitle("mass (GeV)", 1);

  }

}

void EEClusterTask::cleanup(void){

  if ( ! init_ ) return;

  if ( dqmStore_ ) {
    dqmStore_->setCurrentFolder(prefixME_ + "/EEClusterTask");

    if ( meBCEne_ ) dqmStore_->removeElement( meBCEne_->getName() );
    meBCEne_ = 0;

    if ( meBCNum_ ) dqmStore_->removeElement( meBCNum_->getName() );
    meBCNum_ = 0;

    if ( meBCSiz_ ) dqmStore_->removeElement( meBCSiz_->getName() );
    meBCSiz_ = 0;

    if ( meBCEneFwdMap_ ) dqmStore_->removeElement( meBCEneFwdMap_->getName() );
    meBCEneFwdMap_ = 0;

    if ( meBCNumFwdMap_ ) dqmStore_->removeElement( meBCNumFwdMap_->getName() );
    meBCNumFwdMap_ = 0;

    if ( meBCETFwdMap_ ) dqmStore_->removeElement( meBCETFwdMap_->getName() );
    meBCETFwdMap_ = 0;

    if ( meBCSizFwdMap_ ) dqmStore_->removeElement( meBCSizFwdMap_->getName() );
    meBCSizFwdMap_ = 0;

    if ( meBCEneFwdMapProjR_ ) dqmStore_->removeElement( meBCEneFwdMapProjR_->getName() );
    meBCEneFwdMapProjR_ = 0;

    if ( meBCEneFwdMapProjPhi_ ) dqmStore_->removeElement( meBCEneFwdMapProjPhi_->getName() );
    meBCEneFwdMapProjPhi_ = 0;

    if ( meBCNumFwdMapProjR_ ) dqmStore_->removeElement( meBCNumFwdMapProjR_->getName() );
    meBCNumFwdMapProjR_ = 0;

    if ( meBCNumFwdMapProjPhi_ ) dqmStore_->removeElement( meBCNumFwdMapProjPhi_->getName() );
    meBCNumFwdMapProjPhi_ = 0;

    if ( meBCETFwdMapProjR_ ) dqmStore_->removeElement( meBCETFwdMapProjR_->getName() );
    meBCETFwdMapProjR_ = 0;

    if ( meBCETFwdMapProjPhi_ ) dqmStore_->removeElement( meBCETFwdMapProjPhi_->getName() );
    meBCETFwdMapProjPhi_ = 0;

    if ( meBCSizFwdMapProjR_ ) dqmStore_->removeElement( meBCSizFwdMapProjR_->getName() );
    meBCSizFwdMapProjR_ = 0;

    if ( meBCSizFwdMapProjPhi_ ) dqmStore_->removeElement( meBCSizFwdMapProjPhi_->getName() );
    meBCSizFwdMapProjPhi_ = 0;

    if ( meBCEneBwdMap_ ) dqmStore_->removeElement( meBCEneBwdMap_->getName() );
    meBCEneBwdMap_ = 0;

    if ( meBCNumBwdMap_ ) dqmStore_->removeElement( meBCNumBwdMap_->getName() );
    meBCNumBwdMap_ = 0;

    if ( meBCETBwdMap_ ) dqmStore_->removeElement( meBCETBwdMap_->getName() );
    meBCETBwdMap_ = 0;

    if ( meBCSizBwdMap_ ) dqmStore_->removeElement( meBCSizBwdMap_->getName() );
    meBCSizBwdMap_ = 0;

    if ( meBCEneBwdMapProjR_ ) dqmStore_->removeElement( meBCEneBwdMapProjR_->getName() );
    meBCEneBwdMapProjR_ = 0;

    if ( meBCEneBwdMapProjPhi_ ) dqmStore_->removeElement( meBCEneBwdMapProjPhi_->getName() );
    meBCEneBwdMapProjPhi_ = 0;

    if ( meBCNumBwdMapProjR_ ) dqmStore_->removeElement( meBCNumBwdMapProjR_->getName() );
    meBCNumBwdMapProjR_ = 0;

    if ( meBCNumBwdMapProjPhi_ ) dqmStore_->removeElement( meBCNumBwdMapProjPhi_->getName() );
    meBCNumBwdMapProjPhi_ = 0;

    if ( meBCETBwdMapProjR_ ) dqmStore_->removeElement( meBCETBwdMapProjR_->getName() );
    meBCETBwdMapProjR_ = 0;

    if ( meBCETBwdMapProjPhi_ ) dqmStore_->removeElement( meBCETBwdMapProjPhi_->getName() );
    meBCETBwdMapProjPhi_ = 0;

    if ( meBCSizBwdMapProjR_ ) dqmStore_->removeElement( meBCSizBwdMapProjR_->getName() );
    meBCSizBwdMapProjR_ = 0;

    if ( meBCSizBwdMapProjPhi_ ) dqmStore_->removeElement( meBCSizBwdMapProjPhi_->getName() );
    meBCSizBwdMapProjPhi_ = 0;

    if ( meSCEne_ ) dqmStore_->removeElement( meSCEne_->getName() );
    meSCEne_ = 0;

    if ( meSCNum_ ) dqmStore_->removeElement( meSCNum_->getName() );
    meSCNum_ = 0;

    if ( meSCSiz_ ) dqmStore_->removeElement( meSCSiz_->getName() );
    meSCSiz_ = 0;

    if ( mes1s9_ ) dqmStore_->removeElement( mes1s9_->getName() );
    mes1s9_ = 0;

    if ( mes9s25_ ) dqmStore_->removeElement( mes9s25_->getName() );
    mes9s25_ = 0;

    if ( meInvMass_ ) dqmStore_->removeElement( meInvMass_->getName() );
    meInvMass_ = 0;

  }

  init_ = false;

}

void EEClusterTask::endJob(void){

  LogInfo("EEClusterTask") << "analyzed " << ievt_ << " events";

  if ( enableCleanup_ ) this->cleanup();

}

void EEClusterTask::analyze(const Event& e, const EventSetup& c){

  if ( ! init_ ) this->setup();

  ievt_++;

  // --- Endcap Basic Clusters ---

  Handle<BasicClusterCollection> pBasicClusters;

  if ( e.getByLabel(BasicClusterCollection_, pBasicClusters) ) {

    int nbcc = pBasicClusters->size();
    if (nbcc>0) meBCNum_->Fill(float(nbcc));

    for ( BasicClusterCollection::const_iterator bCluster = pBasicClusters->begin(); bCluster != pBasicClusters->end(); ++bCluster ) {

      meBCEne_->Fill(bCluster->energy());
      meBCSiz_->Fill(float(bCluster->getHitsByDetId().size()));

      if ( bCluster->eta() > 0 ) {
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
      } else {
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

  } else {

    LogWarning("EEClusterTask") << BasicClusterCollection_ << " not available";

  }

  // --- Endcap Super Clusters ----

  Handle<SuperClusterCollection> pSuperClusters;

  if ( e.getByLabel(SuperClusterCollection_, pSuperClusters) ) {

    int nscc = pSuperClusters->size();
    if ( nscc > 0 ) meSCNum_->Fill(float(nscc));

    Handle<BasicClusterShapeAssociationCollection> pClusterShapeAssociation;

    if ( ! e.getByLabel(ClusterShapeAssociation_, pClusterShapeAssociation) ) {
      LogWarning("EEClusterTask") << "Can't get collection with label "   << ClusterShapeAssociation_.label();
    }

    TLorentzVector sc1_p(0,0,0,0);
    TLorentzVector sc2_p(0,0,0,0);

    for ( SuperClusterCollection::const_iterator sCluster = pSuperClusters->begin(); sCluster != pSuperClusters->end(); sCluster++ ) {

      // energy, size
      meSCEne_->Fill(sCluster->energy());
      meSCSiz_->Fill(float(sCluster->clustersSize()));

      // seed and shapes
      if ( pClusterShapeAssociation.isValid() ) {
        const ClusterShapeRef& shape = pClusterShapeAssociation->find(sCluster->seed())->val;
        mes1s9_->Fill(shape->eMax()/shape->e3x3());
        mes9s25_->Fill(shape->e3x3()/shape->e5x5());
      }

      // look for the two most energetic super clusters
      if ( sCluster->energy() > sc1_p.Energy() ) {
	sc2_p=sc1_p;
	sc1_p.SetPtEtaPhiE(sCluster->energy()*sin(sCluster->position().theta()),
			   sCluster->eta(), sCluster->phi(), sCluster->energy());
      } else if ( sCluster->energy() > sc2_p.Energy() ) {
	sc2_p.SetPtEtaPhiE(sCluster->energy()*sin(sCluster->position().theta()),
			   sCluster->eta(), sCluster->phi(), sCluster->energy());
      }

    }
    // Get the invariant mass of the two most energetic super clusters
    if ( nscc >= 2) {
      TLorentzVector sum = sc1_p+sc2_p;
      meInvMass_->Fill(sum.M());
    }

  } else {

    LogWarning("EEClusterTask") << SuperClusterCollection_ << " not available";

  }

}
