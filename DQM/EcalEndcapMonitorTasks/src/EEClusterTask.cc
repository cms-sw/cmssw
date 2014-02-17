/*
 * \file EEClusterTask.cc
 *
 * $Date: 2012/04/27 13:46:14 $
 * $Revision: 1.86 $
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

#include "DataFormats/EcalRawData/interface/EcalRawDataCollections.h"
#include "DataFormats/EgammaReco/interface/BasicCluster.h"
#include "DataFormats/EgammaReco/interface/BasicClusterFwd.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "RecoEcal/EgammaCoreTools/interface/EcalClusterTools.h"
#include "Geometry/CaloTopology/interface/CaloTopology.h"
#include "Geometry/CaloEventSetup/interface/CaloTopologyRecord.h"
#include "DataFormats/Math/interface/Point3D.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "CondFormats/EcalObjects/interface/EcalADCToGeVConstant.h"
#include "CondFormats/DataRecord/interface/EcalADCToGeVConstantRcd.h"

#include "DQM/EcalCommon/interface/Numbers.h"

#include "DQM/EcalEndcapMonitorTasks/interface/EEClusterTask.h"

#include "TLorentzVector.h"

EEClusterTask::EEClusterTask(const edm::ParameterSet& ps){

  init_ = false;

  dqmStore_ = edm::Service<DQMStore>().operator->();

  prefixME_ = ps.getUntrackedParameter<std::string>("prefixME", "");

  enableCleanup_ = ps.getUntrackedParameter<bool>("enableCleanup", false);

  mergeRuns_ = ps.getUntrackedParameter<bool>("mergeRuns", false);

  // parameters...
  EcalRawDataCollection_ = ps.getParameter<edm::InputTag>("EcalRawDataCollection");
  BasicClusterCollection_ = ps.getParameter<edm::InputTag>("BasicClusterCollection");
  SuperClusterCollection_ = ps.getParameter<edm::InputTag>("SuperClusterCollection");
  EcalRecHitCollection_ = ps.getParameter<edm::InputTag>("EcalRecHitCollection");

  // histograms...
  meBCEne_ = 0;
  meBCNum_ = 0;
  meBCSiz_ = 0;

  meBCEneFwdMap_ = 0;
  meBCNumFwdMap_ = 0;
  meBCETFwdMap_ = 0;
  meBCSizFwdMap_ = 0;

  meBCEneFwdMapProjEta_ = 0;
  meBCEneFwdMapProjPhi_ = 0;
  meBCNumFwdMapProjEta_ = 0;
  meBCNumFwdMapProjPhi_ = 0;
  meBCETFwdMapProjEta_ = 0;
  meBCETFwdMapProjPhi_ = 0;
  meBCSizFwdMapProjEta_ = 0;
  meBCSizFwdMapProjPhi_ = 0;

  meBCEneBwdMap_ = 0;
  meBCNumBwdMap_ = 0;
  meBCETBwdMap_ = 0;
  meBCSizBwdMap_ = 0;

  meBCEneBwdMapProjEta_ = 0;
  meBCEneBwdMapProjPhi_ = 0;
  meBCNumBwdMapProjEta_ = 0;
  meBCNumBwdMapProjPhi_ = 0;
  meBCETBwdMapProjEta_ = 0;
  meBCETBwdMapProjPhi_ = 0;
  meBCSizBwdMapProjEta_ = 0;
  meBCSizBwdMapProjPhi_ = 0;

  meSCEne_ = 0;
  meSCNum_ = 0;
  meSCSiz_ = 0;

  meSCCrystalSiz_ = 0;
  meSCSeedEne_ = 0;
  meSCEne2_ = 0;
  meSCEneVsEMax_ = 0;
  meSCEneLowScale_ = 0;
  meSCSeedMapOcc_[0] = 0;
  meSCSeedMapOcc_[1] = 0;
  meSCMapSingleCrystal_[0] = 0;
  meSCMapSingleCrystal_[1] = 0;

  mes1s9_ = 0;
  mes1s9thr_ = 0;
  mes9s25_ = 0;

  meInvMassPi0_ = 0;
  meInvMassJPsi_ = 0;
  meInvMassZ0_ = 0;
  meInvMassHigh_ = 0;

  meInvMassPi0Sel_ = 0;
  meInvMassJPsiSel_ = 0;
  meInvMassZ0Sel_ = 0;
  meInvMassHighSel_ = 0;

  thrS4S9_ = 0.85;
  thrClusEt_ = 0.250;
  thrCandEt_ = 0.800;

}

EEClusterTask::~EEClusterTask(){

}

void EEClusterTask::beginJob(void){

  ievt_ = 0;

  if ( dqmStore_ ) {
    dqmStore_->setCurrentFolder(prefixME_ + "/EEClusterTask");
    dqmStore_->rmdir(prefixME_ + "/EEClusterTask");
  }

}

void EEClusterTask::beginRun(const edm::Run& r, const edm::EventSetup& c) {

  Numbers::initGeometry(c, false);

  if ( ! mergeRuns_ ) this->reset();

}

void EEClusterTask::endRun(const edm::Run& r, const edm::EventSetup& c) {

}

void EEClusterTask::reset(void) {

  if ( meBCEne_ ) meBCEne_->Reset();

  if ( meBCNum_ ) meBCNum_->Reset();

  if ( meBCSiz_ ) meBCSiz_->Reset();

  if ( meBCEneFwdMap_ ) meBCEneFwdMap_->Reset();

  if ( meBCNumFwdMap_ ) meBCNumFwdMap_->Reset();

  if ( meBCETFwdMap_ ) meBCETFwdMap_->Reset();

  if ( meBCSizFwdMap_ ) meBCSizFwdMap_->Reset();

  if ( meBCEneFwdMapProjEta_ ) meBCEneFwdMapProjEta_->Reset();

  if ( meBCEneFwdMapProjPhi_ ) meBCEneFwdMapProjPhi_->Reset();

  if ( meBCNumFwdMapProjEta_ ) meBCNumFwdMapProjEta_->Reset();

  if ( meBCNumFwdMapProjPhi_ ) meBCNumFwdMapProjPhi_->Reset();

  if ( meBCETFwdMapProjEta_ ) meBCETFwdMapProjEta_->Reset();

  if ( meBCETFwdMapProjPhi_ ) meBCETFwdMapProjPhi_->Reset();

  if ( meBCSizFwdMapProjEta_ ) meBCSizFwdMapProjEta_->Reset();

  if ( meBCSizFwdMapProjPhi_ ) meBCSizFwdMapProjPhi_->Reset();

  if ( meBCEneBwdMap_ ) meBCEneBwdMap_->Reset();

  if ( meBCNumBwdMap_ ) meBCNumBwdMap_->Reset();

  if ( meBCETBwdMap_ ) meBCETBwdMap_->Reset();

  if ( meBCSizBwdMap_ ) meBCSizBwdMap_->Reset();

  if ( meBCEneBwdMapProjEta_ ) meBCEneBwdMapProjEta_->Reset();

  if ( meBCEneBwdMapProjPhi_ ) meBCEneBwdMapProjPhi_->Reset();

  if ( meBCNumBwdMapProjEta_ ) meBCNumBwdMapProjEta_->Reset();

  if ( meBCNumBwdMapProjPhi_ ) meBCNumBwdMapProjPhi_->Reset();

  if ( meBCETBwdMapProjEta_ ) meBCETBwdMapProjEta_->Reset();

  if ( meBCETBwdMapProjPhi_ ) meBCETBwdMapProjPhi_->Reset();

  if ( meBCSizBwdMapProjEta_ ) meBCSizBwdMapProjEta_->Reset();

  if ( meBCSizBwdMapProjPhi_ ) meBCSizBwdMapProjPhi_->Reset();

  if ( meSCEne_ ) meSCEne_->Reset();

  if ( meSCNum_ ) meSCNum_->Reset();

  if ( meSCSiz_ ) meSCSiz_->Reset();

  if ( meSCCrystalSiz_ ) meSCCrystalSiz_->Reset();

  if ( meSCSeedEne_ ) meSCSeedEne_->Reset();

  if ( meSCEne2_ ) meSCEne2_->Reset();

  if ( meSCEneVsEMax_ ) meSCEneVsEMax_->Reset();

  if ( meSCEneLowScale_ ) meSCEneLowScale_->Reset();

  if ( meSCSeedMapOcc_[0] ) meSCSeedMapOcc_[0]->Reset();

  if ( meSCSeedMapOcc_[1] ) meSCSeedMapOcc_[1]->Reset();

  if ( meSCMapSingleCrystal_[0] ) meSCMapSingleCrystal_[0]->Reset();

  if ( meSCMapSingleCrystal_[1] ) meSCMapSingleCrystal_[1]->Reset();

  if ( mes1s9_ ) mes1s9_->Reset();

  if ( mes1s9thr_ ) mes1s9thr_->Reset();

  if ( mes9s25_ ) mes9s25_->Reset();

  if ( meInvMassPi0_ ) meInvMassPi0_->Reset();

  if ( meInvMassJPsi_ ) meInvMassJPsi_->Reset();

  if ( meInvMassZ0_ ) meInvMassZ0_->Reset();

  if ( meInvMassHigh_ ) meInvMassHigh_->Reset();

  if ( meInvMassPi0Sel_ ) meInvMassPi0Sel_->Reset();

  if ( meInvMassJPsiSel_ ) meInvMassJPsiSel_->Reset();

  if ( meInvMassZ0Sel_ ) meInvMassZ0Sel_->Reset();

  if ( meInvMassHighSel_ ) meInvMassHighSel_->Reset();

}

void EEClusterTask::setup(void){

  init_ = true;

  std::string name;

  if ( dqmStore_ ) {
    dqmStore_->setCurrentFolder(prefixME_ + "/EEClusterTask");

    name = "EECLT BC energy";
    meBCEne_ = dqmStore_->book1D(name, name, 100, 0., 150.);
    meBCEne_->setAxisTitle("energy (GeV)", 1);

    name = "EECLT BC number";
    meBCNum_ = dqmStore_->book1D(name, name, 100, 0., 150.);
    meBCNum_->setAxisTitle("number of clusters", 1);

    name = "EECLT BC size";
    meBCSiz_ = dqmStore_->book1D(name, name, 100, 0., 150.);
    meBCSiz_->setAxisTitle("cluster size", 1);

    name = "EECLT BC energy map EE +";
    meBCEneFwdMap_ = dqmStore_->bookProfile2D(name, name, 20, -150., 150., 20, -150., 150., 100, 0., 500., "s");
    meBCEneFwdMap_->setAxisTitle("x", 1);
    meBCEneFwdMap_->setAxisTitle("y", 2);

    name = "EECLT BC number map EE +";
    meBCNumFwdMap_ = dqmStore_->book2D(name, name, 20, -150., 150., 20, -150., 150.);
    meBCNumFwdMap_->setAxisTitle("x", 1);
    meBCNumFwdMap_->setAxisTitle("y", 2);

    name = "EECLT BC ET map EE +";
    meBCETFwdMap_ = dqmStore_->bookProfile2D(name, name, 20, -150., 150., 20, -150., 150., 100, 0., 500., "s");
    meBCETFwdMap_->setAxisTitle("x", 1);
    meBCETFwdMap_->setAxisTitle("y", 2);

    name = "EECLT BC size map EE +";
    meBCSizFwdMap_ = dqmStore_->bookProfile2D(name, name, 20, -150., 150., 20, -150., 150., 100, 0., 100., "s");
    meBCSizFwdMap_->setAxisTitle("x", 1);
    meBCSizFwdMap_->setAxisTitle("y", 2);

    name = "EECLT BC energy projection eta EE +";
    meBCEneFwdMapProjEta_ = dqmStore_->bookProfile(name, name, 20, 1.479, 3.0, 100, 0., 500., "s");
    meBCEneFwdMapProjEta_->setAxisTitle("eta", 1);
    meBCEneFwdMapProjEta_->setAxisTitle("energy (GeV)", 2);

    name = "EECLT BC energy projection phi EE +";
    meBCEneFwdMapProjPhi_ = dqmStore_->bookProfile(name, name, 50, -M_PI, M_PI, 100, 0., 500., "s");
    meBCEneFwdMapProjPhi_->setAxisTitle("phi", 1);
    meBCEneFwdMapProjPhi_->setAxisTitle("energy (GeV)", 2);

    name = "EECLT BC number projection eta EE +";
    meBCNumFwdMapProjEta_ = dqmStore_->book1D(name, name, 20, 1.479, 3.0);
    meBCNumFwdMapProjEta_->setAxisTitle("eta", 1);
    meBCNumFwdMapProjEta_->setAxisTitle("number of clusters", 2);

    name = "EECLT BC number projection phi EE +";
    meBCNumFwdMapProjPhi_ = dqmStore_->book1D(name, name, 50, -M_PI, M_PI);
    meBCNumFwdMapProjPhi_->setAxisTitle("phi", 1);
    meBCNumFwdMapProjPhi_->setAxisTitle("number of clusters", 2);

    name = "EECLT BC ET projection eta EE +";
    meBCETFwdMapProjEta_ = dqmStore_->bookProfile(name, name, 20, 1.479, 3.0, 100, 0., 500., "s");
    meBCETFwdMapProjEta_->setAxisTitle("eta", 1);
    meBCETFwdMapProjEta_->setAxisTitle("transverse energy (GeV)", 2);

    name = "EECLT BC ET projection phi EE +";
    meBCETFwdMapProjPhi_ = dqmStore_->bookProfile(name, name, 50, -M_PI, M_PI, 100, 0., 500., "s");
    meBCETFwdMapProjPhi_->setAxisTitle("phi", 1);
    meBCETFwdMapProjPhi_->setAxisTitle("transverse energy (GeV)", 2);

    name = "EECLT BC size projection eta EE +";
    meBCSizFwdMapProjEta_ = dqmStore_->bookProfile(name, name, 20, 1.479, 3.0, 100, 0., 100., "s");
    meBCSizFwdMapProjEta_->setAxisTitle("eta", 1);
    meBCSizFwdMapProjEta_->setAxisTitle("cluster size", 2);

    name = "EECLT BC size projection phi EE +";
    meBCSizFwdMapProjPhi_ = dqmStore_->bookProfile(name, name, 50, -M_PI, M_PI, 100, 0., 100., "s");
    meBCSizFwdMapProjPhi_->setAxisTitle("phi", 1);
    meBCSizFwdMapProjPhi_->setAxisTitle("cluster size", 2);

    name = "EECLT BC energy map EE -";
    meBCEneBwdMap_ = dqmStore_->bookProfile2D(name, name, 20, -150., 150., 20, -150., 150., 100, 0., 500., "s");
    meBCEneBwdMap_->setAxisTitle("x", 1);
    meBCEneBwdMap_->setAxisTitle("y", 2);

    name = "EECLT BC number map EE -";
    meBCNumBwdMap_ = dqmStore_->book2D(name, name, 20, -150., 150., 20, -150., 150.);
    meBCNumBwdMap_->setAxisTitle("x", 1);
    meBCNumBwdMap_->setAxisTitle("y", 2);

    name = "EECLT BC ET map EE -";
    meBCETBwdMap_ = dqmStore_->bookProfile2D(name, name, 20, -150., 150., 20, -150., 150., 100, 0., 500., "s");
    meBCETBwdMap_->setAxisTitle("x", 1);
    meBCETBwdMap_->setAxisTitle("y", 2);

    name = "EECLT BC size map EE -";
    meBCSizBwdMap_ = dqmStore_->bookProfile2D(name, name, 20, -150., 150., 20, -150., 150., 100, 0., 100., "s");
    meBCSizBwdMap_->setAxisTitle("x", 1);
    meBCSizBwdMap_->setAxisTitle("y", 2);

    name = "EECLT BC energy projection eta EE -";
    meBCEneBwdMapProjEta_ = dqmStore_->bookProfile(name, name, 20, -3.0, -1.479, 100, 0., 500., "s");
    meBCEneBwdMapProjEta_->setAxisTitle("eta", 1);
    meBCEneBwdMapProjEta_->setAxisTitle("energy (GeV)", 2);

    name = "EECLT BC energy projection phi EE -";
    meBCEneBwdMapProjPhi_ = dqmStore_->bookProfile(name, name, 50, -M_PI, M_PI, 100, 0., 500., "s");
    meBCEneBwdMapProjPhi_->setAxisTitle("phi", 1);
    meBCEneBwdMapProjPhi_->setAxisTitle("energy (GeV)", 2);

    name = "EECLT BC number projection eta EE -";
    meBCNumBwdMapProjEta_ = dqmStore_->book1D(name, name, 20, -3.0, -1.479);
    meBCNumBwdMapProjEta_->setAxisTitle("eta", 1);
    meBCNumBwdMapProjEta_->setAxisTitle("number of clusters", 2);

    name = "EECLT BC number projection phi EE -";
    meBCNumBwdMapProjPhi_ = dqmStore_->book1D(name, name, 50, -M_PI, M_PI);
    meBCNumBwdMapProjPhi_->setAxisTitle("phi", 1);
    meBCNumBwdMapProjPhi_->setAxisTitle("number of clusters", 2);

    name = "EECLT BC ET projection eta EE -";
    meBCETBwdMapProjEta_ = dqmStore_->bookProfile(name, name, 20, -3.0, -1.479, 100, 0., 500., "s");
    meBCETBwdMapProjEta_->setAxisTitle("eta", 1);
    meBCETBwdMapProjEta_->setAxisTitle("transverse energy (GeV)", 2);

    name = "EECLT BC ET projection phi EE -";
    meBCETBwdMapProjPhi_ = dqmStore_->bookProfile(name, name, 50, -M_PI, M_PI, 100, 0., 500., "s");
    meBCETBwdMapProjPhi_->setAxisTitle("phi", 1);
    meBCETBwdMapProjPhi_->setAxisTitle("transverse energy (GeV)", 2);

    name = "EECLT BC size projection eta EE -";
    meBCSizBwdMapProjEta_ = dqmStore_->bookProfile(name, name, 20, -3.0, -1.479, 100, 0., 100., "s");
    meBCSizBwdMapProjEta_->setAxisTitle("eta", 1);
    meBCSizBwdMapProjEta_->setAxisTitle("cluster size", 2);

    name = "EECLT BC size projection phi EE -";
    meBCSizBwdMapProjPhi_ = dqmStore_->bookProfile(name, name, 50, -M_PI, M_PI, 100, 0., 100., "s");
    meBCSizBwdMapProjPhi_->setAxisTitle("phi", 1);
    meBCSizBwdMapProjPhi_->setAxisTitle("cluster size", 2);

    name = "EECLT SC energy";
    meSCEne_ = dqmStore_->book1D(name, name, 100, 0., 150.);
    meSCEne_->setAxisTitle("energy (GeV)", 1);

    name = "EECLT SC number";
    meSCNum_ = dqmStore_->book1D(name, name, 50, 0., 50.);
    meSCNum_->setAxisTitle("number of clusters", 1);

    name = "EECLT SC size";
    meSCSiz_ = dqmStore_->book1D(name, name, 50, 0., 50.);
    meSCSiz_->setAxisTitle("cluster size", 1);

    name = "EECLT SC size (crystal)";
    meSCCrystalSiz_ = dqmStore_->book1D(name, name, 150, 0, 150);
    meSCCrystalSiz_->setAxisTitle("cluster size in crystals", 1);

    name = "EECLT SC seed crystal energy";
    meSCSeedEne_ = dqmStore_->book1D(name, name, 100, 0., 10.);
    meSCSeedEne_->setAxisTitle("seed crystal energy (GeV)", 1);

    name = "EECLT SC e2";
    meSCEne2_ = dqmStore_->book1D(name, name, 100, 0., 10.);
    meSCEne2_->setAxisTitle("seed + highest neighbor crystal energy (GeV)", 1);

    name = "EECLT SC energy vs seed crystal energy";
    meSCEneVsEMax_ = dqmStore_->book2D(name, name, 50, 0., 10., 50, 0., 10.);
    meSCEneVsEMax_->setAxisTitle("seed crystal energy (GeV)", 1);
    meSCEneVsEMax_->setAxisTitle("cluster energy (GeV)", 2);

    name = "EECLT SC energy (low scale)";
    meSCEneLowScale_ = dqmStore_->book1D(name, name, 100, 0., 10.);
    meSCEneLowScale_->setAxisTitle("cluster energy (GeV)", 1);

    name = "EECLT SC seed occupancy map EE -";
    meSCSeedMapOcc_[0] = dqmStore_->book2D(name, name, 20, 0., 100., 20, 0., 100.);
    meSCSeedMapOcc_[0]->setAxisTitle("jx'", 1);
    meSCSeedMapOcc_[0]->setAxisTitle("jy'", 2);

    name = "EECLT SC seed occupancy map EE +";
    meSCSeedMapOcc_[1] = dqmStore_->book2D(name, name, 20, 0., 100., 20, 0., 100.);
    meSCSeedMapOcc_[1]->setAxisTitle("jx'", 1);
    meSCSeedMapOcc_[1]->setAxisTitle("jy'", 2);

    name = "EECLT SC single crystal cluster seed occupancy map EE -";
    meSCMapSingleCrystal_[0] = dqmStore_->book2D(name, name, 20, 0., 100., 20, 0., 100.);
    meSCMapSingleCrystal_[0]->setAxisTitle("jx'", 1);
    meSCMapSingleCrystal_[0]->setAxisTitle("jy'", 2);

    name = "EECLT SC single crystal cluster seed occupancy map EE +";
    meSCMapSingleCrystal_[1] = dqmStore_->book2D(name, name, 20, 0., 100., 20, 0., 100.);
    meSCMapSingleCrystal_[1]->setAxisTitle("jx'", 1);
    meSCMapSingleCrystal_[1]->setAxisTitle("jy'", 2);

    name = "EECLT s1s9";
    mes1s9_ = dqmStore_->book1D(name, name, 50, 0., 1.5);
    mes1s9_->setAxisTitle("s1/s9", 1);

    name = "EECLT s1s9 thr";
    mes1s9thr_ = dqmStore_->book1D(name, name, 50, 0., 1.5);
    mes1s9thr_->setAxisTitle("s1/s9", 1);

    name = "EECLT s9s25";
    mes9s25_ = dqmStore_->book1D(name, name, 75, 0., 1.5);
    mes9s25_->setAxisTitle("s9/s25", 1);

    name = "EECLT dicluster invariant mass Pi0";
    meInvMassPi0_ = dqmStore_->book1D(name, name, 50, 0.0, 0.500);
    meInvMassPi0_->setAxisTitle("mass (GeV)", 1);

    name = "EECLT dicluster invariant mass JPsi";
    meInvMassJPsi_ = dqmStore_->book1D(name, name, 50, 2.9, 3.3);
    meInvMassJPsi_->setAxisTitle("mass (GeV)", 1);

    name = "EECLT dicluster invariant mass Z0";
    meInvMassZ0_ = dqmStore_->book1D(name, name, 50, 40, 110);
    meInvMassZ0_->setAxisTitle("mass (GeV)", 1);

    name = "EECLT dicluster invariant mass high";
    meInvMassHigh_ = dqmStore_->book1D(name, name, 500, 110, 3000);
    meInvMassHigh_->setAxisTitle("mass (GeV)", 1);

    name = "EECLT dicluster invariant mass Pi0 sel";
    meInvMassPi0Sel_ = dqmStore_->book1D(name, name, 50, 0.00, 0.500);
    meInvMassPi0Sel_->setAxisTitle("mass (GeV)", 1);

    name = "EECLT dicluster invariant mass JPsi sel";
    meInvMassJPsiSel_ = dqmStore_->book1D(name, name, 50, 2.9, 3.3);
    meInvMassJPsiSel_->setAxisTitle("mass (GeV)", 1);

    name = "EECLT dicluster invariant mass Z0 sel";
    meInvMassZ0Sel_ = dqmStore_->book1D(name, name, 50, 40, 110);
    meInvMassZ0Sel_->setAxisTitle("mass (GeV)", 1);

    name = "EECLT dicluster invariant mass high sel";
    meInvMassHighSel_ = dqmStore_->book1D(name, name, 500, 110, 3000);
    meInvMassHighSel_->setAxisTitle("mass (GeV)", 1);

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

    if ( meBCEneFwdMapProjEta_ ) dqmStore_->removeElement( meBCEneFwdMapProjEta_->getName() );
    meBCEneFwdMapProjEta_ = 0;

    if ( meBCEneFwdMapProjPhi_ ) dqmStore_->removeElement( meBCEneFwdMapProjPhi_->getName() );
    meBCEneFwdMapProjPhi_ = 0;

    if ( meBCNumFwdMapProjEta_ ) dqmStore_->removeElement( meBCNumFwdMapProjEta_->getName() );
    meBCNumFwdMapProjEta_ = 0;

    if ( meBCNumFwdMapProjPhi_ ) dqmStore_->removeElement( meBCNumFwdMapProjPhi_->getName() );
    meBCNumFwdMapProjPhi_ = 0;

    if ( meBCETFwdMapProjEta_ ) dqmStore_->removeElement( meBCETFwdMapProjEta_->getName() );
    meBCETFwdMapProjEta_ = 0;

    if ( meBCETFwdMapProjPhi_ ) dqmStore_->removeElement( meBCETFwdMapProjPhi_->getName() );
    meBCETFwdMapProjPhi_ = 0;

    if ( meBCSizFwdMapProjEta_ ) dqmStore_->removeElement( meBCSizFwdMapProjEta_->getName() );
    meBCSizFwdMapProjEta_ = 0;

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

    if ( meBCEneBwdMapProjEta_ ) dqmStore_->removeElement( meBCEneBwdMapProjEta_->getName() );
    meBCEneBwdMapProjEta_ = 0;

    if ( meBCEneBwdMapProjPhi_ ) dqmStore_->removeElement( meBCEneBwdMapProjPhi_->getName() );
    meBCEneBwdMapProjPhi_ = 0;

    if ( meBCNumBwdMapProjEta_ ) dqmStore_->removeElement( meBCNumBwdMapProjEta_->getName() );
    meBCNumBwdMapProjEta_ = 0;

    if ( meBCNumBwdMapProjPhi_ ) dqmStore_->removeElement( meBCNumBwdMapProjPhi_->getName() );
    meBCNumBwdMapProjPhi_ = 0;

    if ( meBCETBwdMapProjEta_ ) dqmStore_->removeElement( meBCETBwdMapProjEta_->getName() );
    meBCETBwdMapProjEta_ = 0;

    if ( meBCETBwdMapProjPhi_ ) dqmStore_->removeElement( meBCETBwdMapProjPhi_->getName() );
    meBCETBwdMapProjPhi_ = 0;

    if ( meBCSizBwdMapProjEta_ ) dqmStore_->removeElement( meBCSizBwdMapProjEta_->getName() );
    meBCSizBwdMapProjEta_ = 0;

    if ( meBCSizBwdMapProjPhi_ ) dqmStore_->removeElement( meBCSizBwdMapProjPhi_->getName() );
    meBCSizBwdMapProjPhi_ = 0;

    if ( meSCEne_ ) dqmStore_->removeElement( meSCEne_->getName() );
    meSCEne_ = 0;

    if ( meSCNum_ ) dqmStore_->removeElement( meSCNum_->getName() );
    meSCNum_ = 0;

    if ( meSCSiz_ ) dqmStore_->removeElement( meSCSiz_->getName() );
    meSCSiz_ = 0;

    if ( meSCCrystalSiz_ ) dqmStore_->removeElement( meSCCrystalSiz_->getName() );
    meSCCrystalSiz_ = 0;

    if ( meSCSeedEne_ ) dqmStore_->removeElement( meSCSeedEne_->getName() );
    meSCSeedEne_ = 0;

    if ( meSCEne2_ ) dqmStore_->removeElement( meSCEne2_->getName() );
    meSCEne2_ = 0;

    if ( meSCEneVsEMax_ ) dqmStore_->removeElement( meSCEneVsEMax_->getName() );
    meSCEneVsEMax_ = 0;

    if ( meSCEneLowScale_ ) dqmStore_->removeElement( meSCEneLowScale_->getName() );
    meSCEneLowScale_ = 0;

    if ( meSCSeedMapOcc_[0] ) dqmStore_->removeElement( meSCSeedMapOcc_[0]->getName() );
    meSCSeedMapOcc_[0] = 0;

    if ( meSCSeedMapOcc_[1] ) dqmStore_->removeElement( meSCSeedMapOcc_[1]->getName() );
    meSCSeedMapOcc_[1] = 0;

    if ( meSCMapSingleCrystal_[0] ) dqmStore_->removeElement( meSCMapSingleCrystal_[0]->getName() );
    meSCMapSingleCrystal_[0] = 0;

    if ( meSCMapSingleCrystal_[1] ) dqmStore_->removeElement( meSCMapSingleCrystal_[1]->getName() );
    meSCMapSingleCrystal_[1] = 0;

    if ( mes1s9_ ) dqmStore_->removeElement( mes1s9_->getName() );
    mes1s9_ = 0;

    if ( mes1s9thr_ ) dqmStore_->removeElement( mes1s9thr_->getName() );
    mes1s9thr_ = 0;

    if ( mes9s25_ ) dqmStore_->removeElement( mes9s25_->getName() );
    mes9s25_ = 0;

    if ( meInvMassPi0_ ) dqmStore_->removeElement( meInvMassPi0_->getName() );
    meInvMassPi0_ = 0;

    if ( meInvMassJPsi_ ) dqmStore_->removeElement( meInvMassJPsi_->getName() );
    meInvMassJPsi_ = 0;

    if ( meInvMassZ0_ ) dqmStore_->removeElement( meInvMassZ0_->getName() );
    meInvMassZ0_ = 0;

    if ( meInvMassHigh_ ) dqmStore_->removeElement( meInvMassHigh_->getName() );
    meInvMassHigh_ = 0;

    if ( meInvMassPi0Sel_ ) dqmStore_->removeElement( meInvMassPi0Sel_->getName() );
    meInvMassPi0Sel_ = 0;

    if ( meInvMassJPsiSel_ ) dqmStore_->removeElement( meInvMassJPsiSel_->getName() );
    meInvMassJPsiSel_ = 0;

    if ( meInvMassZ0Sel_ ) dqmStore_->removeElement( meInvMassZ0Sel_->getName() );
    meInvMassZ0Sel_ = 0;

    if ( meInvMassHighSel_ ) dqmStore_->removeElement( meInvMassHighSel_->getName() );
    meInvMassHighSel_ = 0;

  }

  init_ = false;

}

void EEClusterTask::endJob(void){

  edm::LogInfo("EEClusterTask") << "analyzed " << ievt_ << " events";

  if ( enableCleanup_ ) this->cleanup();

}

void EEClusterTask::analyze(const edm::Event& e, const edm::EventSetup& c){

  bool enable = false;

  edm::Handle<EcalRawDataCollection> dcchs;

  if ( e.getByLabel(EcalRawDataCollection_, dcchs) ) {

    for ( EcalRawDataCollection::const_iterator dcchItr = dcchs->begin(); dcchItr != dcchs->end(); ++dcchItr ) {

      if ( Numbers::subDet( *dcchItr ) != EcalEndcap ) continue;

      if ( dcchItr->getRunType() == EcalDCCHeaderBlock::BEAMH4 ||
           dcchItr->getRunType() == EcalDCCHeaderBlock::BEAMH2 ) enable = true;

      if ( dcchItr->getRunType() == EcalDCCHeaderBlock::COSMIC ||
           dcchItr->getRunType() == EcalDCCHeaderBlock::MTCC ||
           dcchItr->getRunType() == EcalDCCHeaderBlock::COSMICS_GLOBAL ||
           dcchItr->getRunType() == EcalDCCHeaderBlock::PHYSICS_GLOBAL ||
           dcchItr->getRunType() == EcalDCCHeaderBlock::COSMICS_LOCAL ||
           dcchItr->getRunType() == EcalDCCHeaderBlock::PHYSICS_LOCAL ) enable = true;

      break;

    }

  } else {

    enable = true;
    edm::LogWarning("EEClusterTask") << EcalRawDataCollection_ << " not available";

  }

  if ( ! enable ) return;

  if ( ! init_ ) this->setup();

  ievt_++;

  // ECAL topology
  edm::ESHandle<CaloTopology> pTopology;
  c.get<CaloTopologyRecord>().get(pTopology);
  if ( !pTopology.isValid() ) {
    edm::LogWarning("EEClusterTask") << "Topology not valid";
    return;
  }
  const CaloTopology* topology = pTopology.product();

  // recHits
  edm::Handle< EcalRecHitCollection > pEERecHits;
  e.getByLabel( EcalRecHitCollection_, pEERecHits );
  if ( !pEERecHits.isValid() ) {
    edm::LogWarning("EEClusterTask") << "RecHit collection " << EcalRecHitCollection_ << " not available.";
    return;
  }
  const EcalRecHitCollection* eeRecHits = pEERecHits.product();

  reco::BasicClusterCollection bcSel;

  // --- Endcap Basic Clusters ---
  edm::Handle<reco::BasicClusterCollection> pBasicClusters;
  if ( e.getByLabel(BasicClusterCollection_, pBasicClusters) ) {

    int nbcc = pBasicClusters->size();
    if (nbcc>0) meBCNum_->Fill(float(nbcc));

    for ( reco::BasicClusterCollection::const_iterator bCluster = pBasicClusters->begin(); bCluster != pBasicClusters->end(); ++bCluster ) {

      meBCEne_->Fill(bCluster->energy());
      meBCSiz_->Fill(float(bCluster->size()));

      if ( bCluster->eta() > 0 ) {
        meBCEneFwdMap_->Fill(bCluster->x(), bCluster->y(), bCluster->energy());
        meBCEneFwdMapProjEta_->Fill( bCluster->eta(), bCluster->energy() );
        meBCEneFwdMapProjPhi_->Fill( bCluster->phi(), bCluster->energy() );

        meBCNumFwdMap_->Fill(bCluster->x(), bCluster->y());
        meBCNumFwdMapProjEta_->Fill( bCluster->eta() );
        meBCNumFwdMapProjPhi_->Fill( bCluster->phi() );

        meBCETFwdMap_->Fill(bCluster->x(), bCluster->y(),  bCluster->energy() * sin(bCluster->position().theta()) );
        meBCETFwdMapProjEta_->Fill( bCluster->eta(), bCluster->energy() * sin(bCluster->position().theta()) );
        meBCETFwdMapProjPhi_->Fill( bCluster->phi(), bCluster->energy() * sin(bCluster->position().theta()) );

        meBCSizFwdMap_->Fill(bCluster->x(), bCluster->y(), float(bCluster->size()) );
        meBCSizFwdMapProjEta_->Fill( bCluster->eta(), float(bCluster->size()) );
        meBCSizFwdMapProjPhi_->Fill( bCluster->phi(), float(bCluster->size()) );
      } else {
        meBCEneBwdMap_->Fill(bCluster->x(), bCluster->y(), bCluster->energy());
        meBCEneBwdMapProjEta_->Fill( bCluster->eta(), bCluster->energy() );
        meBCEneBwdMapProjPhi_->Fill( bCluster->phi(), bCluster->energy() );

        meBCNumBwdMap_->Fill(bCluster->x(), bCluster->y());
        meBCNumBwdMapProjEta_->Fill( bCluster->eta() );
        meBCNumBwdMapProjPhi_->Fill( bCluster->phi() );

        meBCETBwdMap_->Fill(bCluster->x(), bCluster->y(),  bCluster->energy() * sin(bCluster->position().theta()) );
        meBCETBwdMapProjEta_->Fill( bCluster->eta(), bCluster->energy() * sin(bCluster->position().theta()) );
        meBCETBwdMapProjPhi_->Fill( bCluster->phi(), bCluster->energy() * sin(bCluster->position().theta()) );

        meBCSizBwdMap_->Fill(bCluster->x(), bCluster->y(), float(bCluster->size()) );
        meBCSizBwdMapProjEta_->Fill( bCluster->eta(), float(bCluster->size()) );
        meBCSizBwdMapProjPhi_->Fill( bCluster->phi(), float(bCluster->size()) );

        float e2x2 = EcalClusterTools::e2x2( *bCluster, eeRecHits, topology );
        float e3x3 = EcalClusterTools::e3x3( *bCluster, eeRecHits, topology );

        // fill the selected cluster collection
        float pt = std::abs( bCluster->energy()*sin(bCluster->position().theta()) );
        if ( pt > thrClusEt_ && e2x2/e3x3 > thrS4S9_ ) bcSel.push_back(*bCluster);
      }

    }

  } else {

    edm::LogWarning("EEClusterTask") << BasicClusterCollection_ << " not available";

  }

  for ( reco::BasicClusterCollection::const_iterator bc1 = bcSel.begin(); bc1 != bcSel.end(); ++bc1 ) {
    TLorentzVector bc1P;
    bc1P.SetPtEtaPhiE(std::abs(bc1->energy()*sin(bc1->position().theta())),
                      bc1->eta(), bc1->phi(), bc1->energy());
    for ( reco::BasicClusterCollection::const_iterator bc2 = bc1+1; bc2 != bcSel.end(); ++bc2 ) {
      TLorentzVector bc2P;
      bc2P.SetPtEtaPhiE(std::abs(bc2->energy()*sin(bc2->position().theta())),
                        bc2->eta(), bc2->phi(), bc2->energy());

      TLorentzVector candP = bc1P + bc2P;

      if ( candP.Pt() > thrCandEt_ ) {
        float mass = candP.M();
        if ( mass < 0.500 ) {
          meInvMassPi0Sel_->Fill( mass );
        } else if ( mass > 2.9 && mass < 3.3 ) {
          meInvMassJPsiSel_->Fill( mass );
        } else if ( mass > 40 && mass < 110 ) {
          meInvMassZ0Sel_->Fill( mass );
        } else if ( mass > 110 ) {
          meInvMassHighSel_->Fill( mass );
        }

      }

    }
  }

  // --- Endcap Super Clusters ----
  edm::Handle<reco::SuperClusterCollection> pSuperClusters;
  if ( e.getByLabel(SuperClusterCollection_, pSuperClusters) ) {

    int nscc = pSuperClusters->size();
    if ( nscc > 0 ) meSCNum_->Fill(float(nscc));

    TLorentzVector sc1_p(0,0,0,0);
    TLorentzVector sc2_p(0,0,0,0);

    reco::SuperClusterCollection scSel;

    for ( reco::SuperClusterCollection::const_iterator sCluster = pSuperClusters->begin(); sCluster != pSuperClusters->end(); sCluster++ ) {

      // energy, size
      meSCEne_->Fill(sCluster->energy());
      meSCSiz_->Fill(float(sCluster->clustersSize()));

      reco::CaloClusterPtr theSeed = sCluster->seed();

      // Find the seed rec hit
      std::vector< std::pair<DetId,float> > sIds = sCluster->hitsAndFractions();

      float eMax, e2nd;
      EcalRecHitCollection::const_iterator seedItr = eeRecHits->begin();
      EcalRecHitCollection::const_iterator secondItr = eeRecHits->begin();

      for(std::vector< std::pair<DetId,float> >::const_iterator idItr = sIds.begin(); idItr != sIds.end(); ++idItr) {
        DetId id = idItr->first;
        if(id.det() != DetId::Ecal) { continue; }
        EcalRecHitCollection::const_iterator hitItr = eeRecHits->find(id);
        if(hitItr == eeRecHits->end()) { continue; }
        if(hitItr->energy() > secondItr->energy()) { secondItr = hitItr; }
        if(hitItr->energy() > seedItr->energy()) { std::swap(seedItr,secondItr); }
      }

      eMax = seedItr->energy();
      e2nd = secondItr->energy();
      EEDetId seedId = (EEDetId) seedItr->id();

      float e3x3 = EcalClusterTools::e3x3( *theSeed, eeRecHits, topology );
      float e5x5 = EcalClusterTools::e5x5( *theSeed, eeRecHits, topology );

      meSCCrystalSiz_->Fill(sIds.size());
      meSCSeedEne_->Fill(eMax);
      meSCEne2_->Fill(eMax+e2nd);
      meSCEneVsEMax_->Fill(eMax,sCluster->energy());
      meSCEneLowScale_->Fill(sCluster->energy());

      // Prepare to fill maps
      int ism = Numbers::iSM(seedId);
      int eeSide;
      if( ism >= 1 && ism <= 9)
        eeSide = 0;
      else
        eeSide = 1;
      int eex = seedId.ix();
      int eey = seedId.iy();
      float xeex = eex - 0.5;
      float xeey = eey - 0.5;

      meSCSeedMapOcc_[eeSide]->Fill(xeex, xeey);

      if(sIds.size() == 1) meSCMapSingleCrystal_[eeSide]->Fill(xeex, xeey);

      mes1s9_->Fill( eMax/e3x3 );
      if ( eMax > 3.0 ) mes1s9thr_->Fill( eMax/e3x3 );
      mes9s25_->Fill( e3x3/e5x5 );

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
      float mass = sum.M();
      if ( mass < 0.500 ) {
	meInvMassPi0_->Fill( mass );
      } else if ( mass > 2.9 && mass < 3.3 ) {
	meInvMassJPsi_->Fill( mass );
      } else if ( mass > 40 && mass < 110 ) {
	meInvMassZ0_->Fill( mass );
      } else if ( mass > 110 ) {
	meInvMassHigh_->Fill( mass );
      }
    }

  } else {

    edm::LogWarning("EEClusterTask") << SuperClusterCollection_ << " not available";

  }

}
