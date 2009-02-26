/*
 * \file EEClusterTask.cc
 *
 * $Date: 2009/02/05 15:09:07 $
 * $Revision: 1.58 $
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

  meSCCrystalSiz_ = 0;
  meSCSeedEne_ = 0;
  meSCEne2_ = 0;
  meSCEneVsEMax_ = 0;
  meSCEneLowScale_ = 0;
  meSCSeedMapOcc_[0] = 0;
  meSCSeedMapOcc_[1] = 0;
  meSCMapSingleCrystal_[0] = 0;
  meSCMapSingleCrystal_[1] = 0;
  meSCSeedTimingSummary_ = 0;
  meSCSeedTimingMap_[0] = 0;
  meSCSeedTimingMap_[1] = 0;
  for(int i=0;i<18;++i)
    meSCSeedTiming_[i] = 0;

  mes1s9_ = 0;
  mes9s25_ = 0;

  meInvMassPi0_ = 0;
  meInvMassJPsi_ = 0;
  meInvMassZ0_ = 0;
  meInvMassHigh_ = 0;

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

  if ( meSCCrystalSiz_ ) meSCCrystalSiz_->Reset();

  if ( meSCSeedEne_ ) meSCSeedEne_->Reset();

  if ( meSCEne2_ ) meSCEne2_->Reset();

  if ( meSCEneVsEMax_ ) meSCEneVsEMax_->Reset();

  if ( meSCEneLowScale_ ) meSCEneLowScale_->Reset();

  if ( meSCSeedMapOcc_[0] ) meSCSeedMapOcc_[0]->Reset();

  if ( meSCSeedMapOcc_[1] ) meSCSeedMapOcc_[1]->Reset();

  if ( meSCMapSingleCrystal_[0] ) meSCMapSingleCrystal_[0]->Reset();

  if ( meSCMapSingleCrystal_[1] ) meSCMapSingleCrystal_[1]->Reset();

  if ( meSCSeedTimingSummary_ ) meSCSeedTimingSummary_->Reset();

  if ( meSCSeedTimingMap_[0] ) meSCSeedTimingMap_[0]->Reset();

  if ( meSCSeedTimingMap_[1] ) meSCSeedTimingMap_[1]->Reset();

  for(int i=0; i<18; ++i)
    if ( meSCSeedTiming_[i] ) meSCSeedTiming_[i]->Reset();

  if ( mes1s9_ ) mes1s9_->Reset();

  if ( mes9s25_ ) mes9s25_->Reset();

  if ( meInvMassPi0_ ) meInvMassPi0_->Reset();

  if ( meInvMassJPsi_ ) meInvMassJPsi_->Reset();

  if ( meInvMassZ0_ ) meInvMassZ0_->Reset();

  if ( meInvMassHigh_ ) meInvMassHigh_->Reset();

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

    sprintf(histo, "EECLT SC size (crystal)");
    meSCCrystalSiz_ = dqmStore_->book1D(histo, histo, 150, 0, 150);
    meSCCrystalSiz_->setAxisTitle("cluster size in crystals", 1);

    sprintf(histo, "EECLT SC seed crystal energy");
    meSCSeedEne_ = dqmStore_->book1D(histo, histo, 200, 0, 1.8);
    meSCSeedEne_->setAxisTitle("seed crystal energy (GeV)", 1);

    sprintf(histo, "EECLT SC e2");
    meSCEne2_ = dqmStore_->book1D(histo, histo, 200, 0, 1.8);
    meSCEne2_->setAxisTitle("seed + highest neighbor crystal energy (GeV)", 1);

    sprintf(histo, "EECLT SC energy vs seed crystal energy");
    meSCEneVsEMax_ = dqmStore_->book2D(histo, histo, 200, 0, 1.8, 200, 0, 1.8);
    meSCEneVsEMax_->setAxisTitle("seed crystal energy (GeV)", 1);
    meSCEneVsEMax_->setAxisTitle("cluster energy (GeV)", 2);

    sprintf(histo, "EECLT SC energy (low scale)");
    meSCEneLowScale_ = dqmStore_->book1D(histo, histo, 200, 0, 1.8);
    meSCEneLowScale_->setAxisTitle("cluster energy (GeV)", 1);

    sprintf(histo, "EECLT SC seed occupancy map EE -");
    meSCSeedMapOcc_[0] = dqmStore_->book2D(histo, histo, 20, 0., 100., 20, 0., 100.);
    meSCSeedMapOcc_[0]->setAxisTitle("jx", 1);
    meSCSeedMapOcc_[0]->setAxisTitle("jy", 2);

    sprintf(histo, "EECLT SC seed occupancy map EE +");
    meSCSeedMapOcc_[1] = dqmStore_->book2D(histo, histo, 20, 0., 100., 20, 0., 100.);
    meSCSeedMapOcc_[1]->setAxisTitle("jx", 1);
    meSCSeedMapOcc_[1]->setAxisTitle("jy", 2);

    sprintf(histo, "EECLT SC single crystal cluster seed occupancy map EE -");
    meSCMapSingleCrystal_[0] = dqmStore_->book2D(histo, histo, 20, 0., 100., 20, 0., 100.);
    meSCMapSingleCrystal_[0]->setAxisTitle("jx", 1);
    meSCMapSingleCrystal_[0]->setAxisTitle("jy", 2);

    sprintf(histo, "EECLT SC single crystal cluster seed occupancy map EE +");
    meSCMapSingleCrystal_[1] = dqmStore_->book2D(histo, histo, 20, 0., 100., 20, 0., 100.);
    meSCMapSingleCrystal_[1]->setAxisTitle("jx", 1);
    meSCMapSingleCrystal_[1]->setAxisTitle("jy", 2);

    sprintf(histo, "EECLT SC seed crystal timing");
    meSCSeedTimingSummary_ = dqmStore_->book1D(histo, histo, 78, 0., 10.);
    meSCSeedTimingSummary_->setAxisTitle("seed crystal timing", 1);

    sprintf(histo, "EECLT SC seed crystal timing map EE -");
    meSCSeedTimingMap_[0] = dqmStore_->bookProfile2D(histo, histo, 20, 0., 100., 20, 0., 100., 78, 0., 10., "s");
    meSCSeedTimingMap_[0]->setAxisTitle("jx", 1);
    meSCSeedTimingMap_[0]->setAxisTitle("jy", 2);

    sprintf(histo, "EECLT SC seed crystal timing map EE +");
    meSCSeedTimingMap_[1] = dqmStore_->bookProfile2D(histo, histo, 20, 0., 100., 20, 0., 100., 78, 0., 10., "s");
    meSCSeedTimingMap_[1]->setAxisTitle("jx", 1);
    meSCSeedTimingMap_[1]->setAxisTitle("jy", 2);

    dqmStore_->setCurrentFolder(prefixME_ + "/EEClusterTask/Timing");
    for(int i = 0; i < 18; i++) {
      sprintf(histo, "EECLT timing %s", Numbers::sEE(i+1).c_str());
      meSCSeedTiming_[i] = dqmStore_->book1D(histo, histo, 100, 0., 10.);
    }
    dqmStore_->setCurrentFolder(prefixME_ + "/EEClusterTask");

    sprintf(histo, "EECLT s1s9");
    mes1s9_ = dqmStore_->book1D(histo, histo, 50, 0., 1.5);
    mes1s9_->setAxisTitle("s1/s9", 1);

    sprintf(histo, "EECLT s9s25");
    mes9s25_ = dqmStore_->book1D(histo, histo, 75, 0., 1.5);
    mes9s25_->setAxisTitle("s9/s25", 1);

    sprintf(histo, "EECLT dicluster invariant mass Pi0");
    meInvMassPi0_ = dqmStore_->book1D(histo, histo, 50, 0., 0.300);
    meInvMassPi0_->setAxisTitle("mass (GeV)", 1);

    sprintf(histo, "EECLT dicluster invariant mass JPsi");
    meInvMassJPsi_ = dqmStore_->book1D(histo, histo, 50, 2.9, 3.3);
    meInvMassJPsi_->setAxisTitle("mass (GeV)", 1);

    sprintf(histo, "EECLT dicluster invariant mass Z0");
    meInvMassZ0_ = dqmStore_->book1D(histo, histo, 50, 40, 110);
    meInvMassZ0_->setAxisTitle("mass (GeV)", 1);

    sprintf(histo, "EECLT dicluster invariant mass high");
    meInvMassHigh_ = dqmStore_->book1D(histo, histo, 500, 110, 3000);
    meInvMassHigh_->setAxisTitle("mass (GeV)", 1);

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

    if ( meSCSeedTimingSummary_ ) dqmStore_->removeElement( meSCSeedTimingSummary_->getName() );
    meSCSeedTimingSummary_ = 0;

    if ( meSCSeedTimingMap_[0] ) dqmStore_->removeElement( meSCSeedTimingMap_[0]->getName() );
    meSCSeedTimingMap_[0] = 0;

    if ( meSCSeedTimingMap_[1] ) dqmStore_->removeElement( meSCSeedTimingMap_[1]->getName() );
    meSCSeedTimingMap_[1] = 0;

    dqmStore_->setCurrentFolder(prefixME_ + "/EEClusterTask/Timing");
    for(int i=0;i<18;++i) {
      if ( meSCSeedTiming_[i] ) dqmStore_->removeElement( meSCSeedTiming_[i]->getName() );
      meSCSeedTiming_[i] = 0;
    }
    dqmStore_->setCurrentFolder(prefixME_ + "/EEClusterTask");

    if ( mes1s9_ ) dqmStore_->removeElement( mes1s9_->getName() );
    mes1s9_ = 0;

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

  }

  init_ = false;

}

void EEClusterTask::endJob(void){

  LogInfo("EEClusterTask") << "analyzed " << ievt_ << " events";

  if ( enableCleanup_ ) this->cleanup();

}

void EEClusterTask::analyze(const Event& e, const EventSetup& c){

  bool enable = false;

  Handle<EcalRawDataCollection> dcchs;

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
    LogWarning("EEClusterTask") << EcalRawDataCollection_ << " not available";

  }

  if ( ! enable ) return;

  if ( ! init_ ) this->setup();

  ievt_++;

  // --- Endcap Basic Clusters ---

  Handle<BasicClusterCollection> pBasicClusters;

  if ( e.getByLabel(BasicClusterCollection_, pBasicClusters) ) {

    int nbcc = pBasicClusters->size();
    if (nbcc>0) meBCNum_->Fill(float(nbcc));

    for ( BasicClusterCollection::const_iterator bCluster = pBasicClusters->begin(); bCluster != pBasicClusters->end(); ++bCluster ) {

      meBCEne_->Fill(bCluster->energy());
      meBCSiz_->Fill(float(bCluster->size()));

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

        meBCSizFwdMap_->Fill(bCluster->x(), bCluster->y(), float(bCluster->size()) );
        meBCSizFwdMapProjR_->Fill( sqrt(pow(bCluster->x(),2)+pow(bCluster->y(),2)), float(bCluster->size()) );
        meBCSizFwdMapProjPhi_->Fill( bCluster->phi(), float(bCluster->size()) );
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

        meBCSizBwdMap_->Fill(bCluster->x(), bCluster->y(), float(bCluster->size()) );
        meBCSizBwdMapProjR_->Fill( sqrt(pow(bCluster->x(),2)+pow(bCluster->y(),2)), float(bCluster->size()) );
        meBCSizBwdMapProjPhi_->Fill( bCluster->phi(), float(bCluster->size()) );
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

    TLorentzVector sc1_p(0,0,0,0);
    TLorentzVector sc2_p(0,0,0,0);

    for ( SuperClusterCollection::const_iterator sCluster = pSuperClusters->begin(); sCluster != pSuperClusters->end(); sCluster++ ) {

      // energy, size
      meSCEne_->Fill(sCluster->energy());
      meSCSiz_->Fill(float(sCluster->clustersSize()));

      // seed and shapes
      edm::Handle< EcalRecHitCollection > pEERecHits;
      e.getByLabel( EcalRecHitCollection_, pEERecHits );
      if ( pEERecHits.isValid() ) {
        const EcalRecHitCollection *eeRecHits = pEERecHits.product();

	edm::ESHandle<CaloTopology> pTopology;
        c.get<CaloTopologyRecord>().get(pTopology);
        if ( pTopology.isValid() ) {
          const CaloTopology *topology = pTopology.product();
          
          BasicClusterRef theSeed = sCluster->seed();

          // Find the seed rec hit
          std::vector<DetId> sIds = sCluster->getHitsByDetId();

          float eMax, e2nd;
          EcalRecHitCollection::const_iterator seedItr = eeRecHits->begin();
          EcalRecHitCollection::const_iterator secondItr = eeRecHits->begin();

          for(std::vector<DetId>::const_iterator idItr = sIds.begin(); idItr != sIds.end(); ++idItr) {
            if(idItr->det() != DetId::Ecal) { continue; }
            EcalRecHitCollection::const_iterator hitItr = eeRecHits->find((*idItr));
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

          if(sIds.size() == 1)
            meSCMapSingleCrystal_[eeSide]->Fill(xeex, xeey);

          edm::ESHandle<EcalADCToGeVConstant> pAgc;
          c.get<EcalADCToGeVConstantRcd>().get(pAgc);
          const EcalADCToGeVConstant* agc = pAgc.product();
          
          if(pAgc.isValid()) {
            if(seedItr->energy() / agc->getEEValue() > 16) {
              
              float time = seedItr->time() + 5.0;
              
              meSCSeedTimingSummary_->Fill( time );
              meSCSeedTiming_[ism-1]->Fill( time );
              meSCSeedTimingMap_[eeSide]->Fill(xeex, xeey, time);

            }
          } else {
            LogWarning("EEClusterTask") << "EcalADCToGeVConstant not valid";
          }

          mes1s9_->Fill( eMax/e3x3 );
          mes9s25_->Fill( e3x3/e5x5 );
        }
        else {
          LogWarning("EEClusterTask") << "CaloTopology not valid";
        }
      }
      else {
        LogWarning("EEClusterTask") << EcalRecHitCollection_ << " not available";
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
      float mass = sum.M();
      if ( mass < 0.3 ) {
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

    LogWarning("EEClusterTask") << SuperClusterCollection_ << " not available";

  }

}
