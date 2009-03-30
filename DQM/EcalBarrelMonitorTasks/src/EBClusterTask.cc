/*
 * \file EBClusterTask.cc
 *
 * $Date: 2009/02/26 19:51:36 $
 * $Revision: 1.74 $
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
#include "CondFormats/EcalObjects/interface/EcalADCToGeVConstant.h"
#include "CondFormats/DataRecord/interface/EcalADCToGeVConstantRcd.h"

#include <DQM/EcalCommon/interface/Numbers.h>

#include <DQM/EcalBarrelMonitorTasks/interface/EBClusterTask.h>

#include "DataFormats/EcalDetId/interface/EBDetId.h"

#include "TLorentzVector.h"

using namespace cms;
using namespace edm;
using namespace std;
using namespace reco;

EBClusterTask::EBClusterTask(const ParameterSet& ps){

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

  meSCCrystalSiz_ = 0;
  meSCSeedEne_ = 0;
  meSCEne2_ = 0;
  meSCEneVsEMax_ = 0;
  meSCEneLowScale_ = 0;
  meSCSeedMapOcc_ = 0;
  meSCMapSingleCrystal_ = 0;
  meSCSeedTimingSummary_ = 0;
  meSCSeedTimingMap_ = 0;
  for(int i=0;i<36;++i)
     meSCSeedTiming_[i] = 0;
  
  mes1s9_  = 0;
  mes9s25_  = 0;
 
  meInvMassPi0_ = 0;
  meInvMassJPsi_ = 0;
  meInvMassZ0_ = 0;
  meInvMassHigh_ = 0;

}

EBClusterTask::~EBClusterTask(){

}

void EBClusterTask::beginJob(const EventSetup& c){

  ievt_ = 0;

  if ( dqmStore_ ) {
    dqmStore_->setCurrentFolder(prefixME_ + "/EBClusterTask");
    dqmStore_->rmdir(prefixME_ + "/EBClusterTask");
  }

  Numbers::initGeometry(c, false);

}

void EBClusterTask::beginRun(const Run& r, const EventSetup& c) {

  if ( ! mergeRuns_ ) this->reset();

}

void EBClusterTask::endRun(const Run& r, const EventSetup& c) {

}

void EBClusterTask::reset(void) {

  if ( meBCEne_ ) meBCEne_->Reset();

  if ( meBCNum_ ) meBCNum_->Reset();

  if ( meBCSiz_ ) meBCSiz_->Reset();

  if ( meBCEneMap_ ) meBCEneMap_->Reset();

  if ( meBCNumMap_ ) meBCNumMap_->Reset();

  if ( meBCETMap_ ) meBCETMap_->Reset();

  if ( meBCSizMap_ ) meBCSizMap_->Reset();

  if ( meBCEneMapProjEta_ ) meBCEneMapProjEta_->Reset();

  if ( meBCEneMapProjPhi_ ) meBCEneMapProjPhi_->Reset();

  if ( meBCNumMapProjEta_ ) meBCNumMapProjEta_->Reset();

  if ( meBCNumMapProjPhi_ ) meBCNumMapProjPhi_->Reset();

  if ( meBCETMapProjEta_ ) meBCETMapProjEta_->Reset();

  if ( meBCETMapProjPhi_ ) meBCETMapProjPhi_->Reset();

  if ( meBCSizMapProjEta_ ) meBCSizMapProjEta_->Reset();

  if ( meBCSizMapProjPhi_ ) meBCSizMapProjPhi_->Reset();

  if ( meSCEne_ ) meSCEne_->Reset();

  if ( meSCNum_ ) meSCNum_->Reset();

  if ( meSCSiz_ ) meSCSiz_->Reset();

  if ( meSCCrystalSiz_ ) meSCCrystalSiz_->Reset();

  if ( meSCSeedEne_ ) meSCSeedEne_->Reset();
  
  if ( meSCEne2_ ) meSCEne2_->Reset();

  if ( meSCEneVsEMax_ ) meSCEneVsEMax_->Reset();

  if ( meSCEneLowScale_ ) meSCEneLowScale_->Reset();
  
  if ( meSCSeedMapOcc_ ) meSCSeedMapOcc_->Reset();

  if ( meSCMapSingleCrystal_ ) meSCMapSingleCrystal_->Reset();

  if ( meSCSeedTimingSummary_ ) meSCSeedTimingSummary_->Reset();

  if ( meSCSeedTimingMap_ ) meSCSeedTimingMap_->Reset();

  for(int i=0;i<36;++i)
     if( meSCSeedTiming_[i] ) meSCSeedTiming_[i]->Reset();

  if ( mes1s9_ ) mes1s9_->Reset();

  if ( mes9s25_ ) mes9s25_->Reset();

  if ( meInvMassPi0_ ) meInvMassPi0_->Reset();

  if ( meInvMassJPsi_ ) meInvMassJPsi_->Reset();

  if ( meInvMassZ0_ ) meInvMassZ0_->Reset();

  if ( meInvMassHigh_ ) meInvMassHigh_->Reset();

}

void EBClusterTask::setup(void){

  init_ = true;

  char histo[200];

  if ( dqmStore_ ) {
    dqmStore_->setCurrentFolder(prefixME_ + "/EBClusterTask");

    sprintf(histo, "EBCLT BC energy");
    meBCEne_ = dqmStore_->book1D(histo, histo, 100, 0., 150.);
    meBCEne_->setAxisTitle("energy (GeV)", 1);

    sprintf(histo, "EBCLT BC number");
    meBCNum_ = dqmStore_->book1D(histo, histo, 100, 0., 100.);
    meBCNum_->setAxisTitle("number of clusters", 1);

    sprintf(histo, "EBCLT BC size");
    meBCSiz_ = dqmStore_->book1D(histo, histo, 100, 0., 100.);
    meBCSiz_->setAxisTitle("cluster size", 1);

    sprintf(histo, "EBCLT BC energy map");
    meBCEneMap_ = dqmStore_->bookProfile2D(histo, histo, 72, -M_PI*(9+1.5)/9, M_PI*(9-1.5)/9, 34, -1.479, 1.479, 100, 0., 500., "s");
    meBCEneMap_->setAxisTitle("phi", 1);
    meBCEneMap_->setAxisTitle("eta", 2);

    sprintf(histo, "EBCLT BC number map");
    meBCNumMap_ = dqmStore_->book2D(histo, histo, 72, -M_PI*(9+1.5)/9, M_PI*(9-1.5)/9, 34, -1.479, 1.479);
    meBCNumMap_->setAxisTitle("phi", 1);
    meBCNumMap_->setAxisTitle("eta", 2);

    sprintf(histo, "EBCLT BC ET map");
    meBCETMap_ = dqmStore_->bookProfile2D(histo, histo, 72, -M_PI*(9+1.5)/9, M_PI*(9-1.5)/9, 34, -1.479, 1.479, 100, 0., 500., "s");
    meBCETMap_->setAxisTitle("phi", 1);
    meBCETMap_->setAxisTitle("eta", 2);

    sprintf(histo, "EBCLT BC size map");
    meBCSizMap_ = dqmStore_->bookProfile2D(histo, histo, 72, -M_PI*(9+1.5)/9, M_PI*(9-1.5)/9, 34, -1.479, 1.479, 100, 0., 100., "s");
    meBCSizMap_->setAxisTitle("phi", 1);
    meBCSizMap_->setAxisTitle("eta", 2);

    sprintf(histo, "EBCLT BC energy projection eta");
    meBCEneMapProjEta_ = dqmStore_->bookProfile(histo, histo, 34, -1.479, 1.479, 100, 0., 500., "s");
    meBCEneMapProjEta_->setAxisTitle("eta", 1);
    meBCEneMapProjEta_->setAxisTitle("energy (GeV)", 2);

    sprintf(histo, "EBCLT BC energy projection phi");
    meBCEneMapProjPhi_ = dqmStore_->bookProfile(histo, histo, 72, -M_PI*(9+1.5)/9, M_PI*(9-1.5)/9, 100, 0., 500., "s");
    meBCEneMapProjPhi_->setAxisTitle("phi", 1);
    meBCEneMapProjPhi_->setAxisTitle("energy (GeV)", 2);

    sprintf(histo, "EBCLT BC number projection eta");
    meBCNumMapProjEta_ = dqmStore_->book1D(histo, histo, 34, -1.479, 1.479);
    meBCNumMapProjEta_->setAxisTitle("eta", 1);
    meBCNumMapProjEta_->setAxisTitle("number of clusters", 2);

    sprintf(histo, "EBCLT BC number projection phi");
    meBCNumMapProjPhi_ = dqmStore_->book1D(histo, histo, 72, -M_PI*(9+1.5)/9, M_PI*(9-1.5)/9);
    meBCNumMapProjPhi_->setAxisTitle("phi", 1);
    meBCNumMapProjPhi_->setAxisTitle("number of clusters", 2);

    sprintf(histo, "EBCLT BC ET projection eta");
    meBCETMapProjEta_ = dqmStore_->bookProfile(histo, histo, 34, -1.479, 1.479, 100, 0., 500., "s");
    meBCETMapProjEta_->setAxisTitle("eta", 1);
    meBCETMapProjEta_->setAxisTitle("transverse energy (GeV)", 2);

    sprintf(histo, "EBCLT BC ET projection phi");
    meBCETMapProjPhi_ = dqmStore_->bookProfile(histo, histo, 72, -M_PI*(9+1.5)/9, M_PI*(9-1.5)/9, 100, 0., 500., "s");
    meBCETMapProjPhi_->setAxisTitle("phi", 1);
    meBCETMapProjPhi_->setAxisTitle("transverse energy (GeV)", 2);

    sprintf(histo, "EBCLT BC size projection eta");
    meBCSizMapProjEta_ = dqmStore_->bookProfile(histo, histo, 34, -1.479, 1.479, 100, 0., 100., "s");
    meBCSizMapProjEta_->setAxisTitle("eta", 1);
    meBCSizMapProjEta_->setAxisTitle("cluster size", 2);

    sprintf(histo, "EBCLT BC size projection phi");
    meBCSizMapProjPhi_ = dqmStore_->bookProfile(histo, histo, 72, -M_PI*(9+1.5)/9, M_PI*(9-1.5)/9, 100, 0., 100., "s");
    meBCSizMapProjPhi_->setAxisTitle("phi", 1);
    meBCSizMapProjPhi_->setAxisTitle("cluster size", 2);

    sprintf(histo, "EBCLT SC energy");
    meSCEne_ = dqmStore_->book1D(histo, histo, 100, 0., 150.);
    meSCEne_->setAxisTitle("energy (GeV)", 1);

    sprintf(histo, "EBCLT SC number");
    meSCNum_ = dqmStore_->book1D(histo, histo, 50, 0., 50.);
    meSCNum_->setAxisTitle("number of clusters", 1);

    sprintf(histo, "EBCLT SC size");
    meSCSiz_ = dqmStore_->book1D(histo, histo, 50, 0., 50.);
    meSCSiz_->setAxisTitle("cluster size", 1);

    sprintf(histo, "EBCLT SC size (crystal)");
    meSCCrystalSiz_ = dqmStore_->book1D(histo, histo, 150, 0, 150);
    meSCCrystalSiz_->setAxisTitle("cluster size in crystals", 1);

    sprintf(histo, "EBCLT SC seed crystal energy");
    meSCSeedEne_ = dqmStore_->book1D(histo, histo, 200, 0, 1.8);
    meSCSeedEne_->setAxisTitle("seed crystal energy (GeV)", 1);

    sprintf(histo, "EBCLT SC e2");
    meSCEne2_ = dqmStore_->book1D(histo, histo, 200, 0, 1.8);
    meSCEne2_->setAxisTitle("seed + highest neighbor crystal energy (GeV)", 1);

    sprintf(histo, "EBCLT SC energy vs seed crystal energy");
    meSCEneVsEMax_ = dqmStore_->book2D(histo, histo, 200, 0, 1.8, 200, 0, 1.8);
    meSCEneVsEMax_->setAxisTitle("seed crystal energy (GeV)", 1);
    meSCEneVsEMax_->setAxisTitle("cluster energy (GeV)", 2);

    sprintf(histo, "EBCLT SC energy (low scale)");
    meSCEneLowScale_ = dqmStore_->book1D(histo, histo, 200, 0, 1.8);
    meSCEneLowScale_->setAxisTitle("cluster energy (GeV)", 1);

    sprintf(histo, "EBCLT SC seed occupancy map");
    meSCSeedMapOcc_ = dqmStore_->book2D(histo, histo, 72, 0., 360., 34, -85, 85);
    meSCSeedMapOcc_->setAxisTitle("jphi", 1);
    meSCSeedMapOcc_->setAxisTitle("jeta", 2);

    sprintf(histo, "EBCLT SC single crystal cluster seed occupancy map");
    meSCMapSingleCrystal_ = dqmStore_->book2D(histo, histo, 72, 0., 360., 34, -85, 85);
    meSCMapSingleCrystal_->setAxisTitle("jphi", 1);
    meSCMapSingleCrystal_->setAxisTitle("jeta", 2);

    sprintf(histo, "EBCLT SC seed crystal timing");
    meSCSeedTimingSummary_ = dqmStore_->book1D(histo, histo, 78, 0, 10);
    meSCSeedTimingSummary_->setAxisTitle("seed crystal timing", 1);

    sprintf(histo, "EBCLT SC seed crystal timing map");
    meSCSeedTimingMap_ = dqmStore_->bookProfile2D(histo, histo, 72, 0., 360., 34, -85, 85, 78, 0., 10., "s");
    meSCSeedTimingMap_->setAxisTitle("jphi", 1);
    meSCSeedTimingMap_->setAxisTitle("jeta", 2);

    dqmStore_->setCurrentFolder(prefixME_ + "/EBClusterTask/Timing");
    for(int i = 0; i<36; i++) {
      sprintf(histo,"EBCLT timing %s", Numbers::sEB(i+1).c_str());
      meSCSeedTiming_[i] = dqmStore_->book1D(histo, histo, 100, 0., 10.);
    }
    dqmStore_->setCurrentFolder(prefixME_ + "/EBClusterTask");

    sprintf(histo, "EBCLT s1s9");
    mes1s9_ = dqmStore_->book1D(histo, histo, 50, 0., 1.5);
    mes1s9_->setAxisTitle("s1/s9", 1);

    sprintf(histo, "EBCLT s9s25");
    mes9s25_ = dqmStore_->book1D(histo, histo, 75, 0., 1.5);
    mes9s25_->setAxisTitle("s9/s25", 1);

    sprintf(histo, "EBCLT dicluster invariant mass Pi0");
    meInvMassPi0_ = dqmStore_->book1D(histo, histo, 50, 0., 0.300);
    meInvMassPi0_->setAxisTitle("mass (GeV)", 1);

    sprintf(histo, "EBCLT dicluster invariant mass JPsi");
    meInvMassJPsi_ = dqmStore_->book1D(histo, histo, 50, 2.9, 3.3);
    meInvMassJPsi_->setAxisTitle("mass (GeV)", 1);

    sprintf(histo, "EBCLT dicluster invariant mass Z0");
    meInvMassZ0_ = dqmStore_->book1D(histo, histo, 50, 40, 110);
    meInvMassZ0_->setAxisTitle("mass (GeV)", 1);

    sprintf(histo, "EBCLT dicluster invariant mass high");
    meInvMassHigh_ = dqmStore_->book1D(histo, histo, 500, 110, 3000);
    meInvMassHigh_->setAxisTitle("mass (GeV)", 1);

  }

}

void EBClusterTask::cleanup(void){

  if ( ! init_ ) return;

  if ( dqmStore_ ) {
    dqmStore_->setCurrentFolder(prefixME_ + "/EBClusterTask");

    if ( meBCEne_ ) dqmStore_->removeElement( meBCEne_->getName() );
    meBCEne_ = 0;

    if ( meBCNum_ ) dqmStore_->removeElement( meBCNum_->getName() );
    meBCNum_ = 0;

    if ( meBCSiz_ ) dqmStore_->removeElement( meBCSiz_->getName() );
    meBCSiz_ = 0;

    if ( meBCEneMap_ ) dqmStore_->removeElement( meBCEneMap_->getName() );
    meBCEneMap_ = 0;

    if ( meBCNumMap_ ) dqmStore_->removeElement( meBCNumMap_->getName() );
    meBCNumMap_ = 0;

    if ( meBCETMap_ ) dqmStore_->removeElement( meBCETMap_->getName() );
    meBCETMap_ = 0;

    if ( meBCSizMap_ ) dqmStore_->removeElement( meBCSizMap_->getName() );
    meBCSizMap_ = 0;

    if ( meBCEneMapProjEta_ ) dqmStore_->removeElement( meBCEneMapProjEta_->getName() );
    meBCEneMapProjEta_ = 0;

    if ( meBCEneMapProjPhi_ ) dqmStore_->removeElement( meBCEneMapProjPhi_->getName() );
    meBCEneMapProjPhi_ = 0;

    if ( meBCNumMapProjEta_ ) dqmStore_->removeElement( meBCNumMapProjEta_->getName() );
    meBCNumMapProjEta_ = 0;

    if ( meBCNumMapProjPhi_ ) dqmStore_->removeElement( meBCNumMapProjPhi_->getName() );
    meBCNumMapProjPhi_ = 0;

    if ( meBCETMapProjEta_ ) dqmStore_->removeElement( meBCETMapProjEta_->getName() );
    meBCETMapProjEta_ = 0;

    if ( meBCETMapProjPhi_ ) dqmStore_->removeElement( meBCETMapProjPhi_->getName() );
    meBCETMapProjPhi_ = 0;

    if ( meBCSizMapProjEta_ ) dqmStore_->removeElement( meBCSizMapProjEta_->getName() );
    meBCSizMapProjEta_ = 0;

    if ( meBCSizMapProjPhi_ ) dqmStore_->removeElement( meBCSizMapProjPhi_->getName() );
    meBCSizMapProjPhi_ = 0;

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

    if ( meSCSeedMapOcc_ ) dqmStore_->removeElement( meSCSeedMapOcc_->getName() );
    meSCSeedMapOcc_ = 0;

    if ( meSCMapSingleCrystal_ ) dqmStore_->removeElement( meSCMapSingleCrystal_->getName() );
    meSCMapSingleCrystal_ = 0;

    if ( meSCSeedTimingSummary_ ) dqmStore_->removeElement( meSCSeedTimingSummary_->getName() );
    meSCSeedTimingSummary_ = 0;

    if ( meSCSeedTimingMap_ ) dqmStore_->removeElement( meSCSeedTimingMap_->getName() );
    meSCSeedTimingMap_ = 0;

    dqmStore_->setCurrentFolder(prefixME_ + "/EBClusterTask/Timing");
    for(int i = 0; i < 36; i++) {
       if ( meSCSeedTiming_[i] ) dqmStore_->removeElement( meSCSeedTiming_[i]->getName() );
       meSCSeedTiming_[i] = 0;
    }
    dqmStore_->setCurrentFolder(prefixME_ + "/EBClusterTask");

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

void EBClusterTask::endJob(void){

  LogInfo("EBClusterTask") << "analyzed " << ievt_ << " events";

  if ( enableCleanup_ ) this->cleanup();

}

void EBClusterTask::analyze(const Event& e, const EventSetup& c){

  bool enable = false;

  Handle<EcalRawDataCollection> dcchs;

  if ( e.getByLabel(EcalRawDataCollection_, dcchs) ) {

    for ( EcalRawDataCollection::const_iterator dcchItr = dcchs->begin(); dcchItr != dcchs->end(); ++dcchItr ) {

      if ( Numbers::subDet( *dcchItr ) != EcalBarrel ) continue;

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
    LogWarning("EBClusterTask") << EcalRawDataCollection_ << " not available";

  }

  if ( ! enable ) return;

  if ( ! init_ ) this->setup();

  ievt_++;

  // --- Barrel Basic Clusters ---

  Handle<BasicClusterCollection> pBasicClusters;

  if ( e.getByLabel(BasicClusterCollection_, pBasicClusters) ) {

    int nbcc = pBasicClusters->size();
    if ( nbcc > 0 ) meBCNum_->Fill(float(nbcc));

    for ( BasicClusterCollection::const_iterator bCluster = pBasicClusters->begin(); bCluster != pBasicClusters->end(); ++bCluster ) {

      meBCEne_->Fill(bCluster->energy());
      meBCSiz_->Fill(float(bCluster->size()));

      float xphi = bCluster->phi();
      if ( xphi > M_PI*(9-1.5)/9 ) xphi = xphi - M_PI*2;

      meBCEneMap_->Fill(xphi, bCluster->eta(), bCluster->energy());
      meBCEneMapProjEta_->Fill(bCluster->eta(), bCluster->energy());
      meBCEneMapProjPhi_->Fill(xphi, bCluster->energy());

      meBCNumMap_->Fill(xphi, bCluster->eta());
      meBCNumMapProjEta_->Fill(bCluster->eta());
      meBCNumMapProjPhi_->Fill(xphi);

      meBCSizMap_->Fill(xphi, bCluster->eta(), float(bCluster->size()));
      meBCSizMapProjEta_->Fill(bCluster->eta(), float(bCluster->size()));
      meBCSizMapProjPhi_->Fill(xphi, float(bCluster->size()));

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

    TLorentzVector sc1_p(0,0,0,0);
    TLorentzVector sc2_p(0,0,0,0);

    for ( SuperClusterCollection::const_iterator sCluster = pSuperClusters->begin(); sCluster != pSuperClusters->end(); ++sCluster ) {

      // energy, size
      meSCEne_->Fill( sCluster->energy() );
      meSCSiz_->Fill( float(sCluster->clustersSize()) );

      // seed and shapes
      edm::Handle< EcalRecHitCollection > pEBRecHits;
      e.getByLabel( EcalRecHitCollection_, pEBRecHits );
      if ( pEBRecHits.isValid() ) {
        const EcalRecHitCollection *ebRecHits = pEBRecHits.product();

	edm::ESHandle<CaloTopology> pTopology;
        c.get<CaloTopologyRecord>().get(pTopology);
        if ( pTopology.isValid() ) {
          const CaloTopology *topology = pTopology.product();

          // <= CMSSW_3_0_X
          BasicClusterRef theSeed = sCluster->seed();
          // >= CMSSW_3_1_X
          CaloClusterPtr theSeed = sCluster->seed();

	  // Find the seed rec hit
          // <= CMSSW_3_0_X
          // std::vector<DetId> sIds = sCluster->getHitsByDetId();
          // >= CMSSW_3_1_X
          std::vector< std::pair<DetId,float> > sIds = sCluster->hitsAndFractions();

	  float eMax, e2nd;
	  EcalRecHitCollection::const_iterator seedItr = ebRecHits->begin();
	  EcalRecHitCollection::const_iterator secondItr = ebRecHits->begin();

          // <= CMSSW_3_0_X
          // for(std::vector<DetId>::const_iterator idItr = sIds.begin(); idItr != sIds.end(); ++idItr) {
          // if(idItr->det() != DetId::Ecal) { continue; }
          // EcalRecHitCollection::const_iterator hitItr = ebRecHits->find((*idItr));
          // <= CMSSW_3_1_X
	  for(std::vector< std::pair<DetId,float> >::const_iterator idItr = sIds.begin(); idItr != sIds.end(); ++idItr) {
            DetId id = idItr->first;
            if(id.det() != DetId::Ecal) { continue; }
            EcalRecHitCollection::const_iterator hitItr = ebRecHits->find(id);
            if(hitItr == ebRecHits->end()) { continue; }
            if(hitItr->energy() > secondItr->energy()) { secondItr = hitItr; }
            if(hitItr->energy() > seedItr->energy()) { std::swap(seedItr,secondItr); }
	  }

	  eMax = seedItr->energy();
	  e2nd = secondItr->energy();
	  EBDetId seedId = (EBDetId) seedItr->id();

          float e3x3 = EcalClusterTools::e3x3( *theSeed, ebRecHits, topology );
          float e5x5 = EcalClusterTools::e5x5( *theSeed, ebRecHits, topology );

	  meSCCrystalSiz_->Fill(sIds.size());
	  meSCSeedEne_->Fill(eMax);
	  meSCEne2_->Fill(eMax+e2nd);
	  meSCEneVsEMax_->Fill(eMax,sCluster->energy());
	  meSCEneLowScale_->Fill(sCluster->energy());
	  
	  // Prepare to fill maps
	  int ism = Numbers::iSM(seedId);
	  int ebeta = seedId.ieta();
	  int ebphi = seedId.iphi();
	  float xebeta = ebeta - 0.5 * seedId.zside();
	  float xebphi = ebphi - 0.5;
	  meSCSeedMapOcc_->Fill(xebphi,xebeta);
	  if(sIds.size() == 1)
	     meSCMapSingleCrystal_->Fill(xebphi,xebeta);

          float time = seedItr->time() + 5.0;

          edm::ESHandle<EcalADCToGeVConstant> pAgc;
          c.get<EcalADCToGeVConstantRcd>().get(pAgc);
          if(pAgc.isValid()) {
            const EcalADCToGeVConstant* agc = pAgc.product();
            
            if(seedItr->energy() / agc->getEBValue() > 12) {

              meSCSeedTimingSummary_->Fill( time );
              meSCSeedTiming_[ism-1]->Fill( time );
              meSCSeedTimingMap_->Fill( xebphi, xebeta, time );

            }
          } else {
            LogWarning("EBClusterTask") << "EcalADCToGeVConstant not valid";
          }

          mes1s9_->Fill( eMax/e3x3 );
          mes9s25_->Fill( e3x3/e5x5 );
        }
        else {
          LogWarning("EBClusterTask") << "CaloTopology not valid";
        }
      }
      else {
        LogWarning("EBClusterTask") << EcalRecHitCollection_ << " not available";
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

    LogWarning("EBClusterTask") << SuperClusterCollection_ << " not available";

  }

}
