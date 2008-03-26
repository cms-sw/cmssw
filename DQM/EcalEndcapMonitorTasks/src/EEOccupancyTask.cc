/*
 * \file EEOccupancyTask.cc
 *
 * $Date: 2008/01/27 21:02:09 $
 * $Revision: 1.38 $
 * \author G. Della Ricca
 * \author G. Franzoni
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

#include "DataFormats/EcalRawData/interface/EcalRawDataCollections.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "DataFormats/EcalDigi/interface/EEDataFrame.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "DataFormats/EcalRecHit/interface/EcalUncalibratedRecHit.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"

#include <DQM/EcalCommon/interface/Numbers.h>

#include <DQM/EcalEndcapMonitorTasks/interface/EEOccupancyTask.h>

using namespace cms;
using namespace edm;
using namespace std;

EEOccupancyTask::EEOccupancyTask(const ParameterSet& ps){

  init_ = false;

  // get hold of back-end interface
  dbe_ = Service<DaqMonitorBEInterface>().operator->();

  enableCleanup_ = ps.getUntrackedParameter<bool>("enableCleanup", false);

  EEDigiCollection_ = ps.getParameter<edm::InputTag>("EEDigiCollection");
  EcalPnDiodeDigiCollection_ = ps.getParameter<edm::InputTag>("EcalPnDiodeDigiCollection");
  EcalRecHitCollection_ = ps.getParameter<edm::InputTag>("EcalRecHitCollection");
  EcalTrigPrimDigiCollection_ = ps.getParameter<edm::InputTag>("EcalTrigPrimDigiCollection");

  for (int i = 0; i < 18; i++) {
    meOccupancy_[i]    = 0;
    meOccupancyMem_[i] = 0;
  }

  meEEDigiOccupancy_[0] = 0;
  meEEDigiOccupancyProR_[0] = 0;
  meEEDigiOccupancyProPhi_[0] = 0;
  meEEDigiOccupancy_[1] = 0;
  meEEDigiOccupancyProR_[1] = 0;
  meEEDigiOccupancyProPhi_[1] = 0;

  meEERecHitOccupancy_[0] = 0;
  meEERecHitOccupancyProR_[0] = 0;
  meEERecHitOccupancyProPhi_[0] = 0;
  meEERecHitOccupancy_[1] = 0;
  meEERecHitOccupancyProR_[1] = 0;
  meEERecHitOccupancyProPhi_[1] = 0;

  meEERecHitOccupancyThr_[0] = 0;
  meEERecHitOccupancyProRThr_[0] = 0;
  meEERecHitOccupancyProPhiThr_[0] = 0;
  meEERecHitOccupancyThr_[1] = 0;
  meEERecHitOccupancyProRThr_[1] = 0;
  meEERecHitOccupancyProPhiThr_[1] = 0;

  meEETrigPrimDigiOccupancy_[0] = 0;
  meEETrigPrimDigiOccupancyProR_[0] = 0;
  meEETrigPrimDigiOccupancyProPhi_[0] = 0;
  meEETrigPrimDigiOccupancy_[1] = 0;
  meEETrigPrimDigiOccupancyProR_[1] = 0;
  meEETrigPrimDigiOccupancyProPhi_[1] = 0;

  meEETrigPrimDigiOccupancyThr_[0] = 0;
  meEETrigPrimDigiOccupancyProRThr_[0] = 0;
  meEETrigPrimDigiOccupancyProPhiThr_[0] = 0;
  meEETrigPrimDigiOccupancyThr_[1] = 0;
  meEETrigPrimDigiOccupancyProRThr_[1] = 0;
  meEETrigPrimDigiOccupancyProPhiThr_[1] = 0;

  recHitEnergyMin_ = 1.;
  trigPrimEtMin_ = 5.;

}

EEOccupancyTask::~EEOccupancyTask(){

}

void EEOccupancyTask::beginJob(const EventSetup& c){

  ievt_ = 0;

  if ( dbe_ ) {
    dbe_->setCurrentFolder("EcalEndcap/EEOccupancyTask");
    dbe_->rmdir("EcalEndcap/EEOccupancyTask");
  }

  Numbers::initGeometry(c);

}

void EEOccupancyTask::setup(void){

  init_ = true;

  char histo[200];

  if ( dbe_ ) {
    dbe_->setCurrentFolder("EcalEndcap/EEOccupancyTask");

    for (int i = 0; i < 18; i++) {
      sprintf(histo, "EEOT digi occupancy %s", Numbers::sEE(i+1).c_str());
      meOccupancy_[i] = dbe_->book2D(histo, histo, 50, Numbers::ix0EE(i+1)+0., Numbers::ix0EE(i+1)+50., 50, Numbers::iy0EE(i+1)+0., Numbers::iy0EE(i+1)+50.);
      meOccupancy_[i]->setAxisTitle("jx", 1);
      meOccupancy_[i]->setAxisTitle("jy", 2);
      dbe_->tag(meOccupancy_[i], i+1);
    }
    for (int i = 0; i < 18; i++) {
      sprintf(histo, "EEOT MEM digi occupancy %s", Numbers::sEE(i+1).c_str());
      meOccupancyMem_[i] = dbe_->book2D(histo, histo, 10, 0., 10., 5, 0., 5.);
      meOccupancyMem_[i]->setAxisTitle("pseudo-strip", 1);
      meOccupancyMem_[i]->setAxisTitle("channel", 2);
      dbe_->tag(meOccupancyMem_[i], i+1);
    }

    sprintf(histo, "EEOT digi occupancy EE -");
    meEEDigiOccupancy_[0] = dbe_->book2D(histo, histo, 100, 0., 100., 100, 0., 100.);
    meEEDigiOccupancy_[0]->setAxisTitle("jx", 1);
    meEEDigiOccupancy_[0]->setAxisTitle("jy", 2);
    sprintf(histo, "EEOT digi occupancy EE - projection R");
    meEEDigiOccupancyProR_[0] = dbe_->book1D(histo, histo, 22, 0., 55.);
    meEEDigiOccupancyProR_[0]->setAxisTitle("r", 1);
    meEEDigiOccupancyProR_[0]->setAxisTitle("number of digis", 2);
    sprintf(histo, "EEOT digi occupancy EE - projection phi");
    meEEDigiOccupancyProPhi_[0] = dbe_->book1D(histo, histo, 50, -M_PI, M_PI);
    meEEDigiOccupancyProPhi_[0]->setAxisTitle("phi'", 1);
    meEEDigiOccupancyProPhi_[0]->setAxisTitle("number of digis", 2);

    sprintf(histo, "EEOT digi occupancy EE +");
    meEEDigiOccupancy_[1] = dbe_->book2D(histo, histo, 100, 0., 100., 100, 0., 100.);
    meEEDigiOccupancy_[1]->setAxisTitle("jx", 1);
    meEEDigiOccupancy_[1]->setAxisTitle("jy", 2);
    sprintf(histo, "EEOT digi occupancy EE + projection R");
    meEEDigiOccupancyProR_[1] = dbe_->book1D(histo, histo, 22, 0., 55.);
    meEEDigiOccupancyProR_[1]->setAxisTitle("r", 1);
    meEEDigiOccupancyProR_[1]->setAxisTitle("number of digis", 2);
    sprintf(histo, "EEOT digi occupancy EE + projection phi");
    meEEDigiOccupancyProPhi_[1] = dbe_->book1D(histo, histo, 50, -M_PI, M_PI);
    meEEDigiOccupancyProPhi_[1]->setAxisTitle("phi'", 1);
    meEEDigiOccupancyProPhi_[1]->setAxisTitle("number of digis", 2);

    sprintf(histo, "EEOT rec hit occupancy EE -");
    meEERecHitOccupancy_[0] = dbe_->book2D(histo, histo, 100, 0., 100., 100, 0., 100.);
    meEERecHitOccupancy_[0]->setAxisTitle("jx", 1);
    meEERecHitOccupancy_[0]->setAxisTitle("jy", 2);
    sprintf(histo, "EEOT rec hit occupancy EE - projection R");
    meEERecHitOccupancyProR_[0] = dbe_->book1D(histo, histo, 22, 0., 55.);
    meEERecHitOccupancyProR_[0]->setAxisTitle("r", 1);
    meEERecHitOccupancyProR_[0]->setAxisTitle("number of hits", 2);
    sprintf(histo, "EEOT rec hit occupancy EE - projection phi");
    meEERecHitOccupancyProPhi_[0] = dbe_->book1D(histo, histo, 50, -M_PI, M_PI);
    meEERecHitOccupancyProPhi_[0]->setAxisTitle("phi'", 1);
    meEERecHitOccupancyProPhi_[0]->setAxisTitle("number of hits", 2);

    sprintf(histo, "EEOT rec hit occupancy EE +");
    meEERecHitOccupancy_[1] = dbe_->book2D(histo, histo, 100, 0., 100., 100, 0., 100.);
    meEERecHitOccupancy_[1]->setAxisTitle("jx", 1);
    meEERecHitOccupancy_[1]->setAxisTitle("jy", 2);
    sprintf(histo, "EEOT rec hit occupancy EE + projection R");
    meEERecHitOccupancyProR_[1] = dbe_->book1D(histo, histo, 22, 0., 55.);
    meEERecHitOccupancyProR_[1]->setAxisTitle("r", 1);
    meEERecHitOccupancyProR_[1]->setAxisTitle("number of hits", 2);
    sprintf(histo, "EEOT rec hit occupancy EE + projection phi");
    meEERecHitOccupancyProPhi_[1] = dbe_->book1D(histo, histo, 50, -M_PI, M_PI);
    meEERecHitOccupancyProPhi_[1]->setAxisTitle("phi'", 1);
    meEERecHitOccupancyProPhi_[1]->setAxisTitle("number of hits", 2);

    sprintf(histo, "EEOT rec hit thr occupancy EE -");
    meEERecHitOccupancyThr_[0] = dbe_->book2D(histo, histo, 100, 0., 100., 100, 0., 100.);
    meEERecHitOccupancyThr_[0]->setAxisTitle("jx", 1);
    meEERecHitOccupancyThr_[0]->setAxisTitle("jy", 2);
    sprintf(histo, "EEOT rec hit thr occupancy EE - projection R");
    meEERecHitOccupancyProRThr_[0] = dbe_->book1D(histo, histo, 22, 0., 55.);
    meEERecHitOccupancyProRThr_[0]->setAxisTitle("r", 1);
    meEERecHitOccupancyProRThr_[0]->setAxisTitle("number of hits", 2);
    sprintf(histo, "EEOT rec hit thr occupancy EE - projection phi");
    meEERecHitOccupancyProPhiThr_[0] = dbe_->book1D(histo, histo, 50, -M_PI, M_PI);
    meEERecHitOccupancyProPhiThr_[0]->setAxisTitle("phi'", 1);
    meEERecHitOccupancyProPhiThr_[0]->setAxisTitle("number of hits", 2);

    sprintf(histo, "EEOT rec hit thr occupancy EE +");
    meEERecHitOccupancyThr_[1] = dbe_->book2D(histo, histo, 100, 0., 100., 100, 0., 100.);
    meEERecHitOccupancyThr_[1]->setAxisTitle("jx", 1);
    meEERecHitOccupancyThr_[1]->setAxisTitle("jy", 2);
    sprintf(histo, "EEOT rec hit thr occupancy EE + projection R");
    meEERecHitOccupancyProRThr_[1] = dbe_->book1D(histo, histo, 22, 0., 55.);
    meEERecHitOccupancyProRThr_[1]->setAxisTitle("r", 1);
    meEERecHitOccupancyProRThr_[1]->setAxisTitle("number of hits", 2);
    sprintf(histo, "EEOT rec hit thr occupancy EE + projection phi");
    meEERecHitOccupancyProPhiThr_[1] = dbe_->book1D(histo, histo, 50, -M_PI, M_PI);
    meEERecHitOccupancyProPhiThr_[1]->setAxisTitle("phi'", 1);
    meEERecHitOccupancyProPhiThr_[1]->setAxisTitle("number of hits", 2);

    sprintf(histo, "EEOT TP digi occupancy EE -");
    meEETrigPrimDigiOccupancy_[0] = dbe_->book2D(histo, histo, 100, 0., 100., 100, 0., 100.);
    meEETrigPrimDigiOccupancy_[0]->setAxisTitle("jx", 1);
    meEETrigPrimDigiOccupancy_[0]->setAxisTitle("jy", 2);
    sprintf(histo, "EEOT TP digi occupancy EE - projection R");
    meEETrigPrimDigiOccupancyProR_[0] = dbe_->book1D(histo, histo, 22, 0., 55.);
    meEETrigPrimDigiOccupancyProR_[0]->setAxisTitle("r", 1);
    meEETrigPrimDigiOccupancyProR_[0]->setAxisTitle("number of TP digis", 2);
    sprintf(histo, "EEOT TP digi occupancy EE - projection phi");
    meEETrigPrimDigiOccupancyProPhi_[0] = dbe_->book1D(histo, histo, 50, -M_PI, M_PI);
    meEETrigPrimDigiOccupancyProPhi_[0]->setAxisTitle("phi'", 1);
    meEETrigPrimDigiOccupancyProPhi_[0]->setAxisTitle("number of TP digis", 2);

    sprintf(histo, "EEOT TP digi occupancy EE +");
    meEETrigPrimDigiOccupancy_[1] = dbe_->book2D(histo, histo, 100, 0., 100., 100, 0., 100.);
    meEETrigPrimDigiOccupancy_[1]->setAxisTitle("jx", 1);
    meEETrigPrimDigiOccupancy_[1]->setAxisTitle("jy", 2);
    sprintf(histo, "EEOT TP digi occupancy EE + projection R");
    meEETrigPrimDigiOccupancyProR_[1] = dbe_->book1D(histo, histo, 22, 0., 55.);
    meEETrigPrimDigiOccupancyProR_[1]->setAxisTitle("r", 1);
    meEETrigPrimDigiOccupancyProR_[1]->setAxisTitle("number of TP digis", 2);
    sprintf(histo, "EEOT TP digi occupancy EE + projection phi");
    meEETrigPrimDigiOccupancyProPhi_[1] = dbe_->book1D(histo, histo, 50, -M_PI, M_PI);
    meEETrigPrimDigiOccupancyProPhi_[1]->setAxisTitle("phi'", 1);
    meEETrigPrimDigiOccupancyProPhi_[1]->setAxisTitle("number of TP digis", 2);

    sprintf(histo, "EEOT TP digi thr occupancy EE -");
    meEETrigPrimDigiOccupancyThr_[0] = dbe_->book2D(histo, histo, 100, 0., 100., 100, 0., 100.);
    meEETrigPrimDigiOccupancyThr_[0]->setAxisTitle("jx", 1);
    meEETrigPrimDigiOccupancyThr_[0]->setAxisTitle("jy", 2);
    sprintf(histo, "EEOT TP digi thr occupancy EE - projection R");
    meEETrigPrimDigiOccupancyProRThr_[0] = dbe_->book1D(histo, histo, 22, 0., 55.);
    meEETrigPrimDigiOccupancyProRThr_[0]->setAxisTitle("r", 1);
    meEETrigPrimDigiOccupancyProRThr_[0]->setAxisTitle("number of TP digis", 2);
    sprintf(histo, "EEOT TP digi thr occupancy EE - projection phi");
    meEETrigPrimDigiOccupancyProPhiThr_[0] = dbe_->book1D(histo, histo, 50, -M_PI, M_PI);
    meEETrigPrimDigiOccupancyProPhiThr_[0]->setAxisTitle("phi'", 1);
    meEETrigPrimDigiOccupancyProPhiThr_[0]->setAxisTitle("number of TP digis", 2);

    sprintf(histo, "EEOT TP digi thr occupancy EE +");
    meEETrigPrimDigiOccupancyThr_[1] = dbe_->book2D(histo, histo, 100, 0., 100., 100, 0., 100.);
    meEETrigPrimDigiOccupancyThr_[1]->setAxisTitle("jx", 1);
    meEETrigPrimDigiOccupancyThr_[1]->setAxisTitle("jy", 2);
    sprintf(histo, "EEOT TP digi thr occupancy EE + projection R");
    meEETrigPrimDigiOccupancyProRThr_[1] = dbe_->book1D(histo, histo, 22, 0., 55.);
    meEETrigPrimDigiOccupancyProRThr_[1]->setAxisTitle("r", 1);
    meEETrigPrimDigiOccupancyProRThr_[1]->setAxisTitle("number of TP digis", 2);
    sprintf(histo, "EEOT TP digi thr occupancy EE + projection phi");
    meEETrigPrimDigiOccupancyProPhiThr_[1] = dbe_->book1D(histo, histo, 50, -M_PI, M_PI);
    meEETrigPrimDigiOccupancyProPhiThr_[1]->setAxisTitle("phi'", 1);
    meEETrigPrimDigiOccupancyProPhiThr_[1]->setAxisTitle("number of TP digis", 2);

  }

}

void EEOccupancyTask::cleanup(void){

  if ( ! enableCleanup_ ) return;

  if ( dbe_ ) {
    dbe_->setCurrentFolder("EcalEndcap/EEOccupancyTask");

    for (int i = 0; i < 18; i++) {
      if ( meOccupancy_[i] ) dbe_->removeElement( meOccupancy_[i]->getName() );
      meOccupancy_[i] = 0;
      if ( meOccupancyMem_[i] ) dbe_->removeElement( meOccupancyMem_[i]->getName() );
      meOccupancyMem_[i] = 0;
    }

    if ( meEEDigiOccupancy_[0] ) dbe_->removeElement( meEEDigiOccupancy_[0]->getName() );
    meEEDigiOccupancy_[0] = 0;
    if ( meEEDigiOccupancyProR_[0] ) dbe_->removeElement( meEEDigiOccupancyProR_[0]->getName() );
    meEEDigiOccupancyProR_[0] = 0;
    if ( meEEDigiOccupancyProPhi_[0] ) dbe_->removeElement( meEEDigiOccupancyProPhi_[0]->getName() );
    meEEDigiOccupancyProPhi_[0] = 0;

    if ( meEEDigiOccupancy_[1] ) dbe_->removeElement( meEEDigiOccupancy_[1]->getName() );
    meEEDigiOccupancy_[1] = 0;
    if ( meEEDigiOccupancyProR_[1] ) dbe_->removeElement( meEEDigiOccupancyProR_[1]->getName() );
    meEEDigiOccupancyProR_[1] = 0;
    if ( meEEDigiOccupancyProPhi_[1] ) dbe_->removeElement( meEEDigiOccupancyProPhi_[1]->getName() );
    meEEDigiOccupancyProPhi_[1] = 0;

    if ( meEERecHitOccupancy_[0] ) dbe_->removeElement( meEERecHitOccupancy_[0]->getName() );
    meEERecHitOccupancy_[0] = 0;
    if ( meEERecHitOccupancyProR_[0] ) dbe_->removeElement( meEERecHitOccupancyProR_[0]->getName() );
    meEERecHitOccupancyProR_[0] = 0;
    if ( meEERecHitOccupancyProPhi_[0] ) dbe_->removeElement( meEERecHitOccupancyProPhi_[0]->getName() );
    meEERecHitOccupancyProPhi_[0] = 0;

    if ( meEERecHitOccupancy_[1] ) dbe_->removeElement( meEERecHitOccupancy_[1]->getName() );
    meEERecHitOccupancy_[1] = 0;
    if ( meEERecHitOccupancyProR_[1] ) dbe_->removeElement( meEERecHitOccupancyProR_[1]->getName() );
    meEERecHitOccupancyProR_[1] = 0;
    if ( meEERecHitOccupancyProPhi_[1] ) dbe_->removeElement( meEERecHitOccupancyProPhi_[1]->getName() );
    meEERecHitOccupancyProPhi_[1] = 0;

    if ( meEERecHitOccupancyThr_[0] ) dbe_->removeElement( meEERecHitOccupancyThr_[0]->getName() );
    meEERecHitOccupancyThr_[0] = 0;
    if ( meEERecHitOccupancyProRThr_[0] ) dbe_->removeElement( meEERecHitOccupancyProRThr_[0]->getName() );
    meEERecHitOccupancyProRThr_[0] = 0;
    if ( meEERecHitOccupancyProPhiThr_[0] ) dbe_->removeElement( meEERecHitOccupancyProPhiThr_[0]->getName() );
    meEERecHitOccupancyProPhiThr_[0] = 0;

    if ( meEERecHitOccupancyThr_[1] ) dbe_->removeElement( meEERecHitOccupancyThr_[1]->getName() );
    meEERecHitOccupancyThr_[1] = 0;
    if ( meEERecHitOccupancyProRThr_[1] ) dbe_->removeElement( meEERecHitOccupancyProRThr_[1]->getName() );
    meEERecHitOccupancyProRThr_[1] = 0;
    if ( meEERecHitOccupancyProPhiThr_[1] ) dbe_->removeElement( meEERecHitOccupancyProPhiThr_[1]->getName() );
    meEERecHitOccupancyProPhiThr_[1] = 0;

    if ( meEETrigPrimDigiOccupancy_[0] ) dbe_->removeElement( meEETrigPrimDigiOccupancy_[0]->getName() );
    meEETrigPrimDigiOccupancy_[0] = 0;
    if ( meEETrigPrimDigiOccupancyProR_[0] ) dbe_->removeElement( meEETrigPrimDigiOccupancyProR_[0]->getName() );
    meEETrigPrimDigiOccupancyProR_[0] = 0;
    if ( meEETrigPrimDigiOccupancyProPhi_[0] ) dbe_->removeElement( meEETrigPrimDigiOccupancyProPhi_[0]->getName() );
    meEETrigPrimDigiOccupancyProPhi_[0] = 0;

    if ( meEETrigPrimDigiOccupancy_[1] ) dbe_->removeElement( meEETrigPrimDigiOccupancy_[1]->getName() );
    meEETrigPrimDigiOccupancy_[1] = 0;
    if ( meEETrigPrimDigiOccupancyProR_[1] ) dbe_->removeElement( meEETrigPrimDigiOccupancyProR_[1]->getName() );
    meEETrigPrimDigiOccupancyProR_[1] = 0;
    if ( meEETrigPrimDigiOccupancyProPhi_[1] ) dbe_->removeElement( meEETrigPrimDigiOccupancyProPhi_[1]->getName() );
    meEETrigPrimDigiOccupancyProPhi_[1] = 0;

    if ( meEETrigPrimDigiOccupancyThr_[0] ) dbe_->removeElement( meEETrigPrimDigiOccupancyThr_[0]->getName() );
    meEETrigPrimDigiOccupancyThr_[0] = 0;
    if ( meEETrigPrimDigiOccupancyProRThr_[0] ) dbe_->removeElement( meEETrigPrimDigiOccupancyProRThr_[0]->getName() );
    meEETrigPrimDigiOccupancyProRThr_[0] = 0;
    if ( meEETrigPrimDigiOccupancyProPhiThr_[0] ) dbe_->removeElement( meEETrigPrimDigiOccupancyProPhiThr_[0]->getName() );
    meEETrigPrimDigiOccupancyProPhiThr_[0] = 0;

    if ( meEETrigPrimDigiOccupancyThr_[1] ) dbe_->removeElement( meEETrigPrimDigiOccupancyThr_[1]->getName() );
    meEETrigPrimDigiOccupancyThr_[1] = 0;
    if ( meEETrigPrimDigiOccupancyProRThr_[1] ) dbe_->removeElement( meEETrigPrimDigiOccupancyProRThr_[1]->getName() );
    meEETrigPrimDigiOccupancyProRThr_[1] = 0;
    if ( meEETrigPrimDigiOccupancyProPhiThr_[1] ) dbe_->removeElement( meEETrigPrimDigiOccupancyProPhiThr_[1]->getName() );
    meEETrigPrimDigiOccupancyProPhiThr_[1] = 0;

  }

  init_ = false;

}

void EEOccupancyTask::endJob(void) {

  LogInfo("EEOccupancyTask") << "analyzed " << ievt_ << " events";

  if ( init_ ) this->cleanup();

}

void EEOccupancyTask::analyze(const Event& e, const EventSetup& c){

  if ( ! init_ ) this->setup();

  ievt_++;

  Handle<EEDigiCollection> digis;

  if ( e.getByLabel(EEDigiCollection_, digis) ) {

    int need = digis->size();
    LogDebug("EEOccupancyTask") << "event " << ievt_ << " digi collection size " << need;

    for ( EEDigiCollection::const_iterator digiItr = digis->begin(); digiItr != digis->end(); ++digiItr ) {

      EEDataFrame dataframe = (*digiItr);
      EEDetId id = dataframe.id();

      int ix = id.ix();
      int iy = id.iy();

      int ism = Numbers::iSM( id );

      if ( ism >= 1 && ism <= 9 ) ix = 101 - ix;

      float xix = ix - 0.5;
      float xiy = iy - 0.5;

      LogDebug("EEOccupancyTask") << " det id = " << id;
      LogDebug("EEOccupancyTask") << " sm, ix, iy " << ism << " " << ix << " " << iy;

      if ( xix <= 0. || xix >= 100. || xiy <= 0. || xiy >= 100. ) {
        LogWarning("EEOccupancyTask") << " det id = " << id;
        LogWarning("EEOccupancyTask") << " sm, ix, iw " << ism << " " << ix << " " << iy;
        LogWarning("EEOccupancyTask") << " xix, xiy " << xix << " " << xiy;
      }

      if ( meOccupancy_[ism-1] ) meOccupancy_[ism-1]->Fill( xix, xiy );

      int eex = id.ix();
      int eey = id.iy();

      if ( ism >= 1 && ism <= 9 ) eex = 101 - eex;

      float xeex = eex - 0.5;
      float xeey = eey - 0.5;

      if ( ism >=1 && ism <= 9 ) {
        if ( meEEDigiOccupancy_[0] ) meEEDigiOccupancy_[0]->Fill( xeex, xeey );
        if ( meEEDigiOccupancyProR_[0] ) meEEDigiOccupancyProR_[0]->Fill( sqrt(pow(xeex-50.,2)+pow(xeey-50.,2)) );
        if ( meEEDigiOccupancyProPhi_[0] ) meEEDigiOccupancyProPhi_[0]->Fill( atan2(xeey-50.,xeex-50.) );
      } else {
        if ( meEEDigiOccupancy_[1] ) meEEDigiOccupancy_[1]->Fill( xeex, xeey );
        if ( meEEDigiOccupancyProR_[1] ) meEEDigiOccupancyProR_[1]->Fill( sqrt(pow(xeex-50.,2)+pow(xeey-50.,2)) );
        if ( meEEDigiOccupancyProPhi_[1] ) meEEDigiOccupancyProPhi_[1]->Fill( atan2(xeey-50.,xeex-50.) );
      }

    }

  } else {

    LogWarning("EEOccupancyTask") << EEDigiCollection_ << " not available";

  }

  Handle<EcalPnDiodeDigiCollection> PNs;

  if ( e.getByLabel(EcalPnDiodeDigiCollection_, PNs) ) {

    // filling mem occupancy only for the 5 channels belonging
    // to a fully reconstructed PN's

    for ( EcalPnDiodeDigiCollection::const_iterator pnItr = PNs->begin(); pnItr != PNs->end(); ++pnItr ) {

      EcalPnDiodeDigi pn = (*pnItr);
      EcalPnDiodeDetId id = pn.id();

      if ( Numbers::subDet( id ) != EcalEndcap ) continue;

      int   ism   = Numbers::iSM( id );

      float PnId  = (*pnItr).id().iPnId();

      PnId        = PnId - 0.5;
      float st    = 0.0;

      for (int chInStrip = 1; chInStrip <= 5; chInStrip++){
        if ( meOccupancyMem_[ism-1] ) {
           st = chInStrip - 0.5;
           meOccupancyMem_[ism-1]->Fill(PnId, st);
        }
      }

    }

  } else {

    LogWarning("EEOccupancyTask") << EcalPnDiodeDigiCollection_ << " not available";

  }

  Handle<EcalRecHitCollection> rechits;

  if ( e.getByLabel(EcalRecHitCollection_, rechits) ) {

    int nebrh = rechits->size();
    LogDebug("EEOccupancyTask") << "event " << ievt_ << " rec hits collection size " << nebrh;

    for ( EcalRecHitCollection::const_iterator rechitItr = rechits->begin(); rechitItr != rechits->end(); ++rechitItr ) {

      EEDetId id = rechitItr->id();

      int eex = id.ix();
      int eey = id.iy();

      int ism = Numbers::iSM( id );

      if ( ism >= 1 && ism <= 9 ) eex = 101 - eex;

      float xeex = eex - 0.5;
      float xeey = eey - 0.5;

      if ( ism >= 1 && ism <= 9 ) {
        if ( meEERecHitOccupancy_[0] ) meEERecHitOccupancy_[0]->Fill( xeex, xeey );
        if ( meEERecHitOccupancyProR_[0] ) meEERecHitOccupancyProR_[0]->Fill( sqrt(pow(xeex-50.,2)+pow(xeey-50.,2)) );
        if ( meEERecHitOccupancyProPhi_[0] ) meEERecHitOccupancyProPhi_[0]->Fill( atan2(xeey-50.,xeex-50.) );
      } else {
        if ( meEERecHitOccupancy_[1] ) meEERecHitOccupancy_[1]->Fill( xeex, xeey );
        if ( meEERecHitOccupancyProR_[1] ) meEERecHitOccupancyProR_[1]->Fill( sqrt(pow(xeex-50.,2)+pow(xeey-50.,2)) );
        if ( meEERecHitOccupancyProPhi_[1] ) meEERecHitOccupancyProPhi_[1]->Fill( atan2(xeey-50.,xeex-50.) );
      }

      if ( rechitItr->energy() > recHitEnergyMin_ ) { 

        if ( ism >= 1 && ism <= 9 ) {
          if ( meEERecHitOccupancyThr_[0] ) meEERecHitOccupancyThr_[0]->Fill( xeex, xeey );
          if ( meEERecHitOccupancyProRThr_[0] ) meEERecHitOccupancyProRThr_[0]->Fill( sqrt(pow(xeex-50.,2)+pow(xeey-50.,2)) );
          if ( meEERecHitOccupancyProPhiThr_[0] ) meEERecHitOccupancyProPhiThr_[0]->Fill( atan2(xeey-50.,xeex-50.) );
        } else {
          if ( meEERecHitOccupancyThr_[1] ) meEERecHitOccupancyThr_[1]->Fill( xeex, xeey );
          if ( meEERecHitOccupancyProRThr_[1] ) meEERecHitOccupancyProRThr_[1]->Fill( sqrt(pow(xeex-50.,2)+pow(xeey-50.,2)) );
          if ( meEERecHitOccupancyProPhiThr_[1] ) meEERecHitOccupancyProPhiThr_[1]->Fill( atan2(xeey-50.,xeex-50.) );
        }

      }

    }

  } else {

    LogWarning("EEOccupancyTask") << EcalRecHitCollection_ << " not available";

  }

  Handle<EcalTrigPrimDigiCollection> trigPrimDigis;

  if ( e.getByLabel(EcalTrigPrimDigiCollection_, trigPrimDigis) ) {

    int nebtpg = trigPrimDigis->size();
    LogDebug("EEOccupancyTask") << "event " << ievt_ << " trigger primitives digis collection size " << nebtpg;

    for ( EcalTrigPrimDigiCollection::const_iterator tpdigiItr = trigPrimDigis->begin(); tpdigiItr != trigPrimDigis->end(); ++tpdigiItr ) {

      EcalTriggerPrimitiveDigi data = (*tpdigiItr);
      EcalTrigTowerDetId idt = data.id();

      if ( Numbers::subDet( idt ) != EcalEndcap ) continue;
      
      int ismt = Numbers::iSM( idt );
      
      vector<DetId> crystals = Numbers::crystals( idt );
      
      for ( unsigned int i=0; i<crystals.size(); i++ ) {
        
        EEDetId id = crystals[i];
        
        int eex = id.ix();
        int eey = id.iy();
        
        if ( ismt >= 1 && ismt <= 9 ) eex = 101 - eex;

        float xeex = eex - 0.5;
        float xeey = eey - 0.5;

        if ( ismt >= 1 && ismt <= 9 ) {
          if ( meEETrigPrimDigiOccupancy_[0] ) meEETrigPrimDigiOccupancy_[0]->Fill( xeex, xeey );
          if ( meEETrigPrimDigiOccupancyProR_[0] ) meEETrigPrimDigiOccupancyProR_[0]->Fill( sqrt(pow(xeex-50.,2)+pow(xeey-50.,2)) );
          if ( meEETrigPrimDigiOccupancyProPhi_[0] ) meEETrigPrimDigiOccupancyProPhi_[0]->Fill( atan2(xeey-50.,xeex-50.) );
        } else {
          if ( meEETrigPrimDigiOccupancy_[1] ) meEETrigPrimDigiOccupancy_[1]->Fill( xeex, xeey );
          if ( meEETrigPrimDigiOccupancyProR_[1] ) meEETrigPrimDigiOccupancyProR_[1]->Fill( sqrt(pow(xeex-50.,2)+pow(xeey-50.,2)) );
          if ( meEETrigPrimDigiOccupancyProPhi_[1] ) meEETrigPrimDigiOccupancyProPhi_[1]->Fill( atan2(xeey-50.,xeex-50.) );
        }

        if ( data.compressedEt() > trigPrimEtMin_ ) {
        
          if ( ismt >= 1 && ismt <= 9 ) {
            if ( meEETrigPrimDigiOccupancyThr_[0] ) meEETrigPrimDigiOccupancyThr_[0]->Fill( xeex, xeey );
            if ( meEETrigPrimDigiOccupancyProRThr_[0] ) meEETrigPrimDigiOccupancyProRThr_[0]->Fill( sqrt(pow(xeex-50.,2)+pow(xeey-50.,2)) );
            if ( meEETrigPrimDigiOccupancyProPhiThr_[0] ) meEETrigPrimDigiOccupancyProPhiThr_[0]->Fill( atan2(xeey-50.,xeex-50.) );
          } else {
            if ( meEETrigPrimDigiOccupancyThr_[1] ) meEETrigPrimDigiOccupancyThr_[1]->Fill( xeex, xeey );
            if ( meEETrigPrimDigiOccupancyProRThr_[1] ) meEETrigPrimDigiOccupancyProRThr_[1]->Fill( sqrt(pow(xeex-50.,2)+pow(xeey-50.,2)) );
            if ( meEETrigPrimDigiOccupancyProPhiThr_[1] ) meEETrigPrimDigiOccupancyProPhiThr_[1]->Fill( atan2(xeey-50.,xeex-50.) );
          }

        }

      }

    }

  } else {

    LogWarning("EEOccupancyTask") << EcalTrigPrimDigiCollection_ << " not available";

  }

}

