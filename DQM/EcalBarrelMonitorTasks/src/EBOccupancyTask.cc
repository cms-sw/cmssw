/*
 * \file EBOccupancyTask.cc
 *
 * $Date: 2008/03/16 10:30:25 $
 * $Revision: 1.57 $
 * \author G. Della Ricca
 * \author G. Franzoni
 *
*/

#include <iostream>
#include <fstream>
#include <vector>

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DQMServices/Core/interface/MonitorElement.h"

#include "DQMServices/Core/interface/DQMStore.h"

#include "DataFormats/EcalRawData/interface/EcalRawDataCollections.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDigi/interface/EBDataFrame.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "DataFormats/EcalRecHit/interface/EcalUncalibratedRecHit.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"

#include <DQM/EcalCommon/interface/Numbers.h>

#include <DQM/EcalBarrelMonitorTasks/interface/EBOccupancyTask.h>

using namespace cms;
using namespace edm;
using namespace std;

EBOccupancyTask::EBOccupancyTask(const ParameterSet& ps){

  init_ = false;

  // get hold of back-end interface
  dbe_ = Service<DQMStore>().operator->();

  enableCleanup_ = ps.getUntrackedParameter<bool>("enableCleanup", false);

  EcalRawDataCollection_ = ps.getParameter<edm::InputTag>("EcalRawDataCollection");
  EBDigiCollection_ = ps.getParameter<edm::InputTag>("EBDigiCollection");
  EcalPnDiodeDigiCollection_ = ps.getParameter<edm::InputTag>("EcalPnDiodeDigiCollection");
  EcalRecHitCollection_ = ps.getParameter<edm::InputTag>("EcalRecHitCollection");
  EcalTrigPrimDigiCollection_ = ps.getParameter<edm::InputTag>("EcalTrigPrimDigiCollection");

  for (int i = 0; i < 36; i++) {
    meOccupancy_[i]    = 0;
    meOccupancyMem_[i] = 0;
  }

  meEBDigiOccupancy_ = 0;
  meEBDigiOccupancyProjEta_ = 0;
  meEBDigiOccupancyProjPhi_ = 0;

  meEBRecHitOccupancy_ = 0;
  meEBRecHitOccupancyProjEta_ = 0;
  meEBRecHitOccupancyProjPhi_ = 0;

  meEBRecHitOccupancyThr_ = 0;
  meEBRecHitOccupancyProjEtaThr_ = 0;
  meEBRecHitOccupancyProjPhiThr_ = 0;

  meEBTrigPrimDigiOccupancy_ = 0;
  meEBTrigPrimDigiOccupancyProjEta_ = 0;
  meEBTrigPrimDigiOccupancyProjPhi_ = 0;

  meEBTrigPrimDigiOccupancyThr_ = 0;
  meEBTrigPrimDigiOccupancyProjEtaThr_ = 0;
  meEBTrigPrimDigiOccupancyProjPhiThr_ = 0;

  meEBTestPulseDigiOccupancy_ = 0;
  meEBLaserDigiOccupancy_ = 0;
  meEBPedestalDigiOccupancy_ = 0;

  recHitEnergyMin_ = 1.; // GeV
  trigPrimEtMin_ = 5.; // GeV
}

EBOccupancyTask::~EBOccupancyTask(){

}

void EBOccupancyTask::beginJob(const EventSetup& c){

  ievt_ = 0;

  if ( dbe_ ) {
    dbe_->setCurrentFolder("EcalBarrel/EBOccupancyTask");
    dbe_->rmdir("EcalBarrel/EBOccupancyTask");
  }

  Numbers::initGeometry(c);

}

void EBOccupancyTask::setup(void){

  init_ = true;

  char histo[200];

  if ( dbe_ ) {
    dbe_->setCurrentFolder("EcalBarrel/EBOccupancyTask");

    for (int i = 0; i < 36; i++) {
      sprintf(histo, "EBOT digi occupancy %s", Numbers::sEB(i+1).c_str());
      meOccupancy_[i] = dbe_->book2D(histo, histo, 85, 0., 85., 20, 0., 20.);
      meOccupancy_[i]->setAxisTitle("ieta", 1);
      meOccupancy_[i]->setAxisTitle("iphi", 2);
      dbe_->tag(meOccupancy_[i], i+1);
    }
    for (int i = 0; i < 36; i++) {
      sprintf(histo, "EBOT MEM digi occupancy %s", Numbers::sEB(i+1).c_str());
      meOccupancyMem_[i] = dbe_->book2D(histo, histo, 10, 0., 10., 5, 0., 5.);
      meOccupancyMem_[i]->setAxisTitle("pseudo-strip", 1);
      meOccupancyMem_[i]->setAxisTitle("channel", 2);
      dbe_->tag(meOccupancyMem_[i], i+1);
    }

    sprintf(histo, "EBOT digi occupancy");
    meEBDigiOccupancy_ = dbe_->book2D(histo, histo, 72, 0., 360., 34, -85., 85.);
    meEBDigiOccupancy_->setAxisTitle("jphi", 1);
    meEBDigiOccupancy_->setAxisTitle("jeta", 2);
    sprintf(histo, "EBOT digi occupancy projection eta");
    meEBDigiOccupancyProjEta_ = dbe_->book1D(histo, histo, 34, -85., 85.);
    meEBDigiOccupancyProjEta_->setAxisTitle("jeta", 1);
    meEBDigiOccupancyProjEta_->setAxisTitle("number of digis", 2);
    sprintf(histo, "EBOT digi occupancy projection phi");
    meEBDigiOccupancyProjPhi_ = dbe_->book1D(histo, histo, 72, 0., 360.);
    meEBDigiOccupancyProjPhi_->setAxisTitle("jphi", 1);
    meEBDigiOccupancyProjPhi_->setAxisTitle("number of digis", 2);

    sprintf(histo, "EBOT rec hit occupancy");
    meEBRecHitOccupancy_ = dbe_->book2D(histo, histo, 72, 0., 360., 34, -85., 85.);
    meEBRecHitOccupancy_->setAxisTitle("jphi", 1);
    meEBRecHitOccupancy_->setAxisTitle("jeta", 2);
    sprintf(histo, "EBOT rec hit occupancy projection eta");
    meEBRecHitOccupancyProjEta_ = dbe_->book1D(histo, histo, 34, -85., 85.);
    meEBRecHitOccupancyProjEta_->setAxisTitle("jeta", 1);
    meEBRecHitOccupancyProjEta_->setAxisTitle("number of hits", 2);
    sprintf(histo, "EBOT rec hit occupancy projection phi");
    meEBRecHitOccupancyProjPhi_ = dbe_->book1D(histo, histo, 72, 0., 360.);
    meEBRecHitOccupancyProjPhi_->setAxisTitle("jphi", 1);
    meEBRecHitOccupancyProjPhi_->setAxisTitle("number of hits", 2);

    sprintf(histo, "EBOT rec hit thr occupancy");
    meEBRecHitOccupancyThr_ = dbe_->book2D(histo, histo, 72, 0., 360., 34, -85., 85.);
    meEBRecHitOccupancyThr_->setAxisTitle("jphi", 1);
    meEBRecHitOccupancyThr_->setAxisTitle("jeta", 2);
    sprintf(histo, "EBOT rec hit thr occupancy projection eta");
    meEBRecHitOccupancyProjEtaThr_ = dbe_->book1D(histo, histo, 34, -85., 85.);
    meEBRecHitOccupancyProjEtaThr_->setAxisTitle("jeta", 1);
    meEBRecHitOccupancyProjEtaThr_->setAxisTitle("number of hits", 2);
    sprintf(histo, "EBOT rec hit thr occupancy projection phi");
    meEBRecHitOccupancyProjPhiThr_ = dbe_->book1D(histo, histo, 72, 0., 360.);
    meEBRecHitOccupancyProjPhiThr_->setAxisTitle("jphi", 1);
    meEBRecHitOccupancyProjPhiThr_->setAxisTitle("number of hits", 2);

    sprintf(histo, "EBOT TP digi occupancy");
    meEBTrigPrimDigiOccupancy_ = dbe_->book2D(histo, histo, 72, 0., 72., 34, -17., 17.);
    meEBTrigPrimDigiOccupancy_->setAxisTitle("jphi'", 1);
    meEBTrigPrimDigiOccupancy_->setAxisTitle("jeta'", 2);
    sprintf(histo, "EBOT TP digi occupancy projection eta");
    meEBTrigPrimDigiOccupancyProjEta_ = dbe_->book1D(histo, histo, 34, -17., 17.);
    meEBTrigPrimDigiOccupancyProjEta_->setAxisTitle("jeta'", 1);
    meEBTrigPrimDigiOccupancyProjEta_->setAxisTitle("number of TP digis", 2);
    sprintf(histo, "EBOT TP digi occupancy projection phi");
    meEBTrigPrimDigiOccupancyProjPhi_ = dbe_->book1D(histo, histo, 72, 0., 72.);
    meEBTrigPrimDigiOccupancyProjPhi_->setAxisTitle("jphi'", 1);
    meEBTrigPrimDigiOccupancyProjPhi_->setAxisTitle("number of TP digis", 2);

    sprintf(histo, "EBOT TP digi thr occupancy");
    meEBTrigPrimDigiOccupancyThr_ = dbe_->book2D(histo, histo, 72, 0., 72., 34, -17., 17.);
    meEBTrigPrimDigiOccupancyThr_->setAxisTitle("jphi'", 1);
    meEBTrigPrimDigiOccupancyThr_->setAxisTitle("jeta'", 2);
    sprintf(histo, "EBOT TP digi thr occupancy projection eta");
    meEBTrigPrimDigiOccupancyProjEtaThr_ = dbe_->book1D(histo, histo, 34, -17., 17.);
    meEBTrigPrimDigiOccupancyProjEtaThr_->setAxisTitle("jeta'", 1);
    meEBTrigPrimDigiOccupancyProjEtaThr_->setAxisTitle("number of TP digis", 2);
    sprintf(histo, "EBOT TP digi thr occupancy projection phi");
    meEBTrigPrimDigiOccupancyProjPhiThr_ = dbe_->book1D(histo, histo, 72, 0., 72.);
    meEBTrigPrimDigiOccupancyProjPhiThr_->setAxisTitle("jphi'", 1);
    meEBTrigPrimDigiOccupancyProjPhiThr_->setAxisTitle("number of TP digis", 2);

    sprintf(histo, "EBOT test pulse digi occupancy");
    meEBTestPulseDigiOccupancy_ = dbe_->book2D(histo, histo, 72, 0., 360., 34, -85., 85.);
    meEBTestPulseDigiOccupancy_->setAxisTitle("jphi'", 1);
    meEBTestPulseDigiOccupancy_->setAxisTitle("jeta'", 2);

    sprintf(histo, "EBOT laser digi occupancy");
    meEBLaserDigiOccupancy_ = dbe_->book2D(histo, histo, 72, 0., 360., 34, -85., 85.);
    meEBLaserDigiOccupancy_->setAxisTitle("jphi'", 1);
    meEBLaserDigiOccupancy_->setAxisTitle("jeta'", 2);

    sprintf(histo, "EBOT pedestal digi occupancy");
    meEBPedestalDigiOccupancy_ = dbe_->book2D(histo, histo, 72, 0., 360., 34, -85., 85.);
    meEBPedestalDigiOccupancy_->setAxisTitle("jphi'", 1);
    meEBPedestalDigiOccupancy_->setAxisTitle("jeta'", 2);

  }

}

void EBOccupancyTask::cleanup(void){

  if ( ! enableCleanup_ ) return;

  if ( dbe_ ) {
    dbe_->setCurrentFolder("EcalBarrel/EBOccupancyTask");

    for (int i = 0; i < 36; i++) {
      if ( meOccupancy_[i] ) dbe_->removeElement( meOccupancy_[i]->getName() );
      meOccupancy_[i] = 0;
      if ( meOccupancyMem_[i] ) dbe_->removeElement( meOccupancyMem_[i]->getName() );
      meOccupancyMem_[i] = 0;
    }

    if ( meEBDigiOccupancy_ ) dbe_->removeElement( meEBDigiOccupancy_->getName() );
    meEBDigiOccupancy_ = 0;
    if ( meEBDigiOccupancyProjEta_ ) dbe_->removeElement( meEBDigiOccupancyProjEta_->getName() );
    meEBDigiOccupancyProjEta_ = 0;
    if ( meEBDigiOccupancyProjPhi_ ) dbe_->removeElement( meEBDigiOccupancyProjPhi_->getName() );
    meEBDigiOccupancyProjPhi_ = 0;

    if ( meEBRecHitOccupancy_ ) dbe_->removeElement( meEBRecHitOccupancy_->getName() );
    meEBRecHitOccupancy_ = 0;
    if ( meEBRecHitOccupancyProjEta_ ) dbe_->removeElement( meEBRecHitOccupancyProjEta_->getName() );
    meEBRecHitOccupancyProjEta_ = 0;
    if ( meEBRecHitOccupancyProjPhi_ ) dbe_->removeElement( meEBRecHitOccupancyProjPhi_->getName() );
    meEBRecHitOccupancyProjPhi_ = 0;

    if ( meEBRecHitOccupancyThr_ ) dbe_->removeElement( meEBRecHitOccupancyThr_->getName() );
    meEBRecHitOccupancyThr_ = 0;
    if ( meEBRecHitOccupancyProjEtaThr_ ) dbe_->removeElement( meEBRecHitOccupancyProjEtaThr_->getName() );
    meEBRecHitOccupancyProjEtaThr_ = 0;
    if ( meEBRecHitOccupancyProjPhiThr_ ) dbe_->removeElement( meEBRecHitOccupancyProjPhiThr_->getName() );
    meEBRecHitOccupancyProjPhiThr_ = 0;

    if ( meEBTrigPrimDigiOccupancy_ ) dbe_->removeElement( meEBTrigPrimDigiOccupancy_->getName() );
    meEBTrigPrimDigiOccupancy_ = 0;
    if ( meEBTrigPrimDigiOccupancyProjEta_ ) dbe_->removeElement( meEBTrigPrimDigiOccupancyProjEta_->getName() );
    meEBTrigPrimDigiOccupancyProjEta_ = 0;
    if ( meEBTrigPrimDigiOccupancyProjPhi_ ) dbe_->removeElement( meEBTrigPrimDigiOccupancyProjPhi_->getName() );
    meEBTrigPrimDigiOccupancyProjPhi_ = 0;

    if ( meEBTrigPrimDigiOccupancyThr_ ) dbe_->removeElement( meEBTrigPrimDigiOccupancyThr_->getName() );
    meEBTrigPrimDigiOccupancyThr_ = 0;
    if ( meEBTrigPrimDigiOccupancyProjEtaThr_ ) dbe_->removeElement( meEBTrigPrimDigiOccupancyProjEtaThr_->getName() );
    meEBTrigPrimDigiOccupancyProjEtaThr_ = 0;
    if ( meEBTrigPrimDigiOccupancyProjPhiThr_ ) dbe_->removeElement( meEBTrigPrimDigiOccupancyProjPhiThr_->getName() );
    meEBTrigPrimDigiOccupancyProjPhiThr_ = 0;

    if ( meEBTestPulseDigiOccupancy_ ) dbe_->removeElement( meEBTestPulseDigiOccupancy_->getName() );
    meEBTestPulseDigiOccupancy_ = 0;

    if ( meEBLaserDigiOccupancy_ ) dbe_->removeElement( meEBLaserDigiOccupancy_->getName() );
    meEBLaserDigiOccupancy_ = 0;

    if ( meEBPedestalDigiOccupancy_ ) dbe_->removeElement( meEBPedestalDigiOccupancy_->getName() );
    meEBPedestalDigiOccupancy_ = 0;

  }

  init_ = false;

}

void EBOccupancyTask::endJob(void) {

  LogInfo("EBOccupancyTask") << "analyzed " << ievt_ << " events";

  if ( init_ ) this->cleanup();

}

void EBOccupancyTask::analyze(const Event& e, const EventSetup& c){

  if ( ! init_ ) this->setup();

  ievt_++;

  Handle<EcalRawDataCollection> dcchs;

  if ( e.getByLabel(EcalRawDataCollection_, dcchs) ) {
    LogWarning("EBOccupancyTask") << EcalRawDataCollection_ << " not available";
  }

  Handle<EBDigiCollection> digis;

  if ( e.getByLabel(EBDigiCollection_, digis) ) {

    int nebd = digis->size();
    LogDebug("EBOccupancyTask") << "event " << ievt_ << " digi collection size " << nebd;

    for ( EBDigiCollection::const_iterator digiItr = digis->begin(); digiItr != digis->end(); ++digiItr ) {

      EBDataFrame dataframe = (*digiItr);
      EBDetId id = dataframe.id();

      int ic = id.ic();
      int ie = (ic-1)/20 + 1;
      int ip = (ic-1)%20 + 1;

      int ism = Numbers::iSM( id );

      float xie = ie - 0.5;
      float xip = ip - 0.5;

      LogDebug("EBOccupancyTask") << " det id = " << id;
      LogDebug("EBOccupancyTask") << " sm, ieta, iphi " << ism << " " << ie << " " << ip;

      if ( xie <= 0. || xie >= 85. || xip <= 0. || xip >= 20. ) {
        LogWarning("EBOccupancyTask") << " det id = " << id;
        LogWarning("EBOccupancyTask") << " sm, ieta, iphi " << ism << " " << ie << " " << ip;
        LogWarning("EBOccupancyTask") << " xie, xip " << xie << " " << xip;
      }

      if ( meOccupancy_[ism-1] ) meOccupancy_[ism-1]->Fill(xie, xip);

      int ebeta = id.ieta();
      int ebphi = id.iphi();

      float xebeta = ebeta - 0.5*id.zside();
      float xebphi = ebphi - 0.5;

      if ( meEBDigiOccupancy_ ) meEBDigiOccupancy_->Fill( xebphi, xebeta );
      if ( meEBDigiOccupancyProjEta_ ) meEBDigiOccupancyProjEta_->Fill( xebeta );
      if ( meEBDigiOccupancyProjPhi_ ) meEBDigiOccupancyProjPhi_->Fill( xebphi );

      if ( dcchs.isValid() ) {

        for ( EcalRawDataCollection::const_iterator dcchItr = dcchs->begin(); dcchItr != dcchs->end(); ++dcchItr ) {

          EcalDCCHeaderBlock dcch = (*dcchItr);

          if ( Numbers::subDet( dcch ) != EcalBarrel ) continue;

          if ( dcch.getRunType() == EcalDCCHeaderBlock::TESTPULSE_MGPA ||
               dcch.getRunType() == EcalDCCHeaderBlock::TESTPULSE_GAP ) {

            if ( meEBTestPulseDigiOccupancy_ ) meEBTestPulseDigiOccupancy_->Fill( xebphi, xebeta );

          }

          if ( dcch.getRunType() == EcalDCCHeaderBlock::LASER_STD ||
               dcch.getRunType() == EcalDCCHeaderBlock::LASER_GAP ) {

            if ( meEBLaserDigiOccupancy_ ) meEBLaserDigiOccupancy_->Fill( xebphi, xebeta );

          }

          if ( dcch.getRunType() == EcalDCCHeaderBlock::PEDESTAL_STD ||
               dcch.getRunType() == EcalDCCHeaderBlock::PEDESTAL_GAP ) {

            if ( meEBPedestalDigiOccupancy_ ) meEBPedestalDigiOccupancy_->Fill( xebphi, xebeta );

          }

        }

      }

    }

  } else {

    LogWarning("EBOccupancyTask") << EBDigiCollection_ << " not available";

  }

  Handle<EcalPnDiodeDigiCollection> PNs;

  if ( e.getByLabel(EcalPnDiodeDigiCollection_, PNs) ) {

    // filling mem occupancy only for the 5 channels belonging
    // to a fully reconstructed PN's

    for ( EcalPnDiodeDigiCollection::const_iterator pnItr = PNs->begin(); pnItr != PNs->end(); ++pnItr ) {

      EcalPnDiodeDigi pn = (*pnItr);
      EcalPnDiodeDetId id = pn.id();

      if ( Numbers::subDet( id ) != EcalBarrel ) continue;

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

    LogWarning("EBOccupancyTask") << EcalPnDiodeDigiCollection_ << " not available";

  }

  Handle<EcalRecHitCollection> rechits;

  if ( e.getByLabel(EcalRecHitCollection_, rechits) ) {

    int nebrh = rechits->size();
    LogDebug("EBOccupancyTask") << "event " << ievt_ << " rec hits collection size " << nebrh;

    for ( EcalRecHitCollection::const_iterator rechitItr = rechits->begin(); rechitItr != rechits->end(); ++rechitItr ) {

      EBDetId id = rechitItr->id();

      int ebeta = id.ieta();
      int ebphi = id.iphi();

      float xebeta = ebeta - 0.5*id.zside();
      float xebphi = ebphi - 0.5;

      if ( meEBRecHitOccupancy_ ) meEBRecHitOccupancy_->Fill( xebphi, xebeta );
      if ( meEBRecHitOccupancyProjEta_ ) meEBRecHitOccupancyProjEta_->Fill( xebeta );
      if ( meEBRecHitOccupancyProjPhi_ ) meEBRecHitOccupancyProjPhi_->Fill( xebphi );

      if ( rechitItr->energy() > recHitEnergyMin_ ) {

        if ( meEBRecHitOccupancyThr_ ) meEBRecHitOccupancyThr_->Fill( xebphi, xebeta );
        if ( meEBRecHitOccupancyProjEtaThr_ ) meEBRecHitOccupancyProjEtaThr_->Fill( xebeta );
        if ( meEBRecHitOccupancyProjPhiThr_ ) meEBRecHitOccupancyProjPhiThr_->Fill( xebphi );

      }

    }

  } else {

    LogWarning("EBOccupancyTask") << EcalRecHitCollection_ << " not available";

  }

  Handle<EcalTrigPrimDigiCollection> trigPrimDigis;

  if ( e.getByLabel(EcalTrigPrimDigiCollection_, trigPrimDigis) ) {

    int nebtpg = trigPrimDigis->size();
    LogDebug("EBOccupancyTask") << "event " << ievt_ << " trigger primitives digis collection size " << nebtpg;

    for ( EcalTrigPrimDigiCollection::const_iterator tpdigiItr = trigPrimDigis->begin(); tpdigiItr != trigPrimDigis->end(); ++tpdigiItr ) {

      EcalTriggerPrimitiveDigi data = (*tpdigiItr);
      EcalTrigTowerDetId idt = data.id();

      int ebeta = idt.ieta();
      int ebphi = idt.iphi();

      float xebeta = ebeta-0.5*idt.zside();
      float xebphi = ebphi-0.5;

      if ( meEBTrigPrimDigiOccupancy_ ) meEBTrigPrimDigiOccupancy_->Fill( xebphi, xebeta );
      if ( meEBTrigPrimDigiOccupancyProjEta_ ) meEBTrigPrimDigiOccupancyProjEta_->Fill( xebeta );
      if ( meEBTrigPrimDigiOccupancyProjPhi_ ) meEBTrigPrimDigiOccupancyProjPhi_->Fill( xebphi );

      if ( data.compressedEt() > trigPrimEtMin_ ) {

        if ( meEBTrigPrimDigiOccupancyThr_ ) meEBTrigPrimDigiOccupancyThr_->Fill( xebphi, xebeta );
        if ( meEBTrigPrimDigiOccupancyProjEtaThr_ ) meEBTrigPrimDigiOccupancyProjEtaThr_->Fill( xebeta );
        if ( meEBTrigPrimDigiOccupancyProjPhiThr_ ) meEBTrigPrimDigiOccupancyProjPhiThr_->Fill( xebphi );

      }

    }

  } else {

    LogWarning("EBOccupancyTask") << EcalTrigPrimDigiCollection_ << " not available";

  }


}

