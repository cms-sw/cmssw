/*
 * \file EBLaserTask.cc
 *
 * $Date: 2006/09/22 06:01:40 $
 * $Revision: 1.57 $
 * \author G. Della Ricca
 *
*/

#include <DQM/EcalBarrelMonitorTasks/interface/EBLaserTask.h>

EBLaserTask::EBLaserTask(const ParameterSet& ps){

  init_ = false;

  for (int i = 0; i < 36 ; i++) {
    meShapeMapL1_[i] = 0;
    meAmplMapL1_[i] = 0;
    meTimeMapL1_[i] = 0;
    meAmplPNMapL1_[i] = 0;
    mePnAmplMapG01L1_[i] = 0;
    mePnPedMapG01L1_[i] = 0;
    mePnAmplMapG16L1_[i] = 0;
    mePnPedMapG16L1_[i] = 0;
    meShapeMapL2_[i] = 0;
    meAmplMapL2_[i] = 0;
    meTimeMapL2_[i] = 0;
    meAmplPNMapL2_[i] = 0;
    mePnAmplMapG01L2_[i] = 0;
    mePnPedMapG01L2_[i] = 0;
    mePnAmplMapG16L2_[i] = 0;
    mePnPedMapG16L2_[i] = 0;
    meShapeMapL3_[i] = 0;
    meAmplMapL3_[i] = 0;
    meTimeMapL3_[i] = 0;
    meAmplPNMapL3_[i] = 0;
    mePnAmplMapG01L3_[i] = 0;
    mePnPedMapG01L3_[i] = 0;
    mePnAmplMapG16L3_[i] = 0;
    mePnPedMapG16L3_[i] = 0;
    meShapeMapL4_[i] = 0;
    meAmplMapL4_[i] = 0;
    meTimeMapL4_[i] = 0;
    meAmplPNMapL4_[i] = 0;
    mePnAmplMapG01L4_[i] = 0; 
    mePnPedMapG01L4_[i] = 0;
    mePnAmplMapG16L4_[i] = 0;
    mePnPedMapG16L4_[i] = 0;
  }

}

EBLaserTask::~EBLaserTask(){

}

void EBLaserTask::beginJob(const EventSetup& c){

  ievt_ = 0;

  DaqMonitorBEInterface* dbe = 0;

  // get hold of back-end interface
  dbe = Service<DaqMonitorBEInterface>().operator->();

  if ( dbe ) {
    dbe->setCurrentFolder("EcalBarrel/EBLaserTask");
    dbe->rmdir("EcalBarrel/EBLaserTask");
  }

}

void EBLaserTask::setup(void){

  init_ = true;

  Char_t histo[200];

  DaqMonitorBEInterface* dbe = 0;

  // get hold of back-end interface
  dbe = Service<DaqMonitorBEInterface>().operator->();

  if ( dbe ) {
    dbe->setCurrentFolder("EcalBarrel/EBLaserTask");

    dbe->setCurrentFolder("EcalBarrel/EBLaserTask/Laser1");
    for (int i = 0; i < 36 ; i++) {
      sprintf(histo, "EBLT shape SM%02d L1", i+1);
      meShapeMapL1_[i] = dbe->bookProfile2D(histo, histo, 1700, 0., 1700., 10, 0., 10., 4096, 0., 4096., "s");
      dbe->tag(meShapeMapL1_[i], i+1);
      sprintf(histo, "EBLT amplitude SM%02d L1", i+1);
      meAmplMapL1_[i] = dbe->bookProfile2D(histo, histo, 85, 0., 85., 20, 0., 20., 4096, 0., 4096., "s");
      dbe->tag(meAmplMapL1_[i], i+1);
      sprintf(histo, "EBLT timing SM%02d L1", i+1);
      meTimeMapL1_[i] = dbe->bookProfile2D(histo, histo, 85, 0., 85., 20, 0., 20., 4096, 0., 4096., "s");
      dbe->tag(meTimeMapL1_[i], i+1);
      sprintf(histo, "EBLT amplitude over PN SM%02d L1", i+1);
      meAmplPNMapL1_[i] = dbe->bookProfile2D(histo, histo, 85, 0., 85., 20, 0., 20., 4096, 0., 4096., "s");
      dbe->tag(meAmplPNMapL1_[i], i+1);
    }

    dbe->setCurrentFolder("EcalBarrel/EBLaserTask/Laser2");
    for (int i = 0; i < 36 ; i++) {
      sprintf(histo, "EBLT shape SM%02d L2", i+1);
      meShapeMapL2_[i] = dbe->bookProfile2D(histo, histo, 1700, 0., 1700., 10, 0., 10., 4096, 0., 4096., "s");
      dbe->tag(meShapeMapL2_[i], i+1);
      sprintf(histo, "EBLT amplitude SM%02d L2", i+1);
      meAmplMapL2_[i] = dbe->bookProfile2D(histo, histo, 85, 0., 85., 20, 0., 20., 4096, 0., 4096., "s");
      dbe->tag(meAmplMapL2_[i], i+1);
      sprintf(histo, "EBLT timing SM%02d L2", i+1);
      meTimeMapL2_[i] = dbe->bookProfile2D(histo, histo, 85, 0., 85., 20, 0., 20., 4096, 0., 4096., "s");
      dbe->tag(meTimeMapL2_[i], i+1);
      sprintf(histo, "EBLT amplitude over PN SM%02d L2", i+1);
      meAmplPNMapL2_[i] = dbe->bookProfile2D(histo, histo, 85, 0., 85., 20, 0., 20., 4096, 0., 4096., "s");
      dbe->tag(meAmplPNMapL2_[i], i+1);
    }

    dbe->setCurrentFolder("EcalBarrel/EBLaserTask/Laser3");
    for (int i = 0; i < 36 ; i++) {
      sprintf(histo, "EBLT shape SM%02d L3", i+1);
      meShapeMapL3_[i] = dbe->bookProfile2D(histo, histo, 1700, 0., 1700., 10, 0., 10., 4096, 0., 4096., "s");
      dbe->tag(meShapeMapL3_[i], i+1);
      sprintf(histo, "EBLT amplitude SM%02d L3", i+1);
      meAmplMapL3_[i] = dbe->bookProfile2D(histo, histo, 85, 0., 85., 20, 0., 20., 4096, 0., 4096., "s");
      dbe->tag(meAmplMapL3_[i], i+1);
      sprintf(histo, "EBLT timing SM%02d L3", i+1);
      meTimeMapL3_[i] = dbe->bookProfile2D(histo, histo, 85, 0., 85., 20, 0., 20., 4096, 0., 4096., "s");
      dbe->tag(meTimeMapL3_[i], i+1);
      sprintf(histo, "EBLT amplitude over PN SM%02d L3", i+1);
      meAmplPNMapL3_[i] = dbe->bookProfile2D(histo, histo, 85, 0., 85., 20, 0., 20., 4096, 0., 4096., "s");
      dbe->tag(meAmplPNMapL3_[i], i+1);
    }

    dbe->setCurrentFolder("EcalBarrel/EBLaserTask/Laser4");
    for (int i = 0; i < 36 ; i++) {
      sprintf(histo, "EBLT shape SM%02d L4", i+1);
      meShapeMapL4_[i] = dbe->bookProfile2D(histo, histo, 1700, 0., 1700., 10, 0., 10., 4096, 0., 4096., "s");
      dbe->tag(meShapeMapL4_[i], i+1);
      sprintf(histo, "EBLT amplitude SM%02d L4", i+1);
      meAmplMapL4_[i] = dbe->bookProfile2D(histo, histo, 85, 0., 85., 20, 0., 20., 4096, 0., 4096., "s");
      dbe->tag(meAmplMapL4_[i], i+1);
      sprintf(histo, "EBLT timing SM%02d L4", i+1);
      meTimeMapL4_[i] = dbe->bookProfile2D(histo, histo, 85, 0., 85., 20, 0., 20., 4096, 0., 4096., "s");
      dbe->tag(meTimeMapL4_[i], i+1);
      sprintf(histo, "EBLT amplitude over PN SM%02d L4", i+1);
      meAmplPNMapL4_[i] = dbe->bookProfile2D(histo, histo, 85, 0., 85., 20, 0., 20., 4096, 0., 4096., "s");
      dbe->tag(meAmplPNMapL4_[i], i+1);
    }

    dbe->setCurrentFolder("EcalBarrel/EBPnDiodeTask");

    dbe->setCurrentFolder("EcalBarrel/EBPnDiodeTask/Laser1");

    dbe->setCurrentFolder("EcalBarrel/EBPnDiodeTask/Laser1/Gain01");
    for (int i = 0; i < 36 ; i++) {
      sprintf(histo, "EBPDT PNs amplitude SM%02d G01 L1", i+1);
      mePnAmplMapG01L1_[i] = dbe->bookProfile2D(histo, histo, 1, 0., 1., 10, 0., 10., 4096, 0., 4096., "s");
      dbe->tag(mePnAmplMapG01L1_[i], i+1);
      sprintf(histo, "EBPDT PNs pedestal SM%02d G01 L1", i+1);
      mePnPedMapG01L1_[i] = dbe->bookProfile2D(histo, histo, 1, 0., 1., 10, 0., 10., 4096, 0., 4096., "s");
      dbe->tag(mePnPedMapG01L1_[i], i+1);
    }

    dbe->setCurrentFolder("EcalBarrel/EBPnDiodeTask/Laser1/Gain16");
    for (int i = 0; i < 36 ; i++) {
      sprintf(histo, "EBPDT PNs amplitude SM%02d G16 L1", i+1);
      mePnAmplMapG16L1_[i] = dbe->bookProfile2D(histo, histo, 1, 0., 1., 10, 0., 10., 4096, 0., 4096., "s");
      dbe->tag(mePnAmplMapG16L1_[i], i+1);
      sprintf(histo, "EBPDT PNs pedestal SM%02d G16 L1", i+1);
      mePnPedMapG16L1_[i] = dbe->bookProfile2D(histo, histo, 1, 0., 1., 10, 0., 10., 4096, 0., 4096., "s");
      dbe->tag(mePnPedMapG16L1_[i], i+1);
    }

    dbe->setCurrentFolder("EcalBarrel/EBPnDiodeTask/Laser2");

    dbe->setCurrentFolder("EcalBarrel/EBPnDiodeTask/Laser2/Gain01");
    for (int i = 0; i < 36 ; i++) {
      sprintf(histo, "EBPDT PNs amplitude SM%02d G01 L2", i+1);
      mePnAmplMapG01L2_[i] = dbe->bookProfile2D(histo, histo, 1, 0., 1., 10, 0., 10., 4096, 0., 4096., "s");
      dbe->tag(mePnAmplMapG01L2_[i], i+1);
      sprintf(histo, "EBPDT PNs pedestal SM%02d G01 L2", i+1);
      mePnPedMapG01L2_[i] = dbe->bookProfile2D(histo, histo, 1, 0., 1., 10, 0., 10., 4096, 0., 4096., "s");
      dbe->tag(mePnPedMapG01L2_[i], i+1);
    }

    dbe->setCurrentFolder("EcalBarrel/EBPnDiodeTask/Laser2/Gain16");
    for (int i = 0; i < 36 ; i++) {
      sprintf(histo, "EBPDT PNs amplitude SM%02d G16 L2", i+1);
      mePnAmplMapG16L2_[i] = dbe->bookProfile2D(histo, histo, 1, 0., 1., 10, 0., 10., 4096, 0., 4096., "s");
      dbe->tag(mePnAmplMapG16L2_[i], i+1);
      sprintf(histo, "EBPDT PNs pedestal SM%02d G16 L2", i+1);
      mePnPedMapG16L2_[i] = dbe->bookProfile2D(histo, histo, 1, 0., 1., 10, 0., 10., 4096, 0., 4096., "s");
      dbe->tag(mePnPedMapG16L2_[i], i+1);
    }

    dbe->setCurrentFolder("EcalBarrel/EBPnDiodeTask/Laser3");

    dbe->setCurrentFolder("EcalBarrel/EBPnDiodeTask/Laser3/Gain01");
    for (int i = 0; i < 36 ; i++) {
      sprintf(histo, "EBPDT PNs amplitude SM%02d G01 L3", i+1);
      mePnAmplMapG01L3_[i] = dbe->bookProfile2D(histo, histo, 1, 0., 1., 10, 0., 10., 4096, 0., 4096., "s");
      dbe->tag(mePnAmplMapG01L3_[i], i+1);
      sprintf(histo, "EBPDT PNs pedestal SM%02d G01 L3", i+1);
      mePnPedMapG01L3_[i] = dbe->bookProfile2D(histo, histo, 1, 0., 1., 10, 0., 10., 4096, 0., 4096., "s");
      dbe->tag(mePnPedMapG01L3_[i], i+1);
    }

    dbe->setCurrentFolder("EcalBarrel/EBPnDiodeTask/Laser3/Gain16");
    for (int i = 0; i < 36 ; i++) {
      sprintf(histo, "EBPDT PNs amplitude SM%02d G16 L3", i+1);
      mePnAmplMapG16L3_[i] = dbe->bookProfile2D(histo, histo, 1, 0., 1., 10, 0., 10., 4096, 0., 4096., "s");
      dbe->tag(mePnAmplMapG16L3_[i], i+1);
      sprintf(histo, "EBPDT PNs pedestal SM%02d G16 L3", i+1);
      mePnPedMapG16L3_[i] = dbe->bookProfile2D(histo, histo, 1, 0., 1., 10, 0., 10., 4096, 0., 4096., "s");
      dbe->tag(mePnPedMapG16L3_[i], i+1);
    }

    dbe->setCurrentFolder("EcalBarrel/EBPnDiodeTask/Laser4");

    dbe->setCurrentFolder("EcalBarrel/EBPnDiodeTask/Laser4/Gain01");
    for (int i = 0; i < 36 ; i++) {
      sprintf(histo, "EBPDT PNs amplitude SM%02d G01 L4", i+1);
      mePnAmplMapG01L4_[i] = dbe->bookProfile2D(histo, histo, 1, 0., 1., 10, 0., 10., 4096, 0., 4096., "s");
      dbe->tag(mePnAmplMapG01L4_[i], i+1);
      sprintf(histo, "EBPDT PNs pedestal SM%02d G01 L4", i+1);
      mePnPedMapG01L4_[i] = dbe->bookProfile2D(histo, histo, 1, 0., 1., 10, 0., 10., 4096, 0., 4096., "s");
      dbe->tag(mePnPedMapG01L4_[i], i+1);
    }

    dbe->setCurrentFolder("EcalBarrel/EBPnDiodeTask/Laser4/Gain16");
    for (int i = 0; i < 36 ; i++) {
      sprintf(histo, "EBPDT PNs amplitude SM%02d G16 L4", i+1);
      mePnAmplMapG16L4_[i] = dbe->bookProfile2D(histo, histo, 1, 0., 1., 10, 0., 10., 4096, 0., 4096., "s");
      dbe->tag(mePnAmplMapG16L4_[i], i+1);
      sprintf(histo, "EBPDT PNs pedestal SM%02d G16 L4", i+1);
      mePnPedMapG16L4_[i] = dbe->bookProfile2D(histo, histo, 1, 0., 1., 10, 0., 10., 4096, 0., 4096., "s");
      dbe->tag(mePnPedMapG16L4_[i], i+1);
    }

  }

}

void EBLaserTask::cleanup(void){

  DaqMonitorBEInterface* dbe = 0;

  // get hold of back-end interface
  dbe = Service<DaqMonitorBEInterface>().operator->();

  if ( dbe ) {
    dbe->setCurrentFolder("EcalBarrel/EBLaserTask");

    dbe->setCurrentFolder("EcalBarrel/EBLaserTask/Laser1");
    for (int i = 0; i < 36 ; i++) {
      if ( meShapeMapL1_[i] )  dbe->removeElement( meShapeMapL1_[i]->getName() );
      meShapeMapL1_[i] = 0;
      if ( meAmplMapL1_[i] ) dbe->removeElement( meAmplMapL1_[i]->getName() );
      meAmplMapL1_[i] = 0;
      if ( meTimeMapL1_[i] ) dbe->removeElement( meTimeMapL1_[i]->getName() );
      meTimeMapL1_[i] = 0;
      if ( meAmplPNMapL1_[i] ) dbe->removeElement( meAmplPNMapL1_[i]->getName() );
      meAmplPNMapL1_[i] = 0;
    }

    dbe->setCurrentFolder("EcalBarrel/EBLaserTask/Laser2");
    for (int i = 0; i < 36 ; i++) {
      if ( meShapeMapL2_[i] )  dbe->removeElement( meShapeMapL2_[i]->getName() );
      meShapeMapL2_[i] = 0;
      if ( meAmplMapL2_[i] ) dbe->removeElement( meAmplMapL2_[i]->getName() );
      meAmplMapL2_[i] = 0;
      if ( meTimeMapL2_[i] ) dbe->removeElement( meTimeMapL2_[i]->getName() );
      meTimeMapL2_[i] = 0;
      if ( meAmplPNMapL2_[i] ) dbe->removeElement( meAmplPNMapL2_[i]->getName() );
      meAmplPNMapL2_[i] = 0;
    }

    dbe->setCurrentFolder("EcalBarrel/EBLaserTask/Laser3");
    for (int i = 0; i < 36 ; i++) {
      if ( meShapeMapL3_[i] )  dbe->removeElement( meShapeMapL3_[i]->getName() );
      meShapeMapL3_[i] = 0;
      if ( meAmplMapL3_[i] ) dbe->removeElement( meAmplMapL3_[i]->getName() );
      meAmplMapL3_[i] = 0;
      if ( meTimeMapL3_[i] ) dbe->removeElement( meTimeMapL3_[i]->getName() );
      meTimeMapL3_[i] = 0;
      if ( meAmplPNMapL3_[i] ) dbe->removeElement( meAmplPNMapL3_[i]->getName() );
      meAmplPNMapL3_[i] = 0;
    }

    dbe->setCurrentFolder("EcalBarrel/EBLaserTask/Laser4");
    for (int i = 0; i < 36 ; i++) {
      if ( meShapeMapL4_[i] )  dbe->removeElement( meShapeMapL4_[i]->getName() );
      meShapeMapL4_[i] = 0;
      if ( meAmplMapL4_[i] ) dbe->removeElement( meAmplMapL4_[i]->getName() );
      meAmplMapL4_[i] = 0;
      if ( meTimeMapL4_[i] ) dbe->removeElement( meTimeMapL4_[i]->getName() );
      meTimeMapL4_[i] = 0;
      if ( meAmplPNMapL4_[i] ) dbe->removeElement( meAmplPNMapL4_[i]->getName() );
      meAmplPNMapL4_[i] = 0;
    }

    dbe->setCurrentFolder("EcalBarrel/EBPnDiodeTask");

    dbe->setCurrentFolder("EcalBarrel/EBPnDiodeTask/Laser1");

    dbe->setCurrentFolder("EcalBarrel/EBPnDiodeTask/Laser1/Gain01");
    for (int i = 0; i < 36 ; i++) {
      if ( mePnAmplMapG01L1_[i] ) dbe->removeElement( mePnAmplMapG01L1_[i]->getName() );
      mePnAmplMapG01L1_[i] = 0;
      if ( mePnPedMapG01L1_[i] ) dbe->removeElement( mePnPedMapG01L1_[i]->getName() );
      mePnPedMapG01L1_[i] = 0;
    }

    dbe->setCurrentFolder("EcalBarrel/EBPnDiodeTask/Laser1/Gain16");
    for (int i = 0; i < 36 ; i++) {
      if ( mePnAmplMapG16L1_[i] ) dbe->removeElement( mePnAmplMapG16L1_[i]->getName() );
      mePnAmplMapG16L1_[i] = 0;
      if ( mePnPedMapG16L1_[i] ) dbe->removeElement( mePnPedMapG16L1_[i]->getName() );
      mePnPedMapG16L1_[i] = 0;
    }

    dbe->setCurrentFolder("EcalBarrel/EBPnDiodeTask/Laser2");

    dbe->setCurrentFolder("EcalBarrel/EBPnDiodeTask/Laser2/Gain01");
    for (int i = 0; i < 36 ; i++) {
      if ( mePnAmplMapG01L2_[i] ) dbe->removeElement( mePnAmplMapG01L2_[i]->getName() );
      mePnAmplMapG01L2_[i] = 0;
      if ( mePnPedMapG01L2_[i] ) dbe->removeElement( mePnPedMapG01L2_[i]->getName() );
      mePnPedMapG01L2_[i] = 0;
    }

    dbe->setCurrentFolder("EcalBarrel/EBPnDiodeTask/Laser2/Gain16");
    for (int i = 0; i < 36 ; i++) {
      if ( mePnAmplMapG16L2_[i] ) dbe->removeElement( mePnAmplMapG16L2_[i]->getName() );
      mePnAmplMapG16L2_[i] = 0;
      if ( mePnPedMapG16L2_[i] ) dbe->removeElement( mePnPedMapG16L2_[i]->getName() );
      mePnPedMapG16L2_[i] = 0;
    }

    dbe->setCurrentFolder("EcalBarrel/EBPnDiodeTask/Laser3");

    dbe->setCurrentFolder("EcalBarrel/EBPnDiodeTask/Laser3/Gain01");
    for (int i = 0; i < 36 ; i++) {
      if ( mePnAmplMapG01L3_[i] ) dbe->removeElement( mePnAmplMapG01L3_[i]->getName() );
      mePnAmplMapG01L3_[i] = 0;
      if ( mePnPedMapG01L3_[i] ) dbe->removeElement( mePnPedMapG01L3_[i]->getName() );
      mePnPedMapG01L3_[i] = 0;
    }

    dbe->setCurrentFolder("EcalBarrel/EBPnDiodeTask/Laser3/Gain16");
    for (int i = 0; i < 36 ; i++) {
      if ( mePnAmplMapG16L3_[i] ) dbe->removeElement( mePnAmplMapG16L3_[i]->getName() );
      mePnAmplMapG16L3_[i] = 0;
      if ( mePnPedMapG16L3_[i] ) dbe->removeElement( mePnPedMapG16L3_[i]->getName() );
      mePnPedMapG16L3_[i] = 0;
    }

    dbe->setCurrentFolder("EcalBarrel/EBPnDiodeTask/Laser4");

    dbe->setCurrentFolder("EcalBarrel/EBPnDiodeTask/Laser4/Gain01");
    for (int i = 0; i < 36 ; i++) {
      if ( mePnAmplMapG01L4_[i] ) dbe->removeElement( mePnAmplMapG01L4_[i]->getName() );
      mePnAmplMapG01L4_[i] = 0;
      if ( mePnPedMapG01L4_[i] ) dbe->removeElement( mePnPedMapG01L4_[i]->getName() );
      mePnPedMapG01L4_[i] = 0;
    }

    dbe->setCurrentFolder("EcalBarrel/EBPnDiodeTask/Laser4/Gain16");
    for (int i = 0; i < 36 ; i++) {
      if ( mePnAmplMapG16L4_[i] ) dbe->removeElement( mePnAmplMapG16L4_[i]->getName() );
      mePnAmplMapG16L4_[i] = 0;
      if ( mePnPedMapG16L4_[i] ) dbe->removeElement( mePnPedMapG16L4_[i]->getName() );
      mePnPedMapG16L4_[i] = 0;
    }

  }

  init_ = false;

}

void EBLaserTask::endJob(void){

  LogInfo("EBLaserTask") << "analyzed " << ievt_ << " events";

  if ( init_ ) this->cleanup();

}

void EBLaserTask::analyze(const Event& e, const EventSetup& c){

  bool enable = false;
  map<int, EcalDCCHeaderBlock> dccMap;

  Handle<EcalRawDataCollection> dcchs;
  e.getByLabel("ecalEBunpacker", dcchs);

  for ( EcalRawDataCollection::const_iterator dcchItr = dcchs->begin(); dcchItr != dcchs->end(); ++dcchItr ) {

    EcalDCCHeaderBlock dcch = (*dcchItr);

    dccMap[dcch.id()] = dcch;

    if ( dccMap[dcch.id()].getRunType() == EcalDCCHeaderBlock::LASER_STD ) enable = true;

    // uncomment the following line to mix fake 'laser' events w/ cosmic & beam events
    // if ( dccMap[dcch.id()].getRunType() == EcalDCCHeaderBlock::COSMIC ) enable = true;

  }

  if ( ! enable ) return;

  if ( ! init_ ) this->setup();

  ievt_++;

  Handle<EBDigiCollection> digis;
  e.getByLabel("ecalEBunpacker", digis);

  int nebd = digis->size();
  LogDebug("EBLaserTask") << "event " << ievt_ << " digi collection size " << nebd;

  for ( EBDigiCollection::const_iterator digiItr = digis->begin(); digiItr != digis->end(); ++digiItr ) {

    EBDataFrame dataframe = (*digiItr);
    EBDetId id = dataframe.id();

    int ic = id.ic();
    int ie = (ic-1)/20 + 1;
    int ip = (ic-1)%20 + 1;

    int ism = id.ism();

    if ( dccMap[ism-1].getRunType() != EcalDCCHeaderBlock::LASER_STD ) continue;

    LogDebug("EBLaserTask") << " det id = " << id;
    LogDebug("EBLaserTask") << " sm, eta, phi " << ism << " " << ie << " " << ip;

    for (int i = 0; i < 10; i++) {

      EcalMGPASample sample = dataframe.sample(i);
      int adc = sample.adc();
      float gain = 1.;

      MonitorElement* meShapeMap = 0;

      if ( sample.gainId() == 1 ) gain = 1./12.;
      if ( sample.gainId() == 2 ) gain = 1./ 6.;
      if ( sample.gainId() == 3 ) gain = 1./ 1.;

      if ( dccMap[ism-1].getEventSettings().wavelength == 0 ) meShapeMap = meShapeMapL1_[ism-1];
      if ( dccMap[ism-1].getEventSettings().wavelength == 1 ) meShapeMap = meShapeMapL2_[ism-1];
      if ( dccMap[ism-1].getEventSettings().wavelength == 2 ) meShapeMap = meShapeMapL3_[ism-1];
      if ( dccMap[ism-1].getEventSettings().wavelength == 3 ) meShapeMap = meShapeMapL4_[ism-1];

      float xval = float(adc) * gain;

      if ( meShapeMap ) meShapeMap->Fill(ic - 0.5, i + 0.5, xval);

    }

  }

  Handle<EcalPnDiodeDigiCollection> pns;
  e.getByLabel("ecalEBunpacker", pns);

  float adc[36];

  for ( int i = 0; i < 36; i++ ) adc[i] = 0.;

  int nep = pns->size();
  LogDebug("EBLaserTask") << "event " << ievt_ << " pns collection size " << nep;

  for ( EcalPnDiodeDigiCollection::const_iterator pnItr = pns->begin(); pnItr != pns->end(); ++pnItr ) {

    EcalPnDiodeDigi pn = (*pnItr);
    EcalPnDiodeDetId id = pn.id();

//    int ism = id.ism();
    int ism = id.iDCCId();

    int num = id.iPnId();

    if ( dccMap[ism-1].getRunType() != EcalDCCHeaderBlock::LASER_STD ) continue;

    LogDebug("EBLaserTask") << " det id = " << id;
    LogDebug("EBLaserTask") << " sm, num " << ism << " " << num;

    float xvalped = 0.;

    for (int i = 0; i < 4; i++) {

      EcalFEMSample sample = pn.sample(i);
      int adc = sample.adc();

      MonitorElement* mePNPed = 0;

      if ( sample.gainId() == 0 ) {
        if ( dccMap[ism-1].getEventSettings().wavelength == 0 ) mePNPed = mePnPedMapG01L1_[ism-1];
        if ( dccMap[ism-1].getEventSettings().wavelength == 1 ) mePNPed = mePnPedMapG01L2_[ism-1];
        if ( dccMap[ism-1].getEventSettings().wavelength == 2 ) mePNPed = mePnPedMapG01L3_[ism-1];
        if ( dccMap[ism-1].getEventSettings().wavelength == 3 ) mePNPed = mePnPedMapG01L4_[ism-1];
      }
      if ( sample.gainId() == 1 ) {
        if ( dccMap[ism-1].getEventSettings().wavelength == 0 ) mePNPed = mePnPedMapG16L1_[ism-1];
        if ( dccMap[ism-1].getEventSettings().wavelength == 1 ) mePNPed = mePnPedMapG16L2_[ism-1];
        if ( dccMap[ism-1].getEventSettings().wavelength == 2 ) mePNPed = mePnPedMapG16L3_[ism-1];
        if ( dccMap[ism-1].getEventSettings().wavelength == 3 ) mePNPed = mePnPedMapG16L4_[ism-1];
      }

      float xval = float(adc);

      if ( mePNPed ) mePNPed->Fill(0.5, num - 0.5, xval);

      xvalped = xvalped + xval;

    }

    xvalped = xvalped / 4;

    float xvalmax = 0.;

    MonitorElement* mePN = 0;

    for (int i = 0; i < 50; i++) {

      EcalFEMSample sample = pn.sample(i);
      int adc = sample.adc();

      float xval = float(adc);

      if ( xval >= xvalmax ) xvalmax = xval;

    }

    xvalmax = xvalmax - xvalped;

    if ( pn.sample(0).gainId() == 0 ) {
      if ( dccMap[ism-1].getEventSettings().wavelength == 0 ) mePN = mePnAmplMapG01L1_[ism-1];
      if ( dccMap[ism-1].getEventSettings().wavelength == 1 ) mePN = mePnAmplMapG01L2_[ism-1];
      if ( dccMap[ism-1].getEventSettings().wavelength == 2 ) mePN = mePnAmplMapG01L3_[ism-1];
      if ( dccMap[ism-1].getEventSettings().wavelength == 3 ) mePN = mePnAmplMapG01L4_[ism-1];
    }
    if ( pn.sample(0).gainId() == 1 ) {
      if ( dccMap[ism-1].getEventSettings().wavelength == 0 ) mePN = mePnAmplMapG16L1_[ism-1];
      if ( dccMap[ism-1].getEventSettings().wavelength == 1 ) mePN = mePnAmplMapG16L2_[ism-1];
      if ( dccMap[ism-1].getEventSettings().wavelength == 2 ) mePN = mePnAmplMapG16L3_[ism-1];
      if ( dccMap[ism-1].getEventSettings().wavelength == 3 ) mePN = mePnAmplMapG16L4_[ism-1];
    }

    if ( mePN ) mePN->Fill(0.5, num - 0.5, xvalmax);

    if ( num == 1 ) adc[ism-1] = xvalmax;

  }

  Handle<EcalUncalibratedRecHitCollection> hits;
  e.getByLabel("ecalUncalibHitMaker", "EcalUncalibRecHitsEB", hits);

  int neh = hits->size();
  LogDebug("EBLaserTask") << "event " << ievt_ << " hits collection size " << neh;

  for ( EcalUncalibratedRecHitCollection::const_iterator hitItr = hits->begin(); hitItr != hits->end(); ++hitItr ) {

    EcalUncalibratedRecHit hit = (*hitItr);
    EBDetId id = hit.id();

    int ic = id.ic();
    int ie = (ic-1)/20 + 1;
    int ip = (ic-1)%20 + 1;

    int ism = id.ism();

    float xie = ie - 0.5;
    float xip = ip - 0.5;

    if ( dccMap[ism-1].getRunType() != EcalDCCHeaderBlock::LASER_STD ) continue;

    LogDebug("EBLaserTask") << " det id = " << id;
    LogDebug("EBLaserTask") << " sm, eta, phi " << ism << " " << ie << " " << ip;

    MonitorElement* meAmplMap = 0;
    MonitorElement* meTimeMap = 0;
    MonitorElement* meAmplPNMap = 0;

    if ( dccMap[ism-1].getEventSettings().wavelength == 0 ) {
      meAmplMap = meAmplMapL1_[ism-1];
      meTimeMap = meTimeMapL1_[ism-1];
      meAmplPNMap = meAmplPNMapL1_[ism-1];
    }
    if ( dccMap[ism-1].getEventSettings().wavelength == 1 ) {
      meAmplMap = meAmplMapL2_[ism-1];
      meTimeMap = meTimeMapL2_[ism-1];
      meAmplPNMap = meAmplPNMapL2_[ism-1];
    }
    if ( dccMap[ism-1].getEventSettings().wavelength == 2 ) {
      meAmplMap = meAmplMapL3_[ism-1];
      meTimeMap = meTimeMapL3_[ism-1];
      meAmplPNMap = meAmplPNMapL3_[ism-1];
    }
    if ( dccMap[ism-1].getEventSettings().wavelength == 3 ) {
      meAmplMap = meAmplMapL4_[ism-1];
      meTimeMap = meTimeMapL4_[ism-1];
      meAmplPNMap = meAmplPNMapL4_[ism-1];
    }

    float xval = hit.amplitude();
    float yval = hit.jitter();
    float zval = hit.pedestal();

    LogDebug("EBLaserTask") << " hit amplitude " << xval;
    LogDebug("EBLaserTask") << " hit jitter " << yval;
    LogDebug("EBLaserTask") << " hit pedestal " << zval;

    if ( meAmplMap ) meAmplMap->Fill(xie, xip, xval);

    if ( meTimeMap ) meTimeMap->Fill(xie, xip, yval);

    float wval = 0.;
    if ( adc[ism-1] != 0. ) wval = xval / adc[ism-1];

    LogDebug("EBLaserTask") << " hit amplitude over PN " << wval;

    if ( meAmplPNMap ) meAmplPNMap->Fill(xie, xip, wval);

  }

}

