/*
 * \file EBLaserTask.cc
 *
 * $Date: 2007/02/17 14:33:38 $
 * $Revision: 1.69 $
 * \author G. Della Ricca
 *
*/

#include <DQM/EcalBarrelMonitorTasks/interface/EBLaserTask.h>

using namespace cms;
using namespace edm;
using namespace std;

EBLaserTask::EBLaserTask(const ParameterSet& ps){

  init_ = false;

  for (int i = 0; i < 36 ; i++) {
    meShapeMapL1A_[i] = 0;
    meAmplMapL1A_[i] = 0;
    meTimeMapL1A_[i] = 0;
    meAmplPNMapL1A_[i] = 0;
    meShapeMapL1B_[i] = 0;
    meAmplMapL1B_[i] = 0;
    meTimeMapL1B_[i] = 0;
    meAmplPNMapL1B_[i] = 0;
    mePnAmplMapG01L1_[i] = 0;
    mePnPedMapG01L1_[i] = 0;
    mePnAmplMapG16L1_[i] = 0;
    mePnPedMapG16L1_[i] = 0;

    meShapeMapL2A_[i] = 0;
    meAmplMapL2A_[i] = 0;
    meTimeMapL2A_[i] = 0;
    meAmplPNMapL2A_[i] = 0;
    meShapeMapL2B_[i] = 0;
    meAmplMapL2B_[i] = 0;
    meTimeMapL2B_[i] = 0;
    meAmplPNMapL2B_[i] = 0;
    mePnAmplMapG01L2_[i] = 0;
    mePnPedMapG01L2_[i] = 0;
    mePnAmplMapG16L2_[i] = 0;
    mePnPedMapG16L2_[i] = 0;

    meShapeMapL3A_[i] = 0;
    meAmplMapL3A_[i] = 0;
    meTimeMapL3A_[i] = 0;
    meAmplPNMapL3A_[i] = 0;
    meShapeMapL3B_[i] = 0;
    meAmplMapL3B_[i] = 0;
    meTimeMapL3B_[i] = 0;
    meAmplPNMapL3B_[i] = 0;
    mePnAmplMapG01L3_[i] = 0;
    mePnPedMapG01L3_[i] = 0;
    mePnAmplMapG16L3_[i] = 0;
    mePnPedMapG16L3_[i] = 0;

    meShapeMapL4A_[i] = 0;
    meAmplMapL4A_[i] = 0;
    meTimeMapL4A_[i] = 0;
    meAmplPNMapL4A_[i] = 0;
    meShapeMapL4B_[i] = 0;
    meAmplMapL4B_[i] = 0;
    meTimeMapL4B_[i] = 0;
    meAmplPNMapL4B_[i] = 0;
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
      sprintf(histo, "EBLT shape SM%02d L1A", i+1);
      meShapeMapL1A_[i] = dbe->bookProfile2D(histo, histo, 1700, 0., 1700., 10, 0., 10., 4096, 0., 4096., "s");
      dbe->tag(meShapeMapL1A_[i], i+1);
      sprintf(histo, "EBLT amplitude SM%02d L1A", i+1);
      meAmplMapL1A_[i] = dbe->bookProfile2D(histo, histo, 85, 0., 85., 20, 0., 20., 4096, 0., 4096.*12., "s");
      dbe->tag(meAmplMapL1A_[i], i+1);
      sprintf(histo, "EBLT timing SM%02d L1A", i+1);
      meTimeMapL1A_[i] = dbe->bookProfile2D(histo, histo, 85, 0., 85., 20, 0., 20., 250, 0., 10., "s");
      dbe->tag(meTimeMapL1A_[i], i+1);
      sprintf(histo, "EBLT amplitude over PN SM%02d L1A", i+1);
      meAmplPNMapL1A_[i] = dbe->bookProfile2D(histo, histo, 85, 0., 85., 20, 0., 20., 4096, 0., 4096.*12., "s");
      dbe->tag(meAmplPNMapL1A_[i], i+1);

      sprintf(histo, "EBLT shape SM%02d L1B", i+1);
      meShapeMapL1B_[i] = dbe->bookProfile2D(histo, histo, 1700, 0., 1700., 10, 0., 10., 4096, 0., 4096., "s");
      dbe->tag(meShapeMapL1B_[i], i+1);
      sprintf(histo, "EBLT amplitude SM%02d L1B", i+1);
      meAmplMapL1B_[i] = dbe->bookProfile2D(histo, histo, 85, 0., 85., 20, 0., 20., 4096, 0., 4096.*12., "s");
      dbe->tag(meAmplMapL1B_[i], i+1);
      sprintf(histo, "EBLT timing SM%02d L1B", i+1);
      meTimeMapL1B_[i] = dbe->bookProfile2D(histo, histo, 85, 0., 85., 20, 0., 20., 250, 0., 10., "s");
      dbe->tag(meTimeMapL1B_[i], i+1);
      sprintf(histo, "EBLT amplitude over PN SM%02d L1B", i+1);
      meAmplPNMapL1B_[i] = dbe->bookProfile2D(histo, histo, 85, 0., 85., 20, 0., 20., 4096, 0., 4096.*12., "s");
      dbe->tag(meAmplPNMapL1B_[i], i+1);
    }

    dbe->setCurrentFolder("EcalBarrel/EBLaserTask/Laser2");
    for (int i = 0; i < 36 ; i++) {
      sprintf(histo, "EBLT shape SM%02d L2A", i+1);
      meShapeMapL2A_[i] = dbe->bookProfile2D(histo, histo, 1700, 0., 1700., 10, 0., 10., 4096, 0., 4096., "s");
      dbe->tag(meShapeMapL2A_[i], i+1);
      sprintf(histo, "EBLT amplitude SM%02d L2A", i+1);
      meAmplMapL2A_[i] = dbe->bookProfile2D(histo, histo, 85, 0., 85., 20, 0., 20., 4096, 0., 4096.*12., "s");
      dbe->tag(meAmplMapL2A_[i], i+1);
      sprintf(histo, "EBLT timing SM%02d L2A", i+1);
      meTimeMapL2A_[i] = dbe->bookProfile2D(histo, histo, 85, 0., 85., 20, 0., 20., 250, 0., 10., "s");
      dbe->tag(meTimeMapL2A_[i], i+1);
      sprintf(histo, "EBLT amplitude over PN SM%02d L2A", i+1);
      meAmplPNMapL2A_[i] = dbe->bookProfile2D(histo, histo, 85, 0., 85., 20, 0., 20., 4096, 0., 4096.*12., "s");
      dbe->tag(meAmplPNMapL2A_[i], i+1);

      sprintf(histo, "EBLT shape SM%02d L2B", i+1);
      meShapeMapL2B_[i] = dbe->bookProfile2D(histo, histo, 1700, 0., 1700., 10, 0., 10., 4096, 0., 4096., "s");
      dbe->tag(meShapeMapL2B_[i], i+1);
      sprintf(histo, "EBLT amplitude SM%02d L2B", i+1);
      meAmplMapL2B_[i] = dbe->bookProfile2D(histo, histo, 85, 0., 85., 20, 0., 20., 4096, 0., 4096.*12., "s");
      dbe->tag(meAmplMapL2B_[i], i+1);
      sprintf(histo, "EBLT timing SM%02d L2B", i+1);
      meTimeMapL2B_[i] = dbe->bookProfile2D(histo, histo, 85, 0., 85., 20, 0., 20., 250, 0., 10., "s");
      dbe->tag(meTimeMapL2B_[i], i+1);
      sprintf(histo, "EBLT amplitude over PN SM%02d L2B", i+1);
      meAmplPNMapL2B_[i] = dbe->bookProfile2D(histo, histo, 85, 0., 85., 20, 0., 20., 4096, 0., 4096.*12., "s");
      dbe->tag(meAmplPNMapL2B_[i], i+1);
    }

    dbe->setCurrentFolder("EcalBarrel/EBLaserTask/Laser3");
    for (int i = 0; i < 36 ; i++) {
      sprintf(histo, "EBLT shape SM%02d L3A", i+1);
      meShapeMapL3A_[i] = dbe->bookProfile2D(histo, histo, 1700, 0., 1700., 10, 0., 10., 4096, 0., 4096., "s");
      dbe->tag(meShapeMapL3A_[i], i+1);
      sprintf(histo, "EBLT amplitude SM%02d L3A", i+1);
      meAmplMapL3A_[i] = dbe->bookProfile2D(histo, histo, 85, 0., 85., 20, 0., 20., 4096, 0., 4096.*12., "s");
      dbe->tag(meAmplMapL3A_[i], i+1);
      sprintf(histo, "EBLT timing SM%02d L3A", i+1);
      meTimeMapL3A_[i] = dbe->bookProfile2D(histo, histo, 85, 0., 85., 20, 0., 20., 250, 0., 10., "s");
      dbe->tag(meTimeMapL3A_[i], i+1);
      sprintf(histo, "EBLT amplitude over PN SM%02d L3A", i+1);
      meAmplPNMapL3A_[i] = dbe->bookProfile2D(histo, histo, 85, 0., 85., 20, 0., 20., 4096, 0., 4096.*12., "s");
      dbe->tag(meAmplPNMapL3A_[i], i+1);

      sprintf(histo, "EBLT shape SM%02d L3B", i+1);
      meShapeMapL3B_[i] = dbe->bookProfile2D(histo, histo, 1700, 0., 1700., 10, 0., 10., 4096, 0., 4096., "s");
      dbe->tag(meShapeMapL3B_[i], i+1);
      sprintf(histo, "EBLT amplitude SM%02d L3B", i+1);
      meAmplMapL3B_[i] = dbe->bookProfile2D(histo, histo, 85, 0., 85., 20, 0., 20., 4096, 0., 4096.*12., "s");
      dbe->tag(meAmplMapL3B_[i], i+1);
      sprintf(histo, "EBLT timing SM%02d L3B", i+1);
      meTimeMapL3B_[i] = dbe->bookProfile2D(histo, histo, 85, 0., 85., 20, 0., 20., 250, 0., 10., "s");
      dbe->tag(meTimeMapL3B_[i], i+1);
      sprintf(histo, "EBLT amplitude over PN SM%02d L3B", i+1);
      meAmplPNMapL3B_[i] = dbe->bookProfile2D(histo, histo, 85, 0., 85., 20, 0., 20., 4096, 0., 4096.*12., "s");
      dbe->tag(meAmplPNMapL3B_[i], i+1);
    }

    dbe->setCurrentFolder("EcalBarrel/EBLaserTask/Laser4");
    for (int i = 0; i < 36 ; i++) {
      sprintf(histo, "EBLT shape SM%02d L4A", i+1);
      meShapeMapL4A_[i] = dbe->bookProfile2D(histo, histo, 1700, 0., 1700., 10, 0., 10., 4096, 0., 4096., "s");
      dbe->tag(meShapeMapL4A_[i], i+1);
      sprintf(histo, "EBLT amplitude SM%02d L4A", i+1);
      meAmplMapL4A_[i] = dbe->bookProfile2D(histo, histo, 85, 0., 85., 20, 0., 20., 4096, 0., 4096.*12., "s");
      dbe->tag(meAmplMapL4A_[i], i+1);
      sprintf(histo, "EBLT timing SM%02d L4A", i+1);
      meTimeMapL4A_[i] = dbe->bookProfile2D(histo, histo, 85, 0., 85., 20, 0., 20., 250, 0., 10., "s");
      dbe->tag(meTimeMapL4A_[i], i+1);
      sprintf(histo, "EBLT amplitude over PN SM%02d L4A", i+1);
      meAmplPNMapL4A_[i] = dbe->bookProfile2D(histo, histo, 85, 0., 85., 20, 0., 20., 4096, 0., 4096.*12., "s");
      dbe->tag(meAmplPNMapL4A_[i], i+1);

      sprintf(histo, "EBLT shape SM%02d L4B", i+1);
      meShapeMapL4B_[i] = dbe->bookProfile2D(histo, histo, 1700, 0., 1700., 10, 0., 10., 4096, 0., 4096., "s");
      dbe->tag(meShapeMapL4B_[i], i+1);
      sprintf(histo, "EBLT amplitude SM%02d L4B", i+1);
      meAmplMapL4B_[i] = dbe->bookProfile2D(histo, histo, 85, 0., 85., 20, 0., 20., 4096, 0., 4096.*12., "s");
      dbe->tag(meAmplMapL4B_[i], i+1);
      sprintf(histo, "EBLT timing SM%02d L4B", i+1);
      meTimeMapL4B_[i] = dbe->bookProfile2D(histo, histo, 85, 0., 85., 20, 0., 20., 250, 0., 10., "s");
      dbe->tag(meTimeMapL4B_[i], i+1);
      sprintf(histo, "EBLT amplitude over PN SM%02d L4B", i+1);
      meAmplPNMapL4B_[i] = dbe->bookProfile2D(histo, histo, 85, 0., 85., 20, 0., 20., 4096, 0., 4096.*12., "s");
      dbe->tag(meAmplPNMapL4B_[i], i+1);
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
      if ( meShapeMapL1A_[i] )  dbe->removeElement( meShapeMapL1A_[i]->getName() );
      meShapeMapL1A_[i] = 0;
      if ( meAmplMapL1A_[i] ) dbe->removeElement( meAmplMapL1A_[i]->getName() );
      meAmplMapL1A_[i] = 0;
      if ( meTimeMapL1A_[i] ) dbe->removeElement( meTimeMapL1A_[i]->getName() );
      meTimeMapL1A_[i] = 0;
      if ( meAmplPNMapL1A_[i] ) dbe->removeElement( meAmplPNMapL1A_[i]->getName() );
      meAmplPNMapL1A_[i] = 0;

      if ( meShapeMapL1B_[i] )  dbe->removeElement( meShapeMapL1B_[i]->getName() );
      meShapeMapL1B_[i] = 0;
      if ( meAmplMapL1B_[i] ) dbe->removeElement( meAmplMapL1B_[i]->getName() );
      meAmplMapL1B_[i] = 0;
      if ( meTimeMapL1B_[i] ) dbe->removeElement( meTimeMapL1B_[i]->getName() );
      meTimeMapL1B_[i] = 0;
      if ( meAmplPNMapL1B_[i] ) dbe->removeElement( meAmplPNMapL1B_[i]->getName() );
      meAmplPNMapL1B_[i] = 0;
    }

    dbe->setCurrentFolder("EcalBarrel/EBLaserTask/Laser2");
    for (int i = 0; i < 36 ; i++) {
      if ( meShapeMapL2A_[i] )  dbe->removeElement( meShapeMapL2A_[i]->getName() );
      meShapeMapL2A_[i] = 0;
      if ( meAmplMapL2A_[i] ) dbe->removeElement( meAmplMapL2A_[i]->getName() );
      meAmplMapL2A_[i] = 0;
      if ( meTimeMapL2A_[i] ) dbe->removeElement( meTimeMapL2A_[i]->getName() );
      meTimeMapL2A_[i] = 0;
      if ( meAmplPNMapL2A_[i] ) dbe->removeElement( meAmplPNMapL2A_[i]->getName() );
      meAmplPNMapL2A_[i] = 0;

      if ( meShapeMapL2B_[i] )  dbe->removeElement( meShapeMapL2B_[i]->getName() );
      meShapeMapL2B_[i] = 0;
      if ( meAmplMapL2B_[i] ) dbe->removeElement( meAmplMapL2B_[i]->getName() );
      meAmplMapL2B_[i] = 0;
      if ( meTimeMapL2B_[i] ) dbe->removeElement( meTimeMapL2B_[i]->getName() );
      meTimeMapL2B_[i] = 0;
      if ( meAmplPNMapL2B_[i] ) dbe->removeElement( meAmplPNMapL2B_[i]->getName() );
      meAmplPNMapL2B_[i] = 0;
    }

    dbe->setCurrentFolder("EcalBarrel/EBLaserTask/Laser3");
    for (int i = 0; i < 36 ; i++) {
      if ( meShapeMapL3A_[i] )  dbe->removeElement( meShapeMapL3A_[i]->getName() );
      meShapeMapL3A_[i] = 0;
      if ( meAmplMapL3A_[i] ) dbe->removeElement( meAmplMapL3A_[i]->getName() );
      meAmplMapL3A_[i] = 0;
      if ( meTimeMapL3A_[i] ) dbe->removeElement( meTimeMapL3A_[i]->getName() );
      meTimeMapL3A_[i] = 0;
      if ( meAmplPNMapL3A_[i] ) dbe->removeElement( meAmplPNMapL3A_[i]->getName() );
      meAmplPNMapL3A_[i] = 0;

      if ( meShapeMapL3B_[i] )  dbe->removeElement( meShapeMapL3B_[i]->getName() );
      meShapeMapL3B_[i] = 0;
      if ( meAmplMapL3B_[i] ) dbe->removeElement( meAmplMapL3B_[i]->getName() );
      meAmplMapL3B_[i] = 0;
      if ( meTimeMapL3B_[i] ) dbe->removeElement( meTimeMapL3B_[i]->getName() );
      meTimeMapL3B_[i] = 0;
      if ( meAmplPNMapL3B_[i] ) dbe->removeElement( meAmplPNMapL3B_[i]->getName() );
      meAmplPNMapL3B_[i] = 0;
    }

    dbe->setCurrentFolder("EcalBarrel/EBLaserTask/Laser4");
    for (int i = 0; i < 36 ; i++) {
      if ( meShapeMapL4A_[i] )  dbe->removeElement( meShapeMapL4A_[i]->getName() );
      meShapeMapL4A_[i] = 0;
      if ( meAmplMapL4A_[i] ) dbe->removeElement( meAmplMapL4A_[i]->getName() );
      meAmplMapL4A_[i] = 0;
      if ( meTimeMapL4A_[i] ) dbe->removeElement( meTimeMapL4A_[i]->getName() );
      meTimeMapL4A_[i] = 0;
      if ( meAmplPNMapL4A_[i] ) dbe->removeElement( meAmplPNMapL4A_[i]->getName() );
      meAmplPNMapL4A_[i] = 0;

      if ( meShapeMapL4B_[i] )  dbe->removeElement( meShapeMapL4B_[i]->getName() );
      meShapeMapL4B_[i] = 0;
      if ( meAmplMapL4B_[i] ) dbe->removeElement( meAmplMapL4B_[i]->getName() );
      meAmplMapL4B_[i] = 0;
      if ( meTimeMapL4B_[i] ) dbe->removeElement( meTimeMapL4B_[i]->getName() );
      meTimeMapL4B_[i] = 0;
      if ( meAmplPNMapL4B_[i] ) dbe->removeElement( meAmplPNMapL4B_[i]->getName() );
      meAmplPNMapL4B_[i] = 0;
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

    map<int, EcalDCCHeaderBlock>::iterator i = dccMap.find(dcch.id());
    if ( i != dccMap.end() ) continue;

    dccMap[dcch.id()] = dcch;

    if ( dcch.getRunType() == EcalDCCHeaderBlock::LASER_STD ) enable = true;

    // uncomment the following line to mix fake 'laser' events w/ cosmic & beam events
    // if ( dcch.getRunType() == EcalDCCHeaderBlock::COSMIC ) enable = true;

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

    map<int, EcalDCCHeaderBlock>::iterator i = dccMap.find(ism-1);
    if ( i == dccMap.end() ) continue;

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

      if ( ie < 6 || ip > 10 ) {

        if ( dccMap[ism-1].getEventSettings().wavelength == 0 ) meShapeMap = meShapeMapL1A_[ism-1];
        if ( dccMap[ism-1].getEventSettings().wavelength == 1 ) meShapeMap = meShapeMapL2A_[ism-1];
        if ( dccMap[ism-1].getEventSettings().wavelength == 2 ) meShapeMap = meShapeMapL3A_[ism-1];
        if ( dccMap[ism-1].getEventSettings().wavelength == 3 ) meShapeMap = meShapeMapL4A_[ism-1];

      } else {

        if ( dccMap[ism-1].getEventSettings().wavelength == 0 ) meShapeMap = meShapeMapL1B_[ism-1];
        if ( dccMap[ism-1].getEventSettings().wavelength == 1 ) meShapeMap = meShapeMapL2B_[ism-1];
        if ( dccMap[ism-1].getEventSettings().wavelength == 2 ) meShapeMap = meShapeMapL3B_[ism-1];
        if ( dccMap[ism-1].getEventSettings().wavelength == 3 ) meShapeMap = meShapeMapL4B_[ism-1];

      }

//      float xval = float(adc) * gain;
      float xval = float(adc);

      if ( meShapeMap ) meShapeMap->Fill(ic - 0.5, i + 0.5, xval);

    }

  }

  Handle<EcalPnDiodeDigiCollection> pns;
  e.getByLabel("ecalEBunpacker", pns);

  float adcA[36];
  float adcB[36];

  for ( int i = 0; i < 36; i++ ) {
    adcA[i] = 0.;
    adcB[i] = 0.;
  }

  int nep = pns->size();
  LogDebug("EBLaserTask") << "event " << ievt_ << " pns collection size " << nep;

  for ( EcalPnDiodeDigiCollection::const_iterator pnItr = pns->begin(); pnItr != pns->end(); ++pnItr ) {

    EcalPnDiodeDigi pn = (*pnItr);
    EcalPnDiodeDetId id = pn.id();

//    int ism = id.ism();
    int ism = id.iDCCId();

    int num = id.iPnId();

    map<int, EcalDCCHeaderBlock>::iterator i = dccMap.find(ism-1);
    if ( i == dccMap.end() ) continue;

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

    if ( num == 1 ) adcA[ism-1] = xvalmax;
    if ( num == 6 ) adcB[ism-1] = xvalmax;

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

    map<int, EcalDCCHeaderBlock>::iterator i = dccMap.find(ism-1);
    if ( i == dccMap.end() ) continue;

    if ( dccMap[ism-1].getRunType() != EcalDCCHeaderBlock::LASER_STD ) continue;

    LogDebug("EBLaserTask") << " det id = " << id;
    LogDebug("EBLaserTask") << " sm, eta, phi " << ism << " " << ie << " " << ip;

    MonitorElement* meAmplMap = 0;
    MonitorElement* meTimeMap = 0;
    MonitorElement* meAmplPNMap = 0;

    if ( ie < 6 || ip > 10 ) {

      if ( dccMap[ism-1].getEventSettings().wavelength == 0 ) {
        meAmplMap = meAmplMapL1A_[ism-1];
        meTimeMap = meTimeMapL1A_[ism-1];
        meAmplPNMap = meAmplPNMapL1A_[ism-1];
      }
      if ( dccMap[ism-1].getEventSettings().wavelength == 1 ) {
        meAmplMap = meAmplMapL2A_[ism-1];
        meTimeMap = meTimeMapL2A_[ism-1];
        meAmplPNMap = meAmplPNMapL2A_[ism-1];
      }
      if ( dccMap[ism-1].getEventSettings().wavelength == 2 ) {
        meAmplMap = meAmplMapL3A_[ism-1];
        meTimeMap = meTimeMapL3A_[ism-1];
        meAmplPNMap = meAmplPNMapL3A_[ism-1];
      }
      if ( dccMap[ism-1].getEventSettings().wavelength == 3 ) {
        meAmplMap = meAmplMapL4A_[ism-1];
        meTimeMap = meTimeMapL4A_[ism-1];
        meAmplPNMap = meAmplPNMapL4A_[ism-1];
      }

    } else {

      if ( dccMap[ism-1].getEventSettings().wavelength == 0 ) {
        meAmplMap = meAmplMapL1B_[ism-1];
        meTimeMap = meTimeMapL1B_[ism-1];
        meAmplPNMap = meAmplPNMapL1B_[ism-1];
      }
      if ( dccMap[ism-1].getEventSettings().wavelength == 1 ) {
        meAmplMap = meAmplMapL2B_[ism-1];
        meTimeMap = meTimeMapL2B_[ism-1];
        meAmplPNMap = meAmplPNMapL2B_[ism-1];
      }
      if ( dccMap[ism-1].getEventSettings().wavelength == 2 ) {
        meAmplMap = meAmplMapL3B_[ism-1];
        meTimeMap = meTimeMapL3B_[ism-1];
        meAmplPNMap = meAmplPNMapL3B_[ism-1];
      }
      if ( dccMap[ism-1].getEventSettings().wavelength == 3 ) {
        meAmplMap = meAmplMapL4B_[ism-1];
        meTimeMap = meTimeMapL4B_[ism-1];
        meAmplPNMap = meAmplPNMapL4B_[ism-1];
      }

    }

    float xval = hit.amplitude();
    if ( xval <= 0. ) xval = 0.0;
    float yval = hit.jitter();
    if ( yval <= 0. ) yval = 0.0;
    float zval = hit.pedestal();
    if ( zval <= 0. ) zval = 0.0;

    LogDebug("EBLaserTask") << " hit amplitude " << xval;
    LogDebug("EBLaserTask") << " hit jitter " << yval;
    LogDebug("EBLaserTask") << " hit pedestal " << zval;

    if ( meAmplMap ) meAmplMap->Fill(xie, xip, xval);

    if ( meTimeMap ) meTimeMap->Fill(xie, xip, yval);

    float wval = 0.;

    if ( ie < 6 || ip > 10 ) {

      if ( adcA[ism-1] != 0. ) wval = xval / adcA[ism-1];

    } else {

      if ( adcB[ism-1] != 0. ) wval = xval / adcB[ism-1];

    }

    LogDebug("EBLaserTask") << " hit amplitude over PN " << wval;

    if ( meAmplPNMap ) meAmplPNMap->Fill(xie, xip, wval);

  }

}

