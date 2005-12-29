/*
 * \file EBPnDiodeTask.cc
 * 
 * $Date: 2005/12/20 13:51:51 $
 * $Revision: 1.7 $
 * \author G. Della Ricca
 *
*/

#include <DQM/EcalBarrelMonitorTasks/interface/EBPnDiodeTask.h>

EBPnDiodeTask::EBPnDiodeTask(const edm::ParameterSet& ps, DaqMonitorBEInterface* dbe){

//  logFile.open("EBPnDiodeTask.log");

  Char_t histo[20];

  if ( dbe ) {
    dbe->setCurrentFolder("EcalBarrel/EBPnDiodeTask");

    dbe->setCurrentFolder("EcalBarrel/EBPnDiodeTask/Laser1");
    for (int i = 0; i < 36 ; i++) {
      sprintf(histo, "EBPDT PNs SM%02d L1", i+1);
      mePNL1_[i] = dbe->bookProfile2D(histo, histo, 1, 0., 1., 10, 0., 10., 4096, 0., 4096.);
    }

    dbe->setCurrentFolder("EcalBarrel/EBPnDiodeTask/Laser2");
    for (int i = 0; i < 36 ; i++) {
      sprintf(histo, "EBPDT PNs SM%02d L2", i+1);
      mePNL2_[i] = dbe->bookProfile2D(histo, histo, 1, 0., 1., 10, 0., 10., 4096, 0., 4096.);
    }

    dbe->setCurrentFolder("EcalBarrel/EBPnDiodeTask/Laser3");
    for (int i = 0; i < 36 ; i++) {
      sprintf(histo, "EBPDT PNs SM%02d L3", i+1);
      mePNL3_[i] = dbe->bookProfile2D(histo, histo, 1, 0., 1., 10, 0., 10., 4096, 0., 4096.);
    }

    dbe->setCurrentFolder("EcalBarrel/EBPnDiodeTask/Laser4");
    for (int i = 0; i < 36 ; i++) {
      sprintf(histo, "EBPDT PNs SM%02d L4", i+1);
      mePNL4_[i] = dbe->bookProfile2D(histo, histo, 1, 0., 1., 10, 0., 10., 4096, 0., 4096.);
    }

  }

}

EBPnDiodeTask::~EBPnDiodeTask(){

//  logFile.close();

}

void EBPnDiodeTask::beginJob(const edm::EventSetup& c){

  ievt_ = 0;
    
}

void EBPnDiodeTask::endJob(){

  cout << "EBPnDiodeTask: analyzed " << ievt_ << " events" << endl;

}

void EBPnDiodeTask::analyze(const edm::Event& e, const edm::EventSetup& c){

  ievt_++;

  edm::Handle<EcalPnDiodeDigiCollection> pns;
  e.getByLabel("ecalEBunpacker", pns);

//  int nep = pns->size();
//  cout << "EBTestPulseTask: event " << ievt_ << " pns collection size " << nep << endl;

  for ( EcalPnDiodeDigiCollection::const_iterator pnItr = pns->begin(); pnItr != pns->end(); ++pnItr ) {

    EcalPnDiodeDigi pn = (*pnItr);
    EcalPnDiodeDetId id = pn.id();


//    int ism = id.ism();
    int ism = id.iDCCId();
    int num = id.iPnId();

//    logFile << " det id = " << id << endl;
//    logFile << " sm, num " << ism << " " << num << endl;

    float xvalped = 0.;

    for (int i = 0; i < 4; i++) {

      EcalFEMSample sample = pn.sample(i);

      xvalped = xvalped + sample.adc();

    }

    xvalped = xvalped / 4;

    float xvalmax = 0.;

    MonitorElement* mePN = 0;

    for (int i = 0; i < 50; i++) {

      EcalFEMSample sample = pn.sample(i);

      float xval = sample.adc();

//    logFile << " hit amplitude " << xval << endl;

      if ( xval >= xvalmax ) xvalmax = xval;

    }

    xvalmax = xvalmax - xvalped;

    if ( ievt_ >=    1 && ievt_ <=  600 ) mePN = mePNL1_[ism-1];
    if ( ievt_ >=  601 && ievt_ <= 1200 ) mePN = mePNL1_[ism-1];
    if ( ievt_ >= 1201 && ievt_ <= 1800 ) mePN = mePNL2_[ism-1];
    if ( ievt_ >= 1801 && ievt_ <= 2400 ) mePN = mePNL2_[ism-1];

    if ( mePN ) mePN->Fill(0.5, num - 0.5, xvalmax);

  }

}

