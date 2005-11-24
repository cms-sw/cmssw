/*
 * \file EBPnDiodeTask.cc
 * 
 * $Date: 2005/11/22 16:04:29 $
 * $Revision: 1.27 $
 * \author G. Della Ricca
 *
*/

#include <DQM/EcalBarrelMonitorTasks/interface/EBPnDiodeTask.h>

EBPnDiodeTask::EBPnDiodeTask(const edm::ParameterSet& ps, DaqMonitorBEInterface* dbe){

//  logFile.open("EBPnDiodeTask.log");

  Char_t histo[20];

  if ( dbe ) {
    dbe->setCurrentFolder("EcalBarrel/EBPnDiodeTask");
    for (int i = 0; i < 36 ; i++) {
      sprintf(histo, "EBLT PNs SM%02d", i+1);
      mePN_[i] = dbe->bookProfile(histo, histo, 10, 0., 10., 4096, 0., 4096.);
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

  edm::Handle<EcalPnDiodeDigiCollection>  pns;
  e.getByLabel("ecalEBunpacker", pns);

//  int nep = pns->size();
//  cout << "EBTestPulseTask: event " << ievt_ << " pns collection size " << nep << endl;

  for ( EcalPnDiodeDigiCollection::const_iterator pnItr = pns->begin(); pnItr != pns->end(); ++pnItr ) {

    EcalPnDiodeDigi pn = (*pnItr);
    EcalPnDiodeDetId id = pn.id();


//    int ism = id.ism();
    int ism = id.iDCCId();
    int num = id.iPnId();

    float xvalmax = 0.;

    for (int i = 0; i < 50; i++) {

      EcalFEMSample sample = pn.sample(i);

      float xval = sample.adc();

      if ( xvalmax >= xval ) xval = xvalmax;

    }

    if ( num == 1 ) mePN_[ism-1]->Fill(float(num), xvalmax);

  }

}

