/*
 * \file EcalBarrelMonitorModule.cc
 * 
 * $Date: 2005/10/18 08:02:54 $
 * $Revision: 1.19 $
 * \author G. Della Ricca
 *
*/

#include <DQM/EcalBarrelMonitorModule/interface/EcalBarrelMonitorModule.h>

EcalBarrelMonitorModule::EcalBarrelMonitorModule(const edm::ParameterSet& ps){

  logFile.open("EcalBarrelMonitorModule.log");

  dbe = 0;
  if ( ps.getUntrackedParameter<bool>("DBEinterface", false) ) {
    dbe = edm::Service<DaqMonitorBEInterface>().operator->();

    dbe->setVerbose(0);

    edm::Service<MonitorDaemon> daemon;
    daemon.operator->();
  }

  Char_t histo[20];

  outputFile = ps.getUntrackedParameter<string>("outputFile", "");
  if ( outputFile.size() != 0 ) {
    cout << "Ecal Barrel Monitoring histograms will be saved to " << outputFile.c_str() << endl;
  }

  if ( dbe ) {
    dbe->setCurrentFolder("EcalBarrel");
    meStatus  = dbe->bookInt("STATUS");
    meRun     = dbe->bookInt("RUN");
    meEvt     = dbe->bookInt("EVT");

    dbe->setCurrentFolder("EcalBarrel");
    meEbarrel = dbe->book1D("EBMM hits", "EBMM hits ", 100, 0., 61201.);

    dbe->setCurrentFolder("EcalBarrel/EBMonitorEvent");
    for (int i = 0; i < 36 ; i++) {
      sprintf(histo, "EBMM event SM%02d", i+1);
      meEvent[i] = dbe->book2D(histo, histo, 20, 0., 20., 85, 0., 85.);
      meEvent[i]->setResetMe(true);
    }
  }

  pedestal_task  = new EBPedestalTask(ps, dbe);

  testpulse_task = new EBTestPulseTask(ps, dbe);

  laser_task     = new EBLaserTask(ps, dbe);

  cosmic_task    = new EBCosmicTask(ps, dbe);

  if ( dbe ) dbe->showDirStructure();

}

EcalBarrelMonitorModule::~EcalBarrelMonitorModule(){

  delete pedestal_task;

  delete testpulse_task;

  delete laser_task;

  delete cosmic_task;

  logFile.close();

}

void EcalBarrelMonitorModule::beginJob(const edm::EventSetup& c){

  ievt = 0;

  if ( meStatus ) meStatus->Fill(0);

  pedestal_task->beginJob(c);

  testpulse_task->beginJob(c);

  laser_task->beginJob(c);

  cosmic_task->beginJob(c);

}

void EcalBarrelMonitorModule::endJob(void) {

  pedestal_task->endJob();

  testpulse_task->endJob();

  laser_task->endJob();

  cosmic_task->endJob();

  cout << "EcalBarrelMonitorModule: analyzed " << ievt << " events" << endl;

  if ( meStatus ) meStatus->Fill(2);

  if ( outputFile.size() != 0  && dbe ) dbe->save(outputFile);

  sleep(60);
}

void EcalBarrelMonitorModule::analyze(const edm::Event& e, const edm::EventSetup& c){

  if ( meStatus ) meStatus->Fill(1);

  ievt++;

  if ( meRun ) meRun->Fill(14316);
  if ( meEvt ) meEvt->Fill(ievt);

  edm::Handle<EBDigiCollection>  digis;
  e.getByLabel("ecalEBunpacker", digis);

  int neb = digis->size();

  cout << "EcalBarrelMonitorModule: event " << ievt << " collection size " << neb << endl;

  if ( meEbarrel ) meEbarrel->Fill(float(neb));

  for ( EBDigiCollection::const_iterator digiItr = digis->begin(); digiItr != digis->end(); ++digiItr ) {

//    logFile << " Dump the ADC counts for this event " << endl;
//    for ( int i=0; i< (*digiItr).size() ; ++i ) {
//      logFile <<  (*digiItr).sample(i) << " ";
//    }       
//    logFile << " " << endl;

    EBDataFrame dataframe = (*digiItr);
    EBDetId id = dataframe.id();

    int ie = id.ieta();
    int ip = id.iphi();
    int iz = id.zside();

    float xie = iz * (ie - 0.5);
    float xip = ip - 0.5;

    int ism = id.ism();

    logFile << " det id = " << id << endl;
    logFile << " sm, eta, phi " << ism << " " << ie*iz << " " << ip << endl;

    if ( xie <= 0. || xie >= 85. || xip <= 0. || xip >= 20. ) {
      logFile << "ERROR:" << xie << " " << xip << " " << ie << " " << ip << " " << iz << endl;
      return;
    }

    float xvalmax = 0.;

    for (int i = 0; i < 10; i++) {

      EcalMGPASample sample = dataframe.sample(i);
      int adc = sample.adc();
      float gain = 1.;

      if ( sample.gainId() == 1 ) {
        gain = 1./12.;
      }
      if ( sample.gainId() == 2 ) {
        gain = 1./ 6.;
      }
      if ( sample.gainId() == 3 ) {
        gain = 1./ 1.;
      }

      float xval = adc * gain;

      float xrms = 1.0;

      if ( xval >= 3.0 * xrms && xval >= xvalmax ) xvalmax = xval;

    }

    if ( meEvent[ism-1] ) meEvent[ism-1]->Fill(xip, xie, xvalmax);

  }

  pedestal_task->analyze(e, c);

  testpulse_task->analyze(e, c);

  laser_task->analyze(e, c);

  cosmic_task->analyze(e, c);

}

