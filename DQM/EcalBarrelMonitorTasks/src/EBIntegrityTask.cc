/*
 * \file EBIntegrityTask.cc
 * 
 * $Date: 2005/11/25 12:57:37 $
 * $Revision: 1.28 $
 * \author G. Della Ricca
 *
*/

#include <DQM/EcalBarrelMonitorTasks/interface/EBIntegrityTask.h>

EBIntegrityTask::EBIntegrityTask(const edm::ParameterSet& ps, DaqMonitorBEInterface* dbe){

//  logFile_.open("EBIntegrityTask.log");

  Char_t histo[20];

  if ( dbe ) {
    dbe->setCurrentFolder("EcalBarrel");
    dbe->setCurrentFolder("EcalBarrel/EcalIntegrity");
    dbe->setCurrentFolder("EcalBarrel/EcalIntegrity/Gain");
    for (int i = 0; i < 36 ; i++) {
      sprintf(histo, "EI gain SM%02d", i+1);
      meIntegrityGain[i] = dbe->book2D(histo, histo, 85, 0., 85., 20, 0., 20.);
    } 
    
    // checking when channel has unexpected or invalid ID
    dbe->setCurrentFolder("EcalBarrel/EcalIntegrity/ChId");
    for (int i = 0; i < 36 ; i++) {
      sprintf(histo, "EI ChId SM%02d", i+1);
      meIntegrityChId[i] = dbe->book2D(histo, histo, 85, 0., 85., 20, 0., 20.);
    } 

    // checking when trigger tower has unexpected or invalid ID
    dbe->setCurrentFolder("EcalBarrel/EcalIntegrity/TTId");
    for (int i = 0; i < 36 ; i++) {
      sprintf(histo, "EI TTId SM%02d", i+1);
      meIntegrityTTId[i] = dbe->book2D(histo, histo, 17, 0., 17., 4, 0., 4.);
    }

    dbe->setCurrentFolder("EcalBarrel/EcalIntegrity/TTBlockSize");
    for (int i = 0; i < 36 ; i++) {
      sprintf(histo, "EI TTBlockSize SM%02d", i+1);
      meIntegrityTTBlockSize[i] = dbe->book2D(histo, histo, 17, 0., 17., 4, 0., 4.);
    } 

    // checking when number of towers in data different than expected from header
    dbe->setCurrentFolder("EcalBarrel/EcalIntegrity");
    sprintf(histo, "DCC size error");
    meIntegrityDCCSize = dbe->book1D(histo, histo, 36, 1, 37.);

  }

}

EBIntegrityTask::~EBIntegrityTask(){

//  logFile_.close();

}

void EBIntegrityTask::beginJob(const edm::EventSetup& c){

  ievt_ = 0;

}

void EBIntegrityTask::endJob(){

  cout << "EBIntegrityTask: analyzed " << ievt_ << " events" << endl;

}

void EBIntegrityTask::analyze(const edm::Event& e, const edm::EventSetup& c){

  ievt_++;

  if ( meIntegrityDCCSize ) meIntegrityDCCSize->Fill(1.);

  if ( meIntegrityTTId[1] ) meIntegrityTTId[1]->Fill(2., 3.);

  if ( meIntegrityTTBlockSize[1] ) meIntegrityTTBlockSize[1]->Fill(2., 3.);

  if ( meIntegrityChId[1] ) meIntegrityChId[1]->Fill(2., 3.);

  if ( meIntegrityGain[1] ) meIntegrityGain[1]->Fill(2., 3.);

}

