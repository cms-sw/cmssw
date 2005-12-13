/*
 * \file EBIntegrityTask.cc
 * 
 * $Date: 2005/12/12 07:26:28 $
 * $Revision: 1.2 $
 * \author G. Della Ricca
 *
*/

#include <DQM/EcalBarrelMonitorTasks/interface/EBIntegrityTask.h>

EBIntegrityTask::EBIntegrityTask(const edm::ParameterSet& ps, DaqMonitorBEInterface* dbe){

//  logFile_.open("EBIntegrityTask.log");

  Char_t histo[20];

  if ( dbe ) {
    dbe->setCurrentFolder("EcalBarrel");

    // checking when number of towers in data different than expected from header
    dbe->setCurrentFolder("EcalBarrel/EcalIntegrity");
    sprintf(histo, "DCC size error");
    meIntegrityDCCSize = dbe->book1D(histo, histo, 36, 1, 37.);

    // checking when the gain is 0
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

    // checking when trigger tower has unexpected or invalid size
    dbe->setCurrentFolder("EcalBarrel/EcalIntegrity/TTBlockSize");
    for (int i = 0; i < 36 ; i++) {
      sprintf(histo, "EI TTBlockSize SM%02d", i+1);
      meIntegrityTTBlockSize[i] = dbe->book2D(histo, histo, 17, 0., 17., 4, 0., 4.);
    } 

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

  edm::Handle<EBDetIdCollection> ids0;
  e.getByLabel("ecalEBunpacker", "EcalIntegrityDCCSizeErrors", ids0);

  for ( EBDetIdCollection::const_iterator idItr = ids0->begin(); idItr != ids0->end(); ++ idItr ) {

    EBDetId id = (*idItr);

    int ism = id.ism();

    float xism = ism - 0.5;

    if ( meIntegrityDCCSize ) meIntegrityDCCSize->Fill(xism);

  }

  edm::Handle<EBDetIdCollection> ids1;
  e.getByLabel("ecalEBunpacker", "EcalIntegrityGainErrors", ids1);

  for ( EBDetIdCollection::const_iterator idItr = ids1->begin(); idItr != ids1->end(); ++ idItr ) {

    EBDetId id = (*idItr);

    int ie = id.ieta();
    int ip = id.iphi();

    float xie = ie - 0.5;
    float xip = ip - 0.5;

    int ism = id.ism();

    if ( meIntegrityGain[ism-1] ) meIntegrityGain[ism-1]->Fill(xie, xip);

  }

  edm::Handle<EBDetIdCollection> ids2;
  e.getByLabel("ecalEBunpacker", "EcalIntegrityChIdErrors", ids2);

  for ( EBDetIdCollection::const_iterator idItr = ids2->begin(); idItr != ids2->end(); ++ idItr ) {

    EBDetId id = (*idItr);

    int ie = id.ieta();
    int ip = id.iphi();

    float xie = ie - 0.5;
    float xip = ip - 0.5;

    int ism = id.ism();

    if ( meIntegrityChId[ism-1] ) meIntegrityChId[ism-1]->Fill(xie, xip);

  }

  edm::Handle<EcalTrigTowerDetIdCollection> ids3;
  e.getByLabel("ecalEBunpacker", "EcalIntegrityTTIdErrors", ids3);

  for ( EcalTrigTowerDetIdCollection::const_iterator idItr = ids3->begin(); idItr != ids3->end(); ++ idItr ) {

    EcalTrigTowerDetId id = (*idItr);

    int ie = id.ieta();
    int ip = id.iphi();

    float xie = ie + 0.5;
    float xip = ip + 0.5;

//    int ism = id.ism();
    int ism = 1;

    if ( meIntegrityTTId[ism-1] ) meIntegrityTTId[ism-1]->Fill(xie, xip);

  }

  edm::Handle<EcalTrigTowerDetIdCollection> ids4;
  e.getByLabel("ecalEBunpacker", "EcalIntegrityBlockSizeErrors", ids4);

  for ( EcalTrigTowerDetIdCollection::const_iterator idItr = ids4->begin(); idItr != ids4->end(); ++ idItr ) {

    EcalTrigTowerDetId id = (*idItr);

    int ie = id.ieta();
    int ip = id.iphi();

    float xie = ie + 0.5;
    float xip = ip + 0.5;

//    int ism = id.ism();
    int ism = 1;

    if ( meIntegrityTTBlockSize[ism-1] ) meIntegrityTTBlockSize[ism-1]->Fill(xie, xip);

  }

}

