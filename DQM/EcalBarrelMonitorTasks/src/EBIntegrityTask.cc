/*
 * \file EBIntegrityTask.cc
 *
 * $Date: 2006/01/29 17:21:28 $
 * $Revision: 1.8 $
 * \author G. Della Ricca
 *
*/

#include <DQM/EcalBarrelMonitorTasks/interface/EBIntegrityTask.h>

EBIntegrityTask::EBIntegrityTask(const edm::ParameterSet& ps){

//  logFile_.open("EBIntegrityTask.log");

  init_ = false;

  meIntegrityDCCSize = 0;
  for (int i = 0; i < 36 ; i++) {
    meIntegrityGain[i] = 0;
    meIntegrityChId[i] = 0;
    meIntegrityGainSwitch[i] = 0;
    meIntegrityGainSwitchStay[i] = 0;
    meIntegrityTTId[i] = 0;
    meIntegrityTTBlockSize[i] = 0;
  }

  this->setup();

}

EBIntegrityTask::~EBIntegrityTask(){

//  logFile_.close();

}

void EBIntegrityTask::beginJob(const edm::EventSetup& c){

  ievt_ = 0;

}

void EBIntegrityTask::setup(void){

  init_ = true;

  Char_t histo[20];

  DaqMonitorBEInterface* dbe = 0;

  // get hold of back-end interface
  dbe = edm::Service<DaqMonitorBEInterface>().operator->();

  if ( dbe ) {
    dbe->setCurrentFolder("EcalBarrel");

    // checking when number of towers in data different than expected from header
    dbe->setCurrentFolder("EcalBarrel/EBIntegrityTask");
    sprintf(histo, "EBIT DCC size error");
    meIntegrityDCCSize = dbe->book1D(histo, histo, 36, 1, 37.);

    // checking when the gain is 0
    dbe->setCurrentFolder("EcalBarrel/EBIntegrityTask/Gain");
    for (int i = 0; i < 36 ; i++) {
      sprintf(histo, "EBIT gain SM%02d", i+1);
      meIntegrityGain[i] = dbe->book2D(histo, histo, 85, 0., 85., 20, 0., 20.);
    }

    // checking when channel has unexpected or invalid ID
    dbe->setCurrentFolder("EcalBarrel/EBIntegrityTask/ChId");
    for (int i = 0; i < 36 ; i++) {
      sprintf(histo, "EBIT ChId SM%02d", i+1);
      meIntegrityChId[i] = dbe->book2D(histo, histo, 85, 0., 85., 20, 0., 20.);
    }

    // checking when channel has unexpected or invalid ID
    dbe->setCurrentFolder("EcalBarrel/EBIntegrityTask/GainSwitch");
    for (int i = 0; i < 36 ; i++) {
      sprintf(histo, "EBIT gain switch SM%02d", i+1);
      meIntegrityGainSwitch[i] = dbe->book2D(histo, histo, 85, 0., 85., 20, 0., 20.);
    }

    // checking when channel has unexpected or invalid ID
    dbe->setCurrentFolder("EcalBarrel/EBIntegrityTask/GainSwitchStay");
    for (int i = 0; i < 36 ; i++) {
      sprintf(histo, "EBIT gain switch stay SM%02d", i+1);
      meIntegrityGainSwitchStay[i] = dbe->book2D(histo, histo, 85, 0., 85., 20, 0., 20.);
    }

    // checking when trigger tower has unexpected or invalid ID
    dbe->setCurrentFolder("EcalBarrel/EBIntegrityTask/TTId");
    for (int i = 0; i < 36 ; i++) {
      sprintf(histo, "EBIT TTId SM%02d", i+1);
      meIntegrityTTId[i] = dbe->book2D(histo, histo, 17, 0., 17., 4, 0., 4.);
    }

    // checking when trigger tower has unexpected or invalid size
    dbe->setCurrentFolder("EcalBarrel/EBIntegrityTask/TTBlockSize");
    for (int i = 0; i < 36 ; i++) {
      sprintf(histo, "EBIT TTBlockSize SM%02d", i+1);
      meIntegrityTTBlockSize[i] = dbe->book2D(histo, histo, 17, 0., 17., 4, 0., 4.);
    }

  }

}

void EBIntegrityTask::endJob(){

  cout << "EBIntegrityTask: analyzed " << ievt_ << " events" << endl;

}

void EBIntegrityTask::analyze(const edm::Event& e, const edm::EventSetup& c){

  if ( ! init_ ) this->setup();

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

  edm::Handle<EBDetIdCollection> ids3;
  e.getByLabel("ecalEBunpacker", "EcalIntegrityGainSwitchErrors", ids3);

  for ( EBDetIdCollection::const_iterator idItr = ids3->begin(); idItr != ids3->end(); ++ idItr ) {

    EBDetId id = (*idItr);

    int ie = id.ieta();
    int ip = id.iphi();

    float xie = ie - 0.5;
    float xip = ip - 0.5;

    int ism = id.ism();

    if ( meIntegrityGainSwitch[ism-1] ) meIntegrityGainSwitch[ism-1]->Fill(xie, xip);

  }

  edm::Handle<EBDetIdCollection> ids4;
  e.getByLabel("ecalEBunpacker", "EcalIntegrityGainSwitchStayErrors", ids4);

  for ( EBDetIdCollection::const_iterator idItr = ids4->begin(); idItr != ids4->end(); ++ idItr ) {

    EBDetId id = (*idItr);

    int ie = id.ieta();
    int ip = id.iphi();

    float xie = ie - 0.5;
    float xip = ip - 0.5;

    int ism = id.ism();

    if ( meIntegrityGainSwitchStay[ism-1] ) meIntegrityGainSwitchStay[ism-1]->Fill(xie, xip);

  }

  edm::Handle<EcalTrigTowerDetIdCollection> ids5;
  e.getByLabel("ecalEBunpacker", "EcalIntegrityTTIdErrors", ids5);

  for ( EcalTrigTowerDetIdCollection::const_iterator idItr = ids5->begin(); idItr != ids5->end(); ++ idItr ) {

    EcalTrigTowerDetId id = (*idItr);

    int ie = id.ieta();
    int ip = id.iphi();

    float xie = ie + 0.5;
    float xip = ip + 0.5;

//    int ism = id.ism();
    int ism = 1;

    if ( meIntegrityTTId[ism-1] ) meIntegrityTTId[ism-1]->Fill(xie, xip);

  }

  edm::Handle<EcalTrigTowerDetIdCollection> ids6;
  e.getByLabel("ecalEBunpacker", "EcalIntegrityBlockSizeErrors", ids6);

  for ( EcalTrigTowerDetIdCollection::const_iterator idItr = ids6->begin(); idItr != ids6->end(); ++ idItr ) {

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

