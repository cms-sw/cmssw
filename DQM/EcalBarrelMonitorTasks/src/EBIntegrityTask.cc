/*
 * \file EBIntegrityTask.cc
 *
 * $Date: 2006/04/07 09:34:02 $
 * $Revision: 1.12 $
 * \author G. Della Ricca
 *
 */

#include <DQM/EcalBarrelMonitorTasks/interface/EBIntegrityTask.h>

EBIntegrityTask::EBIntegrityTask(const ParameterSet& ps){

  init_ = false;

  meIntegrityDCCSize = 0;
  for (int i = 0; i < 36 ; i++) {
    meIntegrityGain[i] = 0;
    meIntegrityChId[i] = 0;
    meIntegrityGainSwitch[i] = 0;
    meIntegrityGainSwitchStay[i] = 0;
    meIntegrityTTId[i] = 0;
    meIntegrityTTBlockSize[i] = 0;
    meIntegrityMemChId[i] = 0;
    meIntegrityMemGain[i] = 0;
    meIntegrityMemTTId[i] = 0;
    meIntegrityMemTTBlockSize[i] = 0;
  }
  
}


EBIntegrityTask::~EBIntegrityTask(){

}

void EBIntegrityTask::beginJob(const EventSetup& c){

  ievt_ = 0;

}

void EBIntegrityTask::setup(void){

  init_ = true;

  Char_t histo[20];

  DaqMonitorBEInterface* dbe = 0;

  // get hold of back-end interface
  dbe = Service<DaqMonitorBEInterface>().operator->();

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

    // checking when mem channels have unexpected ID
    dbe->setCurrentFolder("EcalBarrel/EBIntegrityTask/MemChId");
    for (int i = 0; i < 36 ; i++) {
      sprintf(histo, "EBIT MemChId SM%02d", i+1);
      meIntegrityMemChId[i] = dbe->book2D(histo, histo, 10, 0., 10., 5, 0., 5.);
    }

    // checking when mem samples have second bit encoding the gain different from 0
    // note: strictly speaking, this does not corrupt the mem sample gain value (since only first bit is considered)
    // but indicates that data are not completely correct
    dbe->setCurrentFolder("EcalBarrel/EBIntegrityTask/MemGain");
    for (int i = 0; i < 36 ; i++) {
      sprintf(histo, "EBIT MemGain SM%02d", i+1);
      meIntegrityMemGain[i] = dbe->book2D(histo, histo, 10, 0., 10., 5, 0., 5.);
    }
    
    // checking when mem tower block has unexpected ID
    dbe->setCurrentFolder("EcalBarrel/EBIntegrityTask/MemTTId");
    for (int i = 0; i < 36 ; i++) {
      sprintf(histo, "EBIT MemTTId SM%02d", i+1);
      meIntegrityMemTTId[i] = dbe->book2D(histo, histo, 2, 0., 2., 1, 0., 1.);
    }

    // checking when mem tower block has invalid size
    dbe->setCurrentFolder("EcalBarrel/EBIntegrityTask/MemSize");
    for (int i = 0; i < 36 ; i++) {
      sprintf(histo, "EBIT MemSize SM%02d", i+1);
      meIntegrityMemTTBlockSize[i] = dbe->book2D(histo, histo, 2, 0., 2., 1, 0., 1.);
    }
    
  }
  
}// end setup


void EBIntegrityTask::endJob(){

  LogInfo("EBIntegrityTask") << "analyzed " << ievt_ << " events";

}

void EBIntegrityTask::analyze(const Event& e, const EventSetup& c){

  if ( ! init_ ) this->setup();

  ievt_++;

  Handle<EBDetIdCollection> ids0;
  e.getByLabel("ecalEBunpacker", "EcalIntegrityDCCSizeErrors", ids0);

  for ( EBDetIdCollection::const_iterator idItr = ids0->begin(); idItr != ids0->end(); ++ idItr ) {

    EBDetId id = (*idItr);

    int ism = id.ism();

    float xism = ism - 0.5;

    if ( meIntegrityDCCSize ) meIntegrityDCCSize->Fill(xism);

  }

  Handle<EBDetIdCollection> ids1;
  e.getByLabel("ecalEBunpacker", "EcalIntegrityGainErrors", ids1);

  for ( EBDetIdCollection::const_iterator idItr = ids1->begin(); idItr != ids1->end(); ++ idItr ) {

    EBDetId id = (*idItr);

    int ic = id.ic();
    int ie = (ic-1)/20 + 1;
    int ip = (ic-1)%20 + 1;

    int ism = id.ism();

    float xie = ie - 0.5;
    float xip = ip - 0.5;

    if ( meIntegrityGain[ism-1] ) meIntegrityGain[ism-1]->Fill(xie, xip);

  }

  Handle<EBDetIdCollection> ids2;
  e.getByLabel("ecalEBunpacker", "EcalIntegrityChIdErrors", ids2);

  for ( EBDetIdCollection::const_iterator idItr = ids2->begin(); idItr != ids2->end(); ++ idItr ) {

    EBDetId id = (*idItr);

    int ic = id.ic();
    int ie = (ic-1)/20 + 1;
    int ip = (ic-1)%20 + 1;

    int ism = id.ism();

    float xie = ie - 0.5;
    float xip = ip - 0.5;

    if ( meIntegrityChId[ism-1] ) meIntegrityChId[ism-1]->Fill(xie, xip);

  }

  Handle<EBDetIdCollection> ids3;
  e.getByLabel("ecalEBunpacker", "EcalIntegrityGainSwitchErrors", ids3);

  for ( EBDetIdCollection::const_iterator idItr = ids3->begin(); idItr != ids3->end(); ++ idItr ) {

    EBDetId id = (*idItr);

    int ic = id.ic();
    int ie = (ic-1)/20 + 1;
    int ip = (ic-1)%20 + 1;

    int ism = id.ism();

    float xie = ie - 0.5;
    float xip = ip - 0.5;

    if ( meIntegrityGainSwitch[ism-1] ) meIntegrityGainSwitch[ism-1]->Fill(xie, xip);

  }

  Handle<EBDetIdCollection> ids4;
  e.getByLabel("ecalEBunpacker", "EcalIntegrityGainSwitchStayErrors", ids4);

  for ( EBDetIdCollection::const_iterator idItr = ids4->begin(); idItr != ids4->end(); ++ idItr ) {

    EBDetId id = (*idItr);

    int ic = id.ic();
    int ie = (ic-1)/20 + 1;
    int ip = (ic-1)%20 + 1;

    int ism = id.ism();

    float xie = ie - 0.5;
    float xip = ip - 0.5;

    if ( meIntegrityGainSwitchStay[ism-1] ) meIntegrityGainSwitchStay[ism-1]->Fill(xie, xip);

  }

  Handle<EcalTrigTowerDetIdCollection> ids5;
  e.getByLabel("ecalEBunpacker", "EcalIntegrityTTIdErrors", ids5);

  for ( EcalTrigTowerDetIdCollection::const_iterator idItr = ids5->begin(); idItr != ids5->end(); ++ idItr ) {

    EcalTrigTowerDetId id = (*idItr);

    int iet = id.ieta();
    int ipt = id.iphi();

//    int ismt = id.ism();
    int ismt = 1;

    float xiet = iet + 0.5;
    float xipt = ipt + 0.5;

    if ( meIntegrityTTId[ismt-1] ) meIntegrityTTId[ismt-1]->Fill(xiet, xipt);

  }


  Handle<EcalTrigTowerDetIdCollection> ids6;
  e.getByLabel("ecalEBunpacker", "EcalIntegrityBlockSizeErrors", ids6);

  for ( EcalTrigTowerDetIdCollection::const_iterator idItr = ids6->begin(); idItr != ids6->end(); ++ idItr ) {

    EcalTrigTowerDetId id = (*idItr);

    int iet = id.ieta();
    int ipt = id.iphi();

    //    int ismt = id.ism();
    int ismt = 1;
    
    //     if ( meIntegrityTTBlockSize[ism-1] ) meIntegrityTTBlockSize[ism-1]->Fill(xie, xip);
     if ( meIntegrityTTBlockSize[ismt-1] ) meIntegrityTTBlockSize[ismt-1]->Fill(iet, ipt);
     // gio: here I am in doubt !!!!!!!
   }



  Handle<EcalElectronicsIdCollection> ids7;
  e.getByLabel("ecalEBunpacker", "EcalIntegrityMemTtIdErrors", ids7);

  for ( EcalElectronicsIdCollection::const_iterator idItr = ids7->begin(); idItr != ids7->end(); ++ idItr ) {

    EcalElectronicsId id = (*idItr);

    int itt   = id.towerId();
    float iTt = itt + 0.5 - 69;
    int ism = 1;

    if ( meIntegrityMemTTId[ism-1] ) meIntegrityMemTTId[ism-1]->Fill(iTt,1);
    
  }



  Handle<EcalElectronicsIdCollection> ids8;
  e.getByLabel("ecalEBunpacker", "EcalIntegrityMemBlockSize", ids8);

  for ( EcalElectronicsIdCollection::const_iterator idItr = ids8->begin(); idItr != ids8->end(); ++ idItr ) {

    EcalElectronicsId id = (*idItr);
    
    int itt   = id.towerId();
    float iTt = itt + 0.5 -69;
    int ism = 1;
    
    if ( meIntegrityMemTTBlockSize[ism-1] ) meIntegrityMemTTBlockSize[ism-1]->Fill(iTt,1);
    
  }

  
  
  Handle<EcalElectronicsIdCollection> ids9;
  e.getByLabel("ecalEBunpacker", "EcalIntegrityMemChIdErrors", ids9);

  for ( EcalElectronicsIdCollection::const_iterator idItr = ids9->begin(); idItr != ids9->end(); ++ idItr ) {

    EcalElectronicsId id = (*idItr);

    int ism = 1;

    int chid = id.channelId();
    int ie = EBIntegrityTask::chMemAbscissa[chid];
    int ip = EBIntegrityTask::chMemOrdinate[chid];

    int iTt = id.towerId();
    ie += (iTt-69)*5;

    float xie = ie - 0.5;
    float xip = ip - 0.5;
        
    if ( meIntegrityMemChId[ism-1] ) meIntegrityMemChId[ism-1]->Fill(xie,xip);
    
  }




// //=======
//     float xiet = iet + 0.5;
//     float xipt = ipt + 0.5;
//     //>>>>>>> 1.12


//     //<<<<<<< EBIntegrityTask.cc
//     //=======
//     if ( meIntegrityTTBlockSize[ismt-1] ) meIntegrityTTBlockSize[ismt-1]->Fill(xiet, xipt);
//     //>>>>>>> 1.12


  Handle<EcalElectronicsIdCollection> ids10;
  e.getByLabel("ecalEBunpacker", "EcalIntegrityMemGainErrors", ids10);

  for ( EcalElectronicsIdCollection::const_iterator idItr = ids10->begin(); idItr != ids10->end(); ++ idItr ) {

    EcalElectronicsId id = (*idItr);

    int ism = 1;

    int chid = id.channelId();
    int ie = EBIntegrityTask::chMemAbscissa[chid];
    int ip = EBIntegrityTask::chMemOrdinate[chid];

    int iTt = id.towerId();
    ie += (iTt-69)*5;

    float xie = ie - 0.5;
    float xip = ip - 0.5;
        
    
    if ( meIntegrityMemGain[ism-1] ) meIntegrityMemGain[ism-1]->Fill(xie,xip);
    
  }

}//  end analyze



const	 int  EBIntegrityTask::chMemAbscissa [25] = 
  {
    1,1, 1, 1, 1,
    2, 2, 2, 2, 2,
    3, 3, 3, 3, 3,
    4, 4, 4, 4, 4,
    5, 5, 5, 5, 5
  };

const	 int  EBIntegrityTask::chMemOrdinate [25] = 
  {
    1,2, 3, 4, 5,
    5, 4, 3, 2, 1,
    1, 2, 3, 4, 5,
    5, 4, 3, 2, 1,
    1, 2, 3,4, 5
  };
