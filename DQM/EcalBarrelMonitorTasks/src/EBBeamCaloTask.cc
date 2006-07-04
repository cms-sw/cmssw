/*
 * \file EBBeamCaloTask.cc
 *
 * $Date: 2006/06/27 08:40:58 $
 * $Revision: 1.13 $
 * \author A. Ghezzi
 *
 */


#include <DQM/EcalBarrelMonitorTasks/interface/EBBeamCaloTask.h>
#include <DQM/EcalBarrelMonitorTasks/interface/EBMUtilsTasks.h>

EBBeamCaloTask::EBBeamCaloTask(const ParameterSet& ps){

  //LogDebug("EBBeamCaloTask") << " 1 construct  ";
  init_ = false;
  //digiProducer_ = ps.getUntrackedParameter<string>("digiProducer", "ecalEBunpacker");
  digiProducer_ = ps.getParameter<string>("digiProducer");
  //DCCHeaderProducer_= ps.getUntrackedParameter<string>("dccHeaderProducer", "ecalEBunpacker");

  for (int i = 0; i < cryInArray_ ; i++) {
    meBBCaloPulseProf_[i]=0;
    meBBCaloPulseProfG12_[i]=0;
    meBBCaloGains_[i]=0;
    meBBCaloEne_[i]=0;

    meBBCaloPulseProfMoving_[i]=0;
    meBBCaloPulseProfG12Moving_[i]=0;
    meBBCaloGainsMoving_[i]=0;
    meBBCaloEneMoving_[i]=0;
  }

  meBBCaloCryRead_ = 0;
  meBBCaloCryReadMoving_ = 0;

  meBBNumCaloCryRead_ = 0;
  meBBCaloAllNeededCry_ = 0;

  meBBCaloE3x3_ = 0;
  meBBCaloE3x3Moving_ = 0;

  meBBCaloCryOnBeam_ = 0;
  meBBCaloMaxEneCry_ = 0;
  TableMoving_ = 0;

//   for(int u=0;u<1701;u++){
//     meBBCaloE3x3Cry_[u]=0;
//     meBBCaloE1Cry_[u]=0;
//   }
  // LogDebug("EBBeamCaloTask") << " 2 construct  ";
  

}

EBBeamCaloTask::~EBBeamCaloTask(){

}

void EBBeamCaloTask::beginJob(const EventSetup& c){

  ievt_ = 0;

}

void EBBeamCaloTask::setup(void){

  init_ = true;

  Char_t histo[200];

  PreviousTableStatus_[0]=0;//let's start with stable...
  PreviousTableStatus_[1]=0;//let's start with stable...

  lastStableStatus_=0;
  for(int u=0;u<10;u++){cib_[u]=0;}
  changed_tb_status_= false;
  evt_after_change_ =0;
  wasFakeChange_= false;
  table_step_=1;
  crystal_step_=1;
  DaqMonitorBEInterface* dbe = 0;

  // get hold of back-end interface
  dbe = Service<DaqMonitorBEInterface>().operator->();

  if ( dbe ) {
    dbe->setCurrentFolder("EcalBarrel/EBBeamCaloTask");
    
    for (int i = 0; i < cryInArray_ ; i++) {
      sprintf(histo, "EBBCT pulse profile cry %01d", i+1);
      //considering the gain the range is 4096*12 ~ 50000
      meBBCaloPulseProf_[i] = dbe->bookProfile(histo, histo, 10,0.,10.,50000,0.,50000.,"s");

      sprintf(histo, "EBBCT pulse profile in G12 cry %01d", i+1);
      meBBCaloPulseProfG12_[i] = dbe->bookProfile(histo, histo, 10,0.,10.,4096,0.,4096.,"s");

      sprintf(histo, "EBBCT found gains cry %01d", i+1);
      meBBCaloGains_[i] =  dbe->book1D(histo,histo,14,0.,14.);
      // g1-> bin 2, g6-> bin 7, g12-> bin 13
      
      sprintf(histo, "EBBCT rec energy cry %01d", i+1);
      meBBCaloEne_[i] =  dbe->book1D(histo,histo,2000,0.,9000.);
      //9000 ADC in G12 equivalent is about 330 GeV

      //////////////////////////////// me for the moving table////////////////////////////////////////////

      sprintf(histo, "EBBCT pulse profile moving table cry %01d", i+1);
      //considering the gain the range is 4096*12 ~ 50000
      meBBCaloPulseProfMoving_[i] = dbe->bookProfile(histo, histo, 10,0.,10.,50000,0.,50000.,"s");

      sprintf(histo, "EBBCT pulse profile in G12 moving table cry %01d", i+1);
      meBBCaloPulseProfG12Moving_[i] = dbe->bookProfile(histo, histo, 10,0.,10.,4096,0.,4096.,"s");

      sprintf(histo, "EBBCT found gains moving table cry %01d", i+1);
      meBBCaloGainsMoving_[i] =  dbe->book1D(histo,histo,14,0.,14.);
      // g1-> bin 2, g6-> bin 7, g12-> bin 13

      sprintf(histo, "EBBCT rec energy moving table cry %01d", i+1);
      meBBCaloEneMoving_[i] =  dbe->book1D(histo,histo,2000,0.,9000.);
      //9000 ADC in G12 equivalent is about 330 GeV

    }
    
//     dbe->setCurrentFolder("EcalBarrel/EBBeamCaloTask/EnergyHistos");
//     for(int u=0; u< 1701;u++){
//       sprintf(histo, "EBBCT rec Ene sum 3x3 cry: %04d",u);
//       meBBCaloE3x3Cry_[u] = dbe->book1D(histo,histo,1000,0.,4500.);

//       sprintf(histo, "EBBCT rec Energy1 cry: %04d",u);
//       meBBCaloE1Cry_[u] = dbe->book1D(histo,histo,1000,0.,4500.);
//     }
    
//     dbe->setCurrentFolder("EcalBarrel/EBBeamCaloTask");
    sprintf(histo, "EBBCT readout crystals");
    meBBCaloCryRead_  =  dbe->book2D(histo,histo,9,-4.,5.,9,-4.,5.);
    //matrix of readout crystal around cry in beam

    sprintf(histo, "EBBCT readout crystals table moving");
    meBBCaloCryReadMoving_  =  dbe->book2D(histo,histo,9,-4.,5.,9,-4.,5.);
    //matrix of readout crystal around cry in beam

    sprintf(histo, "EBBCT all needed crystals readout");
    meBBCaloAllNeededCry_ = dbe->book1D(histo,histo,3,-1.,2.);
    // not all needed cry are readout-> bin 1, all needed cry are readout-> bin 3
    
    sprintf(histo, "EBBCT number of readout crystals");
    meBBNumCaloCryRead_ = dbe->book1D(histo,histo,1700,1.,1701.);
    
    sprintf(histo, "EBBCT rec Ene sum 3x3");
    meBBCaloE3x3_ = dbe->book1D(histo,histo,9000,0.,9000.);
    //9000 ADC in G12 equivalent is about 330 GeV

    sprintf(histo, "EBBCT rec Ene sum 3x3 table moving");
    meBBCaloE3x3Moving_ = dbe->book1D(histo,histo,9000,0.,9000.);
    //9000 ADC in G12 equivalent is about 330 GeV
    
    sprintf(histo, "EBBCT crystal on beam");
    meBBCaloCryOnBeam_ = dbe->book2D(histo, histo, 85, 0., 85., 20, 0., 20.);
    
    sprintf(histo, "EBBCT crystal with maximum rec energy");
    meBBCaloMaxEneCry_ = dbe->book2D(histo, histo, 85, 0., 85., 20, 0., 20.);
    
    sprintf(histo, "EBBCT table is moving");
    TableMoving_ = dbe->book1D(histo,histo,2,0.,1.1);
    //table is moving-> bin 1, table is not moving-> bin 2

    sprintf(histo, "EBBCT crystals done");
    CrystalsDone_= dbe->book1D(histo,histo,1700,0.,1700.);
    
    sprintf(histo, "EBBCT crystal in beam vs event");
    CrystalInBeam_vs_Event_ = dbe->bookProfile(histo, histo, 20000,0.,200000.,1802,-101.,1701.,"s");
    // 1 bin each 10 events
    // when table is moving for just one events fill with -100
    // when table is constantely moving do not fill...

    sprintf(histo, "EBBCT errors in the number of readout crystals");
    meEBBCaloReadCryErrors_= dbe->book1D(histo, histo, 425,1.,86.);

    sprintf(histo, "EBBCT average rec energy in the single cristal");
    meEBBCaloE1vsCry_= dbe->book1D(histo, histo, 85,1.,86.);

    sprintf(histo, "EBBCT average rec energy in the 3x3 array");
    meEBBCaloE3x3vsCry_= dbe->book1D(histo, histo,85,1.,86.);

    sprintf(histo, "EBBCT energy deposition in the 3x3");
    meEBBCaloBeamCentered_= dbe->book2D(histo, histo,3,-2,2,3,-2,2);

  }
  
}

void EBBeamCaloTask::cleanup(void){

  DaqMonitorBEInterface* dbe = 0;
  
  // get hold of back-end interface
  dbe = Service<DaqMonitorBEInterface>().operator->();

  if ( dbe ) {
    dbe->setCurrentFolder("EcalBarrel/EBBeamCaloTask");
    for (int i = 0; i < cryInArray_ ; i++) {
      if ( meBBCaloPulseProf_[i] ) dbe->removeElement( meBBCaloPulseProf_[i]->getName() );
      meBBCaloPulseProf_[i] = 0;
      if ( meBBCaloPulseProfG12_[i] ) dbe->removeElement( meBBCaloPulseProfG12_[i]->getName() );
      meBBCaloPulseProfG12_[i] = 0;
      if ( meBBCaloGains_[i] ) dbe->removeElement( meBBCaloGains_[i]->getName() );
      meBBCaloGains_[i] = 0;
      if ( meBBCaloEne_[i] ) dbe->removeElement( meBBCaloEne_[i]->getName() );
      meBBCaloEne_[i] = 0;

      if ( meBBCaloPulseProfMoving_[i] ) dbe->removeElement( meBBCaloPulseProfMoving_[i]->getName() );
      meBBCaloPulseProfMoving_[i] = 0;
      if ( meBBCaloPulseProfG12Moving_[i] ) dbe->removeElement( meBBCaloPulseProfG12Moving_[i]->getName() );
      meBBCaloPulseProfG12Moving_[i] = 0;
      if ( meBBCaloGainsMoving_[i] ) dbe->removeElement( meBBCaloGainsMoving_[i]->getName() );
      meBBCaloGainsMoving_[i] = 0;
      if ( meBBCaloEneMoving_[i] ) dbe->removeElement( meBBCaloEneMoving_[i]->getName() );
      meBBCaloEneMoving_[i] = 0;
    }

//     dbe->setCurrentFolder("EcalBarrel/EBBeamCaloTask/EnergyHistos");
//     for(int u=0; u< 1701;u++){ 
//       if ( meBBCaloE3x3Cry_[u] ) dbe->removeElement( meBBCaloE3x3Cry_[u]->getName() );
//       meBBCaloE3x3Cry_[u] = 0;
//       if ( meBBCaloE1Cry_[u] ) dbe->removeElement( meBBCaloE1Cry_[u]->getName() );
//       meBBCaloE1Cry_[u] = 0;
//     }

//     dbe->setCurrentFolder("EcalBarrel/EBBeamCaloTask");
    if ( meBBCaloCryRead_ ) dbe->removeElement( meBBCaloCryRead_->getName() );
    meBBCaloCryRead_ = 0;
    if ( meBBCaloCryReadMoving_ ) dbe->removeElement( meBBCaloCryReadMoving_->getName() );
    meBBCaloCryReadMoving_ = 0;
    if ( meBBCaloAllNeededCry_ ) dbe->removeElement( meBBCaloAllNeededCry_->getName() );
    meBBCaloAllNeededCry_ = 0;
    if ( meBBNumCaloCryRead_ ) dbe->removeElement( meBBNumCaloCryRead_->getName() );
    meBBNumCaloCryRead_ = 0;
    if ( meBBCaloE3x3_ ) dbe->removeElement( meBBCaloE3x3_->getName() );
    meBBCaloE3x3_ = 0;
    if ( meBBCaloE3x3Moving_ ) dbe->removeElement( meBBCaloE3x3Moving_->getName() );
    meBBCaloE3x3Moving_ = 0;
    if ( meBBCaloCryOnBeam_ ) dbe->removeElement( meBBCaloCryOnBeam_->getName() );
    meBBCaloCryOnBeam_ = 0;
    if ( meBBCaloMaxEneCry_ ) dbe->removeElement( meBBCaloMaxEneCry_->getName() );
    meBBCaloMaxEneCry_ = 0;
    if ( TableMoving_ ) dbe->removeElement( TableMoving_->getName() );
    TableMoving_ = 0;
    if ( CrystalsDone_ ) dbe->removeElement( CrystalsDone_->getName() );
    CrystalsDone_ = 0;
    if ( CrystalInBeam_vs_Event_ ) dbe->removeElement( CrystalInBeam_vs_Event_->getName() );
    CrystalInBeam_vs_Event_ = 0;
    if( meEBBCaloReadCryErrors_ ) dbe->removeElement( meEBBCaloReadCryErrors_->getName() );
    meEBBCaloReadCryErrors_ = 0;
    if( meEBBCaloE1vsCry_ ) dbe->removeElement( meEBBCaloE1vsCry_->getName() );
    meEBBCaloE1vsCry_ = 0;
    if( meEBBCaloE3x3vsCry_ ) dbe->removeElement( meEBBCaloE3x3vsCry_->getName() );
    meEBBCaloE3x3vsCry_ = 0;
    if( meEBBCaloBeamCentered_ ) dbe->removeElement( meEBBCaloBeamCentered_->getName() );
    meEBBCaloBeamCentered_= 0;
  }

}

void EBBeamCaloTask::endJob(void){

  LogInfo("EBBeamCaloTask") << "analyzed " << ievt_ << " events";

  this->cleanup();

}

void EBBeamCaloTask::analyze(const Event& e, const EventSetup& c){
  bool enable = false;
  map<int, EcalDCCHeaderBlock> dccMap;
  
  Handle<EcalRawDataCollection> dcchs;
  Handle<EcalTBEventHeader> pEvH;
  try{
    e.getByLabel("ecalEBunpacker", dcchs);
  
    int nebc = dcchs->size();
    LogDebug("EBBeamCaloTask") << "event: " << ievt_ << " DCC headers collection size: " << nebc;
    
    for ( EcalRawDataCollection::const_iterator dcchItr = dcchs->begin(); dcchItr != dcchs->end(); ++dcchItr ) {
      
      EcalDCCHeaderBlock dcch = (*dcchItr);
      
      if ( dcch.getRunType() == EcalDCCHeaderBlock::BEAMH4
	   || dcch.getRunType() == EcalDCCHeaderBlock::BEAMH2  ) enable = true;
    }
    
  }
  catch ( std::exception& ex) {
    LogDebug("EBBeamCaloTask") << " EcalRawDataCollection not in event. Trying EcalTBEventHeader (2004 data)." << std::endl;
    
    try {
      e.getByType(pEvH);
      enable = true;
      LogDebug("EBBeamCaloTask") << " EcalTBEventHeader found, instead." << std::endl;
    } 
    catch ( std::exception& ex ) {
      LogError("EBBeamCaloTask") << "EcalTBEventHeader not present in event TOO! Returning." << std::endl;
    }
    
  }
  
  if ( ! enable ) return;
  if ( ! init_ ) this->setup();

  ievt_++; 

  int cry_in_beam = 0; 
  cry_in_beam = 704;//just for test, to be filled with info from the event
  
  bool reset_histos_stable = false;
  bool reset_histos_moving = false;
  bool tb_moving = false;//just for test, to be filled with info from the event
  bool skip_this_event = false;

 //  if(ievt_ > 500){tb_moving=true; }
//   if(ievt_ > 1000){tb_moving=false; cry_in_beam = 705;}
//   if(ievt_ > 2000){tb_moving=true; }
//   if(ievt_ > 2500){tb_moving=false; cry_in_beam = 706;}
//   if(ievt_ > 3500){tb_moving=true; }

  if(ievt_ > 3300){tb_moving=true; }
  if(ievt_ > 6100){tb_moving=false; cry_in_beam = 705;}
  if(ievt_ == 6201){tb_moving=true; }
  if(ievt_ > 9000){tb_moving=true; }
  if(ievt_ == 11021){tb_moving=false; }
  if(ievt_ > 12100){tb_moving=false; cry_in_beam = 706;}
  if(ievt_ > 15320){tb_moving=true; }
  if(ievt_ > 15500){tb_moving=false; cry_in_beam = 707;}
  //if(tb_moving) {CrystalInBeam_vs_Event_->Fill(ievt_,-100.);}
  //else{CrystalInBeam_vs_Event_->Fill(ievt_,float(cry_in_beam));}
  
  if(tb_moving){
    TableMoving_->Fill(1);
    if( PreviousTableStatus_[0] == 0 &&  PreviousTableStatus_[1] == 1 && lastStableStatus_ == 0){
      reset_histos_moving=true;
      wasFakeChange_ = false;
      // ! Warning! This works in the assumption that the crystal in beam stay the same
      // while the tb is moving and is set to the new one only when the table
      // reaches the new position
      lastStableStatus_ = 1;
      CrystalsDone_->Fill(float(cry_in_beam)-0.5);
    }
    else if(PreviousTableStatus_[1] == 0) {
      skip_this_event=true;
      changed_tb_status_ = true;
      wasFakeChange_ = true;
    }
    // just skip the first event when the table change status
    PreviousTableStatus_[0] = PreviousTableStatus_[1];
    PreviousTableStatus_[1] = 1;
  }
  else {
    TableMoving_->Fill(0);
    if( PreviousTableStatus_[0] == 1 &&  PreviousTableStatus_[1] == 0 && lastStableStatus_ == 1){
      reset_histos_stable=true;
      wasFakeChange_ = false;
      lastStableStatus_ = 0;
    }
    else if(PreviousTableStatus_[1] == 1) {
      skip_this_event=true;
      changed_tb_status_ = true;
      //      evt_after_change_ =0;
      wasFakeChange_ = true;
    }
    // just skip the first event when the table change status
    PreviousTableStatus_[0]=PreviousTableStatus_[1];
    PreviousTableStatus_[1]=0;
  }

  if (! changed_tb_status_){

    if( !tb_moving ) {CrystalInBeam_vs_Event_->Fill(ievt_,float(cry_in_beam));}
  }
  else{
    if(tb_moving){cib_[evt_after_change_]=-100;}
    else {cib_[evt_after_change_]=cry_in_beam;}
    
    if(evt_after_change_ >= 9){
      evt_after_change_ =0;
      if(wasFakeChange_){
	for(int u=0; u<10; u++){
	  CrystalInBeam_vs_Event_->Fill(ievt_-9+u , cib_[u]);
	}
      }
      changed_tb_status_=false;//for a real change just skip the first 10 events after change
    }
    else{evt_after_change_ ++;}
  }
  
  if(skip_this_event){
    LogInfo("EBBeamCaloTask") << "event " << ievt_ << " : skipping this event!! ";
    return;}
 
  if(reset_histos_moving){
    LogInfo("EBBeamCaloTask") << "event " << ievt_ << " resetting histos for stable table!! ";
    
    table_step_++;
    

    //here the follwowing histos should be reset
    for (int u=0;u<cryInArray_;u++){
      EBMUtilsTasks::resetHisto( meBBCaloPulseProfMoving_[u] );
      EBMUtilsTasks::resetHisto( meBBCaloPulseProfG12Moving_[u] );
      EBMUtilsTasks::resetHisto( meBBCaloGainsMoving_[u] );
      EBMUtilsTasks::resetHisto( meBBCaloEneMoving_[u] );
    }
    EBMUtilsTasks::resetHisto( meBBCaloCryReadMoving_ );
    EBMUtilsTasks::resetHisto( meBBCaloE3x3Moving_ );

    
  }
  
  
  if(reset_histos_stable){
    LogInfo("EBBeamCaloTask") << "event " << ievt_ << " resetting histos for moving table!! ";
    meEBBCaloE1vsCry_->setBinContent(crystal_step_ , meBBCaloEne_[4]->getMean() );
    meEBBCaloE1vsCry_->setBinError(crystal_step_ , meBBCaloEne_[4]->getRMS() );
    meEBBCaloE3x3vsCry_->setBinContent(crystal_step_ , meBBCaloE3x3_->getMean() );
    meEBBCaloE3x3vsCry_->setBinError(crystal_step_ , meBBCaloE3x3_->getRMS() );
    
    crystal_step_++;

    //here the follwowing histos should be reset
    for (int u=0;u<cryInArray_;u++){
      EBMUtilsTasks::resetHisto( meBBCaloPulseProf_[u] );
      EBMUtilsTasks::resetHisto( meBBCaloPulseProfG12_[u] );
      EBMUtilsTasks::resetHisto( meBBCaloGains_[u] );
      EBMUtilsTasks::resetHisto( meBBCaloEne_[u] );
    }
    EBMUtilsTasks::resetHisto( meBBCaloCryRead_ );
    EBMUtilsTasks::resetHisto( meBBCaloE3x3_ );
    EBMUtilsTasks::resetHisto( meEBBCaloBeamCentered_ );	
  }

  int eta_c = ( cry_in_beam-1)/20 ;
  int phi_c = ( cry_in_beam-1)%20 ;
  
  float xie = eta_c + 0.5;
  float xip = phi_c + 0.5;
  if (!tb_moving) {meBBCaloCryOnBeam_->Fill(xie,xip);}

  Handle<EBDigiCollection> digis;
  //e.getByLabel("ecalEBunpacker", digis);
  e.getByLabel(digiProducer_, digis);
  int nebd = digis->size();
  LogDebug("EBBeamCaloTask") << "event " << ievt_ << " digi collection size " << nebd;

  meBBNumCaloCryRead_->Fill(nebd);
  
  //matrix 7x7 around cry in beam
  int cry_to_beRead[49]; //0 or -1 for non existing crystals (eg 1702)
  for(int u=0;u<49;u++){cry_to_beRead[u]=0;}
  // chech that all the crystals in the 7x7 exist 
  for(int de=-3; de<4; de++){
    for(int dp=-3; dp<4; dp++){
      int cry_num = (phi_c+dp) + 20*(eta_c+de) +1;
      int u = de -7*dp + 24;
      if(cry_num<1 || cry_num> 1701){cry_to_beRead[u]=-1;}
    }
  }
  
  for ( EBDigiCollection::const_iterator digiItr = digis->begin(); digiItr != digis->end(); ++digiItr ) {

    EBDataFrame dataframe = (*digiItr);
    EBDetId id = dataframe.id();

    //int ism = id.ism();
    // FIX this if can not work on the 2004 data since they do not fill in the  EcalDCCHeaderBlock
    //if ( dccMap[ism-1].getRunType() != EcalDCCHeaderBlock::BEAMH4 ) continue;//FIX ME add the autoscan runtype

    int ic = id.ic();
    int ie = (ic-1)/20;
    int ip = (ic-1)%20;

    int deta_c= ie - eta_c;
    int dphi_c= ip - phi_c;
    if (! tb_moving){meBBCaloCryRead_->Fill(deta_c, dphi_c);}
    else {meBBCaloCryReadMoving_->Fill(deta_c, dphi_c);}

    if(abs(deta_c) >3 || abs(dphi_c) >3){continue;}
    int i_toBeRead = deta_c -7*dphi_c + 24;
    if( i_toBeRead > -1 &&  i_toBeRead <49){ 
      cry_to_beRead[i_toBeRead]++;
      //if( (ievt_ == 4000 || ievt_ == 13000 || ievt_ == 13002 ) &&   i_toBeRead == 5){ cry_to_beRead[i_toBeRead] -=1;}
    }

    if(abs(deta_c) >1 || abs(dphi_c) >1){continue;}
    int i_in_array = deta_c -3*dphi_c + 4;
    

    //LogDebug("EBBeamCaloTask") << " det id = " << id;
    //LogDebug("EBBeamCaloTask") << " sm, eta, phi " << ism << " " << ie << " " << ip;
    //LogDebug("EBBeamCaloTask") << " deta, dphi, i_in_array, i_toBeRead " << deta_c  << " " <<  dphi_c << " " <<i_in_array<<" "<<i_toBeRead;

    if( i_in_array < 0 || i_in_array > 8 ){continue;}

    for (int i = 0; i < 10; i++) {
      EcalMGPASample sample = dataframe.sample(i);
      int adc = sample.adc();
      int gainid = sample.gainId();
      //if( (ievt_ == 15400 || ievt_ == 15600 || ievt_ == 15700 ) &&   i_in_array == 4 && i == 4){ gainid =2;}
      //if( (ievt_ == 15400 || ievt_ == 15600 || ievt_ == 15700 ) &&   i_in_array == 6 && i == 6){ gainid =3;}

      if ( gainid == 1 ){// gain 12
	if(! tb_moving){
	  meBBCaloPulseProfG12_[i_in_array]->Fill(i,float(adc));
	  meBBCaloPulseProf_[i_in_array]->Fill(i,float(adc));
	  meBBCaloGains_[i_in_array]->Fill(12);
	}
	else{
	  meBBCaloPulseProfG12Moving_[i_in_array]->Fill(i,float(adc));
	  meBBCaloPulseProfMoving_[i_in_array]->Fill(i,float(adc));
	  meBBCaloGainsMoving_[i_in_array]->Fill(12);
	}
      }
      else if ( gainid == 2 ){// gain 6 
	float val = (float(adc)-defaultPede_)*2 + defaultPede_;
	if(! tb_moving){
	  meBBCaloPulseProf_[i_in_array]->Fill(i,val);
	  meBBCaloGains_[i_in_array]->Fill(6);
	}
	else{
	  meBBCaloPulseProfMoving_[i_in_array]->Fill(i,val);
	  meBBCaloGainsMoving_[i_in_array]->Fill(6);
	}
      }
      else if ( gainid == 3 ){// gain 1 
	float val = (float(adc)-defaultPede_)*12 + defaultPede_;
	if(! tb_moving){
	meBBCaloPulseProf_[i_in_array]->Fill(i,val);
	meBBCaloGains_[i_in_array]->Fill(1);
	}
	else{
	meBBCaloPulseProfMoving_[i_in_array]->Fill(i,val);
	meBBCaloGainsMoving_[i_in_array]->Fill(1);
	}
      }
    }// end of loop over samples
  }// end of loop over digis
  
  //now  if everything was correct cry_to_beRead should be filled with 1 or -1 but not 0
  bool all_cry_readout = true;
  for(int u =0; u<49;u++){if(cry_to_beRead[u]==0){all_cry_readout = false;}}
  // if( ievt_ == 4000 || ievt_ == 13000 || ievt_ == 13002 ) {all_cry_readout = false;}
  if(all_cry_readout){ meBBCaloAllNeededCry_->Fill(1.5);}//bin3
  else {
    meBBCaloAllNeededCry_->Fill(-0.5);
    if( tb_moving ) {meEBBCaloReadCryErrors_->Fill( crystal_step_+0.5 );}
    else {meEBBCaloReadCryErrors_->Fill( crystal_step_ );}
  }//bin1

  //the part involving rechits

  Handle<EcalUncalibratedRecHitCollection> hits;
  e.getByLabel("ecalUncalibHitMaker", "EcalUncalibRecHitsEB", hits);
  int neh = hits->size();
  LogDebug("EBBeamCaloTask") << "event " << ievt_ << " hits collection size " << neh;
  float ene3x3=0;
  float maxEne = 0; 
  int ieM =-1, ipM = -1;//for the crystal with maximum energy deposition
  float cryInBeamEne =0;
  for ( EcalUncalibratedRecHitCollection::const_iterator hitItr = hits->begin(); hitItr != hits->end(); ++hitItr ) {

    EcalUncalibratedRecHit hit = (*hitItr);
    EBDetId id = hit.id();
    int ism = id.ism();
    // FIX this if can not work on the 2004 data since they do not fill in the  EcalDCCHeaderBlock
    //    if ( dccMap[ism-1].getRunType() != EcalDCCHeaderBlock::BEAMH4 ) continue;//FIX ME add the autoscan runtype

    int ic = id.ic();
    int ie = (ic-1)/20;
    int ip = (ic-1)%20;

    int deta_c= ie - eta_c;
    int dphi_c= ip - phi_c;

    
    int i_in_array = deta_c -3*dphi_c + 4;
    LogDebug("EBBeamCaloTask") << " rechits det id = " << id;
    LogDebug("EBBeamCaloTask") << " rechits sm, eta, phi " << ism << " " << ie << " " << ip;
    LogDebug("EBBeamCaloTask") << " rechits deta, dphi, i_in_array" << deta_c  << " " <<  dphi_c << " " <<i_in_array;
  
    float R_ene = hit.amplitude();
    if(R_ene > maxEne){
      maxEne=R_ene;
      ieM =ie; ipM = ip;
    }
    if(abs(deta_c) >1 || abs(dphi_c) >1){continue;}
    meEBBCaloBeamCentered_->Fill(deta_c,dphi_c,R_ene);
    
    if( i_in_array < 0 || i_in_array > 8 ){continue;}
    
    //LogDebug("EBBeamCaloTask") <<"In the array, cry: "<<ic<<" rec ene: "<<R_ene;

    if(i_in_array == 4){cryInBeamEne = R_ene;}
    if(! tb_moving){meBBCaloEne_[i_in_array]->Fill(R_ene);}
    else{meBBCaloEneMoving_[i_in_array]->Fill(R_ene);}
    ene3x3 += R_ene;

  }//end of loop over rechits

  if (!tb_moving){
    meBBCaloE3x3_->Fill(ene3x3);
//     if( cry_in_beam > 0 && cry_in_beam < 1701){
//       meBBCaloE3x3Cry_[cry_in_beam]->Fill(ene3x3);
//       meBBCaloE1Cry_[cry_in_beam]->Fill(cryInBeamEne);
//     }
    meBBCaloMaxEneCry_->Fill(ieM,ipM);
  }
  else{meBBCaloE3x3Moving_->Fill(ene3x3);}
  /////////////
}

