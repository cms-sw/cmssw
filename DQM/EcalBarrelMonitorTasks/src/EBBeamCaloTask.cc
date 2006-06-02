
#include <DQM/EcalBarrelMonitorTasks/interface/EBBeamCaloTask.h>
#include <DQM/EcalBarrelMonitorClient/interface/EBMUtilsClient.h>
EBBeamCaloTask::EBBeamCaloTask(const ParameterSet& ps){
  LogDebug("EBBeamCaloTask") << " 1 construct  ";
  init_ = false;
  //digiProducer_ = ps.getUntrackedParameter<string>("digiProducer", "ecalEBunpacker");
  digiProducer_ = ps.getParameter<string>("digiProducer");
  //DCCHeaderProducer_= ps.getUntrackedParameter<string>("dccHeaderProducer", "ecalEBunpacker");
  for (int i = 0; i < cryInArray_ ; i++) {
    meBBCaloPulseProf_[i]=0;
    meBBCaloPulseProfG12_[i]=0;
    meBBCaloGains_[i]=0;
    meBBCaloEne_[i]=0;
  }
  meBBCaloCryRead_ = 0;
  meBBNumCaloCryRead_ = 0;
  meBBCaloAllNeededCry_ = 0;
  meBBCaloE3x3_ = 0;
  meBBCaloCryOnBeam_ = 0;
  meBBCaloMaxEneCry_ = 0;
  TableMoving_ = 0;
  LogDebug("EBBeamCaloTask") << " 2 construct  ";
}

EBBeamCaloTask::~EBBeamCaloTask(){

}

void EBBeamCaloTask::beginJob(const EventSetup& c){

  ievt_ = 0;

}

void EBBeamCaloTask::setup(void){
  init_ = true;

  Char_t histo[20];
  PreviousTableStatus_[0]=0;//let's start with stable...
  PreviousTableStatus_[1]=0;//let's start with stable...

  DaqMonitorBEInterface* dbe = 0;

  // get hold of back-end interface
  dbe = Service<DaqMonitorBEInterface>().operator->();

  if ( dbe ) {
    dbe->setCurrentFolder("EcalBarrel/EBBeamCaloTask");
    
    for (int i = 0; i < cryInArray_ ; i++) {
      sprintf(histo, "EBBCT pulse profile cry: %01d", i+1);
      //considering the gain the range is 4096*12 ~ 50000
      meBBCaloPulseProf_[i] = dbe->bookProfile(histo, histo, 10,0.,10.,50000,0.,50000.);

      sprintf(histo, "EBBCT pulse profile in G12 cry: %01d", i+1);
      meBBCaloPulseProfG12_[i] = dbe->bookProfile(histo, histo, 10,0.,10.,4096,0.,4096.);

      sprintf(histo, "EBBCT found gains cry: %01d", i+1);
      meBBCaloGains_[i] =  dbe->book1D(histo,histo,14,0.,14.);
      // g1-> bin 2, g6-> bin 7, g12-> bin 13

      sprintf(histo, "EBBCT rec energy cry: %01d", i+1);
      meBBCaloEne_[i] =  dbe->book1D(histo,histo,9000,0.,9000.);
      //9000 ADC in G12 equivalent is about 330 GeV
    }

    sprintf(histo, "EBBCT readout crystals");
    meBBCaloCryRead_  =  dbe->book2D(histo,histo,9,-4.,5.,9,-4.,5.);
    //matrix of readout crystal around cry in beam

    sprintf(histo, "EBBCT all needed crystals readout");
    meBBCaloAllNeededCry_ = dbe->book1D(histo,histo,3,-1.,2.);
    // not all needed cry are readout-> bin 1, all needed cry are readout-> bin 3
    
    sprintf(histo, "EBBCT number of readout crystals");
    meBBNumCaloCryRead_ = dbe->book1D(histo,histo,1700,1.,1701.);

    sprintf(histo, "EBBCT rec Ene sum 3x3");
    meBBCaloE3x3_ = dbe->book1D(histo,histo,9000,0.,9000.);
    //9000 ADC in G12 equivalent is about 330 GeV

    sprintf(histo, "EBBCT crystal on beam");
    meBBCaloCryOnBeam_ = dbe->book2D(histo, histo, 85, 0., 85., 20, 0., 20.);

    sprintf(histo, "EBBCT crystal with maximum rec energy");
    meBBCaloMaxEneCry_ = dbe->book2D(histo, histo, 85, 0., 85., 20, 0., 20.);
    
    sprintf(histo, "EBBCT table is moving");
    TableMoving_ = dbe->book1D(histo,histo,2,0.,1.1);
    //table is moving-> bin 1, table is not moving-> bin 2
   
  }
    
}

void EBBeamCaloTask::endJob(){

  LogInfo("EBBeamCaloTask") << "analyzed " << ievt_ << " events";
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
    
  int cry_in_beam = 0; 
  cry_in_beam = 704;//just for test, to be filled with info from the event
  
  bool reset_histos = false;
  bool tb_moving = false;//just for test, to be filled with info from the event
  bool skip_this_event = false;

  //  if(ievt_ > 500){tb_moving=true; }
  //if(ievt_ > 1000){tb_moving=false; cry_in_beam = 705;}

  if(tb_moving){
    TableMoving_->Fill(1);
    if( PreviousTableStatus_[0] == 0 &&  PreviousTableStatus_[1] == 1){reset_histos=true;}
    else if(PreviousTableStatus_[1] == 0) {skip_this_event=true;}
    // just skip the first event when the table change status
    PreviousTableStatus_[0]=PreviousTableStatus_[1];
    PreviousTableStatus_[1]=1;
  }
  else {
    TableMoving_->Fill(0);
    if( PreviousTableStatus_[0] == 1 &&  PreviousTableStatus_[1] == 0){reset_histos=true;}
    else if(PreviousTableStatus_[1] == 1) {skip_this_event=true;}
    PreviousTableStatus_[0]=PreviousTableStatus_[1];
    PreviousTableStatus_[1]=0;
  }
 
  ievt_++; 
  if(skip_this_event){
    LogInfo("EBBeamCaloTask") << "event " << ievt_ << " : skipping this event!! ";
    //cout<< "event " << ievt_ << " : skipping this event!! "<<endl;
    return;}
 
  if(reset_histos){
        LogDebug("EBBeamCaloTask") << "event " << ievt_ << " resetting histos!! ";
	//cout << "event " << ievt_ << " resetting histos!! ";
    //here the follwowing histos should be reset
	for (int u=0;u<cryInArray_;u++){
	  EBMUtilsClient::resetHisto( meBBCaloPulseProf_[u] );
	  EBMUtilsClient::resetHisto( meBBCaloPulseProfG12_[u] );
	   EBMUtilsClient::resetHisto( meBBCaloGains_[u] );
	   EBMUtilsClient::resetHisto( meBBCaloEne_[u] );
	}
	 EBMUtilsClient::resetHisto( meBBCaloCryRead_ );
    // meBBCaloAllNeededCry_;
    // ?? boh meBBNumCaloCryRead_;
      EBMUtilsClient::resetHisto( meBBCaloE3x3_ );

  }

 

  int eta_c = ( cry_in_beam-1)/20 ;
  int phi_c = ( cry_in_beam-1)%20 ;
  
//   cryIn3x3_[0] = (phi_c -1) + 20*(eta_c -1) +1;
//   cryIn3x3_[1] = (phi_c -1) + 20*(eta_c)    +1;
//   cryIn3x3_[2] = (phi_c -1) + 20*(eta_c +1) +1;
//   cryIn3x3_[3] = (phi_c)    + 20*(eta_c -1) +1;
//   cryIn3x3_[4] = (phi_c)    + 20*(eta_c)    +1;
//   cryIn3x3_[5] = (phi_c)    + 20*(eta_c +1) +1;
//   cryIn3x3_[6] = (phi_c+1)  + 20*(eta_c -1) +1;
//   cryIn3x3_[7] = (phi_c+1)  + 20*(eta_c)    +1;
//   cryIn3x3_[8] = (phi_c+1)  + 20*(eta_c +1) +1;

  float xie = eta_c + 0.5;
  float xip = phi_c + 0.5;
  meBBCaloCryOnBeam_->Fill(xie,xip);

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
      int u = de -7*dp + 24;// FIX ME to be check via a cout
      //std::cout<<"de, dp, cry, u: "<<de <<" "<<	dp<<" "<<cry_num <<" "<< u;// <<std::endl;
      //if(u<0 || u > 48) {std::cout<<"ERROR de, dp, cry, u"<<de <<" "<<	dp<<" "<<cry_num <<" "<< u<<std::endl;}
      if(cry_num<1 || cry_num> 1701){cry_to_beRead[u]=-1;}
      //std::cout<<"  to be read: "<<cry_to_beRead[u]<<endl;
    }
  }
  
  for ( EBDigiCollection::const_iterator digiItr = digis->begin(); digiItr != digis->end(); ++digiItr ) {

    EBDataFrame dataframe = (*digiItr);
    EBDetId id = dataframe.id();

    int ism = id.ism();
    // FIX this if can not work on the 2004 data since they do not fill in the  EcalDCCHeaderBlock
    //if ( dccMap[ism-1].getRunType() != EcalDCCHeaderBlock::BEAMH4 ) continue;//FIX ME add the autoscan runtype

    int ic = id.ic();
    int ie = (ic-1)/20;
    int ip = (ic-1)%20;

    int deta_c= ie - eta_c;
    int dphi_c= ip - phi_c;
    meBBCaloCryRead_->Fill(deta_c, dphi_c);
    
    if(abs(deta_c) >3 || abs(deta_c) >3){continue;}
    int i_toBeRead = deta_c -7*dphi_c + 24;
    if( i_toBeRead > -1 &&  i_toBeRead <49){ cry_to_beRead[i_toBeRead]++;}

    if(abs(deta_c) >1 || abs(deta_c) >1){continue;}
    int i_in_array = deta_c -3*dphi_c + 4;
    

    //LogDebug("EBBeamCaloTask") << " det id = " << id;
    //LogDebug("EBBeamCaloTask") << " sm, eta, phi " << ism << " " << ie << " " << ip;
    //LogDebug("EBBeamCaloTask") << " deta, dphi, i_in_array, i_toBeRead " << deta_c  << " " <<  dphi_c << " " <<i_in_array<<" "<<i_toBeRead;

    //cout << " det id = " << id<<endl;
    //cout << " sm, eta, phi " << ism << " " << ie << " " << ip<<endl;
    //cout << " deta, dphi, i_in_array, i_toBeRead " << deta_c  << " " <<  dphi_c << " " <<i_in_array<<" "<<i_toBeRead<<endl;

    if( i_in_array < 0 || i_in_array > 8 ){continue;}

    //cout << " det id = " << id<<endl;
    //cout << " sm, eta, phi " << ism << " " << ie << " " << ip<<endl;
    //cout << " deta, dphi, i_in_array, i_toBeRead " << deta_c  << " " <<  dphi_c << " " <<i_in_array<<" "<<i_toBeRead<<endl;
    //cout<<"##########################################################"<<endl;
    for (int i = 0; i < 10; i++) {
      EcalMGPASample sample = dataframe.sample(i);
      int adc = sample.adc();
      if ( sample.gainId() == 1 ){// gain 12 
	meBBCaloPulseProfG12_[i_in_array]->Fill(i,float(adc));
	meBBCaloPulseProf_[i_in_array]->Fill(i,float(adc));
	meBBCaloGains_[i_in_array]->Fill(12);
      }
      else if ( sample.gainId() == 2 ){// gain 6 
	float val = (float(adc)-defaultPede_)*2 + defaultPede_;
	meBBCaloPulseProf_[i_in_array]->Fill(i,val);
	meBBCaloGains_[i_in_array]->Fill(6);
      }
      else if ( sample.gainId() == 3 ){// gain 1 
	float val = (float(adc)-defaultPede_)*12 + defaultPede_;
	meBBCaloPulseProf_[i_in_array]->Fill(i,val);
	meBBCaloGains_[i_in_array]->Fill(1);
      }
    }// end of loop over samples
  }// end of loop over digis
  
  //now  if everything was correct cry_to_beRead should be filled with 1 or -1 but not 0
  bool all_cry_readout = true;
  for(int u =0; u<49;u++){if(cry_to_beRead[u]==0){all_cry_readout = false;}}
  if(all_cry_readout){meBBCaloAllNeededCry_->Fill(1.5);}//bin3
  else {meBBCaloAllNeededCry_->Fill(-0.5);}//bin1

  //the part involving rechits

  Handle<EcalUncalibratedRecHitCollection> hits;
  e.getByLabel("ecalUncalibHitMaker", "EcalUncalibRecHitsEB", hits);
  
  int neh = hits->size();
  LogDebug("EBBeamCaloTask") << "event " << ievt_ << " hits collection size " << neh;
  float ene3x3=0;
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
  
    if(abs(deta_c) >1 || abs(deta_c) >1){continue;}
   

    if( i_in_array < 0 || i_in_array > 8 ){continue;}
    float R_ene = hit.amplitude();
    //LogDebug("EBBeamCaloTask") <<"In the array, cry: "<<ic<<" rec ene: "<<R_ene;
    //cout <<"In the array, cry: "<<ic<<" rec ene: "<<R_ene<<endl;
    //cout<<"@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"<<endl;

    meBBCaloEne_[i_in_array]->Fill(R_ene);
    ene3x3 += R_ene;

  }//end of loop over rechits

  meBBCaloE3x3_->Fill(ene3x3);


  /////////////
}

