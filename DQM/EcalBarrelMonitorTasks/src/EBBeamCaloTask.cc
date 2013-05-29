/*
 * \file EBBeamCaloTask.cc
 *
 * $Date: 2011/08/23 00:25:30 $
 * $Revision: 1.80.4.1 $
 * \author A. Ghezzi
 *
 */

#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DQMServices/Core/interface/MonitorElement.h"

#include "DQMServices/Core/interface/DQMStore.h"

#include "DataFormats/EcalRawData/interface/EcalRawDataCollections.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDigi/interface/EBDataFrame.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "DataFormats/EcalRecHit/interface/EcalUncalibratedRecHit.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"

#include "TBDataFormats/EcalTBObjects/interface/EcalTBCollections.h"

#include "DQM/EcalCommon/interface/Numbers.h"

#include "DQM/EcalBarrelMonitorTasks/interface/EBBeamCaloTask.h"

EBBeamCaloTask::EBBeamCaloTask(const edm::ParameterSet& ps){

  init_ = false;

  dqmStore_ = edm::Service<DQMStore>().operator->();

  prefixME_ = ps.getUntrackedParameter<std::string>("prefixME", "");

  enableCleanup_ = ps.getUntrackedParameter<bool>("enableCleanup", false);

  mergeRuns_ = ps.getUntrackedParameter<bool>("mergeRuns", false);

  EcalTBEventHeader_ = ps.getParameter<edm::InputTag>("EcalTBEventHeader");
  EcalRawDataCollection_ = ps.getParameter<edm::InputTag>("EcalRawDataCollection");
  EBDigiCollection_ = ps.getParameter<edm::InputTag>("EBDigiCollection");
  EcalUncalibratedRecHitCollection_ = ps.getParameter<edm::InputTag>("EcalUncalibratedRecHitCollection");

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
  meBBCaloE3x3Moving_ = 0;

  meBBCaloCryOnBeam_ = 0;
  meBBCaloMaxEneCry_ = 0;
  TableMoving_ = 0;

  CrystalsDone_ = 0;
  CrystalInBeam_vs_Event_ = 0;
  meEBBCaloReadCryErrors_ = 0;
  meEBBCaloE1vsCry_ = 0;
  meEBBCaloE3x3vsCry_ = 0;
  meEBBCaloEntriesVsCry_ = 0;
  meEBBCaloBeamCentered_ = 0;

  meEBBCaloE1MaxCry_ = 0;

  meEBBCaloDesync_ = 0;

}

EBBeamCaloTask::~EBBeamCaloTask(){

}

void EBBeamCaloTask::beginJob(void){

  ievt_ = 0;

  profileArranged_ = false;

  if ( dqmStore_ ) {
    dqmStore_->setCurrentFolder(prefixME_ + "/EBBeamCaloTask");
    dqmStore_->rmdir(prefixME_ + "/EBBeamCaloTask");
  }

}

void EBBeamCaloTask::beginRun(const edm::Run& r, const edm::EventSetup& c) {

  Numbers::initGeometry(c, false);

  if ( ! mergeRuns_ ) this->reset();

}

void EBBeamCaloTask::endRun(const edm::Run& r, const edm::EventSetup& c) {

}

void EBBeamCaloTask::reset(void) {

    for (int i = 0; i < cryInArray_ ; i++) {
      if ( meBBCaloPulseProf_[i] ) meBBCaloPulseProf_[i]->Reset();
      if ( meBBCaloPulseProfG12_[i] ) meBBCaloPulseProfG12_[i]->Reset();
      if ( meBBCaloGains_[i] ) meBBCaloGains_[i]->Reset();
      if ( meBBCaloEne_[i] ) meBBCaloEne_[i]->Reset();
    }

    if ( meBBCaloCryRead_ ) meBBCaloCryRead_->Reset();
    if ( meBBCaloAllNeededCry_ ) meBBCaloAllNeededCry_->Reset();
    if ( meBBNumCaloCryRead_ ) meBBNumCaloCryRead_->Reset();
    if ( meBBCaloE3x3_ ) meBBCaloE3x3_->Reset();
    if ( meBBCaloE3x3Moving_ ) meBBCaloE3x3Moving_->Reset();
    if ( meBBCaloCryOnBeam_ ) meBBCaloCryOnBeam_->Reset();
    if ( meBBCaloMaxEneCry_ ) meBBCaloMaxEneCry_->Reset();
    if ( TableMoving_ ) TableMoving_->Reset();
    if ( CrystalsDone_ ) CrystalsDone_->Reset();
    if ( CrystalInBeam_vs_Event_ ) CrystalInBeam_vs_Event_->Reset();
    if( meEBBCaloReadCryErrors_ ) meEBBCaloReadCryErrors_->Reset();
    if( meEBBCaloE1vsCry_ ) meEBBCaloE1vsCry_->Reset();
    if( meEBBCaloE3x3vsCry_ ) meEBBCaloE3x3vsCry_->Reset();
    if( meEBBCaloEntriesVsCry_ )  meEBBCaloEntriesVsCry_->Reset();
    if( meEBBCaloBeamCentered_ ) meEBBCaloBeamCentered_->Reset();
    if( meEBBCaloE1MaxCry_ ) meEBBCaloE1MaxCry_->Reset();
    if( meEBBCaloDesync_ ) meEBBCaloDesync_->Reset();

}

void EBBeamCaloTask::setup(void){

  init_ = true;
  profileArranged_= false;

  PreviousTableStatus_[0]=0;//let's start with stable...
  PreviousTableStatus_[1]=0;//let's start with stable...

  PreviousCrystalinBeam_[0] = 0;
  PreviousCrystalinBeam_[1] = 0;
  PreviousCrystalinBeam_[2] = -1;
  // PreviousCrystalinBeam_[2] = -1 is needed to have a correct step vs cry matching
  lastStableStatus_=0;
  for(int u=0;u<10;u++){cib_[u]=0;}
  changed_tb_status_= false;
  changed_cry_in_beam_ = false;
  evt_after_change_ =0;
  wasFakeChange_= false;
  table_step_=1;
  crystal_step_=1;
  event_last_reset_ = 0;
  last_cry_in_beam_ = 0;
  previous_cry_in_beam_ = 1;

  if ( dqmStore_ ) {
    dqmStore_->setCurrentFolder(prefixME_ + "/EBBeamCaloTask");

    std::stringstream ss;
    std::string name;

    for (int i = 0; i < cryInArray_ ; i++) {
      ss << std::setw(1) << std::setfill('0') << i+1;

      name = "EBBCT pulse profile cry ";
      //considering the gain the range is 4096*12 ~ 50000
      meBBCaloPulseProf_[i] = dqmStore_->bookProfile(name + ss.str(), name + ss.str(), 10,0.,10.,50000,0.,50000.,"s");

      name = "EBBCT pulse profile in G12 cry ";
      meBBCaloPulseProfG12_[i] = dqmStore_->bookProfile(name + ss.str(), name + ss.str(), 10,0.,10.,4096,0.,4096.,"s");
      meBBCaloPulseProfG12_[i]->setAxisTitle("#sample", 1);
      meBBCaloPulseProfG12_[i]->setAxisTitle("ADC", 2);

      name = "EBBCT found gains cry ";
      meBBCaloGains_[i] =  dqmStore_->book1D(name + ss.str(), name + ss.str(),14,0.,14.);
      meBBCaloGains_[i]->setAxisTitle("gain", 1);
      // g1-> bin 2, g6-> bin 7, g12-> bin 13

      name = "EBBCT rec energy cry ";
      meBBCaloEne_[i] =  dqmStore_->book1D(name + ss.str(), name + ss.str(),500,0.,9000.);
      meBBCaloEne_[i]->setAxisTitle("rec ene (ADC)", 1);
      //9000 ADC in G12 equivalent is about 330 GeV

    }

    name = "EBBCT readout crystals";
    meBBCaloCryRead_ =  dqmStore_->book2D(name, name,9,-4.,5.,9,-4.,5.);
    //matrix of readout crystal around cry in beam

    name = "EBBCT all needed crystals readout";
    meBBCaloAllNeededCry_ = dqmStore_->book1D(name, name,3,-1.,2.);
    // not all needed cry are readout-> bin 1, all needed cry are readout-> bin 3

    name = "EBBCT readout crystals number";
    meBBNumCaloCryRead_ = dqmStore_->book1D(name, name,1701,0.,1701.);
    meBBNumCaloCryRead_->setAxisTitle("number of read crystals", 1);

    name = "EBBCT rec Ene sum 3x3";
    meBBCaloE3x3_ = dqmStore_->book1D(name, name,500,0.,9000.);
    meBBCaloE3x3_->setAxisTitle("rec ene (ADC)", 1);
    //9000 ADC in G12 equivalent is about 330 GeV

    name = "EBBCT rec Ene sum 3x3 table moving";
    meBBCaloE3x3Moving_ = dqmStore_->book1D(name, name,500,0.,9000.);
    //9000 ADC in G12 equivalent is about 330 GeV

    name = "EBBCT crystal on beam";
    meBBCaloCryOnBeam_ = dqmStore_->book2D(name, name, 85, 0., 85., 20, 0., 20.);

    name = "EBBCT crystal with maximum rec energy";
    meBBCaloMaxEneCry_ = dqmStore_->book2D(name, name, 85, 0., 85., 20, 0., 20.);

    name = "EBBCT table is moving";
    TableMoving_ = dqmStore_->book1D(name,name,2,0.,1.1);
    TableMoving_->setAxisTitle("table status (0=stable, 1=moving)", 1);
    //table is moving-> bin 2, table is not moving-> bin 1

    name = "EBBCT crystals done";
    CrystalsDone_ = dqmStore_->book1D(name,name,1700,1.,1701.);
    CrystalsDone_->setAxisTitle("crystal", 1);
    CrystalsDone_->setAxisTitle("step in the scan", 2);
    //for a crystal done the corresponing bin is filled with the step in the
    //autoscan pertainig to the given crystales

    name = "EBBCT crystal in beam vs event";
    CrystalInBeam_vs_Event_ = dqmStore_->bookProfile(name, name, 20000,0.,400000.,1802,-101.,1701.,"s");
    CrystalInBeam_vs_Event_->setAxisTitle("event", 1);
    CrystalInBeam_vs_Event_->setAxisTitle("crystal in beam", 2);
    // 1 bin each 20 events
    // when table is moving for just one events fill with -100

    name = "EBBCT readout crystals errors";
    meEBBCaloReadCryErrors_ = dqmStore_->book1D(name, name, 425,1.,86.);
    meEBBCaloReadCryErrors_->setAxisTitle("step in the scan", 1);

    name = "EBBCT average rec energy in the single crystal";
    meEBBCaloE1vsCry_ = dqmStore_->bookProfile(name, name, 1700,1.,1701.,500,0.,9000.,"s");
    meEBBCaloE1vsCry_->setAxisTitle("crystal", 1);
    meEBBCaloE1vsCry_->setAxisTitle("rec energy (ADC)", 2);

    name = "EBBCT average rec energy in the 3x3 array";
    meEBBCaloE3x3vsCry_ = dqmStore_->bookProfile(name, name, 1700,1.,1701.,500,0.,9000.,"s");
    meEBBCaloE3x3vsCry_->setAxisTitle("crystal", 1);
    meEBBCaloE3x3vsCry_->setAxisTitle("rec energy (ADC)", 2);

    name = "EBBCT number of entries";
    meEBBCaloEntriesVsCry_ = dqmStore_->book1D(name, name,1700,1.,1701.);
    meEBBCaloEntriesVsCry_->setAxisTitle("crystal", 1);
    meEBBCaloEntriesVsCry_->setAxisTitle("number of events (prescaled)", 2);

    name = "EBBCT energy deposition in the 3x3";
    meEBBCaloBeamCentered_ = dqmStore_->book2D(name, name,3,-1.5,1.5,3,-1.5,1.5);
    meEBBCaloBeamCentered_->setAxisTitle("\\Delta \\eta", 1);
    meEBBCaloBeamCentered_->setAxisTitle("\\Delta \\phi", 2);

    name = "EBBCT E1 in the max cry";
    meEBBCaloE1MaxCry_= dqmStore_->book1D(name,name,500,0.,9000.);
    meEBBCaloE1MaxCry_->setAxisTitle("rec Ene (ADC)", 1);

    name = "EBBCT Desynchronization vs step";
    meEBBCaloDesync_= dqmStore_->book1D(name, name, 85 ,1.,86.);
    meEBBCaloDesync_->setAxisTitle("step", 1);
    meEBBCaloDesync_->setAxisTitle("Desynchronized events", 2);

  }

}

void EBBeamCaloTask::cleanup(void){

  if ( ! init_ ) return;

  if ( dqmStore_ ) {
    dqmStore_->setCurrentFolder(prefixME_ + "/EBBeamCaloTask");
    for (int i = 0; i < cryInArray_ ; i++) {
      if ( meBBCaloPulseProf_[i] ) dqmStore_->removeElement( meBBCaloPulseProf_[i]->getName() );
      meBBCaloPulseProf_[i] = 0;
      if ( meBBCaloPulseProfG12_[i] ) dqmStore_->removeElement( meBBCaloPulseProfG12_[i]->getName() );
      meBBCaloPulseProfG12_[i] = 0;
      if ( meBBCaloGains_[i] ) dqmStore_->removeElement( meBBCaloGains_[i]->getName() );
      meBBCaloGains_[i] = 0;
      if ( meBBCaloEne_[i] ) dqmStore_->removeElement( meBBCaloEne_[i]->getName() );
      meBBCaloEne_[i] = 0;

    }

    if ( meBBCaloCryRead_ ) dqmStore_->removeElement( meBBCaloCryRead_->getName() );
    meBBCaloCryRead_ = 0;
    if ( meBBCaloAllNeededCry_ ) dqmStore_->removeElement( meBBCaloAllNeededCry_->getName() );
    meBBCaloAllNeededCry_ = 0;
    if ( meBBNumCaloCryRead_ ) dqmStore_->removeElement( meBBNumCaloCryRead_->getName() );
    meBBNumCaloCryRead_ = 0;
    if ( meBBCaloE3x3_ ) dqmStore_->removeElement( meBBCaloE3x3_->getName() );
    meBBCaloE3x3_ = 0;
    if ( meBBCaloE3x3Moving_ ) dqmStore_->removeElement( meBBCaloE3x3Moving_->getName() );
    meBBCaloE3x3Moving_ = 0;
    if ( meBBCaloCryOnBeam_ ) dqmStore_->removeElement( meBBCaloCryOnBeam_->getName() );
    meBBCaloCryOnBeam_ = 0;
    if ( meBBCaloMaxEneCry_ ) dqmStore_->removeElement( meBBCaloMaxEneCry_->getName() );
    meBBCaloMaxEneCry_ = 0;
    if ( TableMoving_ ) dqmStore_->removeElement( TableMoving_->getName() );
    TableMoving_ = 0;
    if ( CrystalsDone_ ) dqmStore_->removeElement( CrystalsDone_->getName() );
    CrystalsDone_ = 0;
    if ( CrystalInBeam_vs_Event_ ) dqmStore_->removeElement( CrystalInBeam_vs_Event_->getName() );
    CrystalInBeam_vs_Event_ = 0;
    if( meEBBCaloReadCryErrors_ ) dqmStore_->removeElement( meEBBCaloReadCryErrors_->getName() );
    meEBBCaloReadCryErrors_ = 0;
    if( meEBBCaloE1vsCry_ ) dqmStore_->removeElement( meEBBCaloE1vsCry_->getName() );
    meEBBCaloE1vsCry_ = 0;
    if( meEBBCaloE3x3vsCry_ ) dqmStore_->removeElement( meEBBCaloE3x3vsCry_->getName() );
    meEBBCaloE3x3vsCry_ = 0;
    if( meEBBCaloEntriesVsCry_ )  dqmStore_->removeElement( meEBBCaloEntriesVsCry_->getName() );
    meEBBCaloEntriesVsCry_ = 0;
    if( meEBBCaloBeamCentered_ ) dqmStore_->removeElement( meEBBCaloBeamCentered_->getName() );
    meEBBCaloBeamCentered_ = 0;
    if( meEBBCaloE1MaxCry_ ) dqmStore_->removeElement(meEBBCaloE1MaxCry_->getName() );
    meEBBCaloE1MaxCry_ = 0;
    if( meEBBCaloDesync_ ) dqmStore_->removeElement(meEBBCaloDesync_->getName() );
    meEBBCaloDesync_ = 0;
  }

  init_ = false;

}

void EBBeamCaloTask::endJob(void){

  edm::LogInfo("EBBeamCaloTask") << "analyzed " << ievt_ << " events";

  if ( enableCleanup_ ) this->cleanup();

}

void EBBeamCaloTask::analyze(const edm::Event& e, const edm::EventSetup& c){

  bool enable = false;

  edm::Handle<EcalRawDataCollection> dcchs;

  if ( e.getByLabel(EcalRawDataCollection_, dcchs) ) {

    for ( EcalRawDataCollection::const_iterator dcchItr = dcchs->begin(); dcchItr != dcchs->end(); ++dcchItr ) {

      if ( Numbers::subDet( *dcchItr ) != EcalBarrel ) continue;

      if ( dcchItr->getRunType() == EcalDCCHeaderBlock::BEAMH4 ||
           dcchItr->getRunType() == EcalDCCHeaderBlock::BEAMH2 ) enable = true;

    }

  } else {
    edm::LogWarning("EBBeamCaloTask") << EcalRawDataCollection_ << " not available";
  }

  if ( ! enable ) return;
  if ( ! init_ ) this->setup();
  ievt_++;

  edm::Handle<EcalTBEventHeader> pEventHeader;
  const EcalTBEventHeader* evtHeader=0;

  if ( e.getByLabel(EcalTBEventHeader_, pEventHeader) ) {
    evtHeader = pEventHeader.product(); // get a ptr to the product
  } else {
    std::cerr << "Error! can't get the product for the event header" << std::endl;
  }

  //FIX ME, in the task we should use LV1 instead of ievt_ (prescaling)
  int cry_in_beam = 0;
  bool tb_moving = false;//just for test, to be filled with info from the event
  int event = 0;

  if(evtHeader){
    cry_in_beam = evtHeader->crystalInBeam();
    tb_moving = evtHeader->tableIsMoving();
    event = evtHeader->eventNumber();
    if( evtHeader->syncError() ) {meEBBCaloDesync_->Fill(crystal_step_);}
  }
  else {
    cry_in_beam =   previous_cry_in_beam_;
    tb_moving = lastStableStatus_;
    event = previous_ev_num_ +10;
  }

  previous_cry_in_beam_ = cry_in_beam;
  previous_ev_num_ = event;


  bool reset_histos_stable = false;
  bool reset_histos_moving = false;

  bool skip_this_event = false;

  if(ievt_ < 3){last_cry_in_beam_ = cry_in_beam;}

  if(tb_moving){

    TableMoving_->Fill(1);
    if( PreviousTableStatus_[0] == 0 &&  PreviousTableStatus_[1] == 1 && lastStableStatus_ == 0){
      reset_histos_moving=true;
      wasFakeChange_ = false;
      // ! Warning! This works in the assumption that the crystal in beam stay the same
      // while the tb is moving and is set to the new one only when the table
      // reaches the new position
      lastStableStatus_ = 1;

    }
    else if( PreviousTableStatus_[1] == 0) {
      skip_this_event=true;
      changed_tb_status_ = true;
      wasFakeChange_ = true;
    }
    // just skip the first event when the table change status
    PreviousTableStatus_[0] = PreviousTableStatus_[1];
    PreviousTableStatus_[1] = 1;
  }//end of if(tb_moving)

  else {// table is not moving

    TableMoving_->Fill(0);
    if( PreviousTableStatus_[0] == 1 &&  PreviousTableStatus_[1] == 0 && lastStableStatus_ == 1){
      //reset_histos_stable = true;
      wasFakeChange_ = false;
      lastStableStatus_ = 0;
    }
    else if(PreviousTableStatus_[1] == 1) {
      skip_this_event=true;
      changed_tb_status_ = true;
      wasFakeChange_ = true;
    }
    // just skip the first event when the table change status
    PreviousTableStatus_[0]=PreviousTableStatus_[1];
    PreviousTableStatus_[1]=0;

    // check also whether cry in beam  has changed
    if(  PreviousCrystalinBeam_[0] == PreviousCrystalinBeam_[1]  &&   PreviousCrystalinBeam_[1] != PreviousCrystalinBeam_[2] && PreviousCrystalinBeam_[2] == cry_in_beam ){
      reset_histos_stable=true;
      wasFakeChange_ = false;
    }
    else if (PreviousCrystalinBeam_[2] != cry_in_beam){
      changed_cry_in_beam_ = true;
      skip_this_event=true;
      wasFakeChange_ = true;
    }
    // }

    PreviousCrystalinBeam_[0] = PreviousCrystalinBeam_[1];
    PreviousCrystalinBeam_[1] = PreviousCrystalinBeam_[2];
    PreviousCrystalinBeam_[2] =  cry_in_beam;
  }


    if( !tb_moving ) {CrystalInBeam_vs_Event_->Fill(event,float(cry_in_beam));}
    else{CrystalInBeam_vs_Event_->Fill(event,-100); }
    if ( !profileArranged_ ){
      float dd=0;
      int mbin =0;
      for( int bin=1; bin < 20001; bin++ ){
	float temp = CrystalInBeam_vs_Event_->getBinContent(bin);
	if(temp>0){ dd= temp+0.01; mbin=bin; break;}
      }
      if(mbin >0) { CrystalInBeam_vs_Event_->Fill(20*mbin-1,dd);}
      profileArranged_ = true;
    }

  if(reset_histos_moving){
    edm::LogInfo("EBBeamCaloTask") << "event " << ievt_ << " resetting histos for moving table!! ";

    table_step_++;

    meBBCaloE3x3Moving_->Reset();

  }


  if(reset_histos_stable){
    if( event - event_last_reset_ > 30){//to be tuned, to avoid a double reset for the change in the table status and
                                        //in the crystal in beam. This works ONLY if the crystal in beam stay the same
                                        // while the table is moving.
                                        //One can also think to remove the reset of the histograms when the table change
                                        // status from moving to stable, and to leave the reset only if the cry_in_beam changes.

      edm::LogInfo("EBBeamCaloTask") << "event " << ievt_ << " resetting histos for stable table!! ";

      event_last_reset_ = event;

      last_cry_in_beam_ = cry_in_beam;
      crystal_step_++;

      //here the follwowing histos should be reset
      for (int u=0;u<cryInArray_;u++){
	meBBCaloPulseProf_[u]->Reset();
	meBBCaloPulseProfG12_[u]->Reset();
	meBBCaloGains_[u]->Reset();
	meBBCaloEne_[u]->Reset();
      }
      meBBCaloCryRead_->Reset();
      meBBCaloE3x3_->Reset();
      meEBBCaloBeamCentered_->Reset();
    }
  }

 if(skip_this_event){
   edm::LogInfo("EBBeamCaloTask") << "event " << event <<" analyzed: "<<ievt_ << " : skipping this event!! ";
   return;}

 // now CrystalsDone_ contains the crystal on beam at the beginning fo a new step, and not when it has finished !!
 // <5 just to avoid that we skip the event just after the reset and we do not set CrystalsDone_ .
 // if( ievt_ - event_last_reset_ < 5){ CrystalsDone_->setBinContent(cry_in_beam , crystal_step_ );}
 CrystalsDone_->setBinContent(cry_in_beam , crystal_step_ );
  int eta_c = ( cry_in_beam-1)/20 ;
  int phi_c = ( cry_in_beam-1)%20 ;

  float xie = eta_c + 0.5;
  float xip = phi_c + 0.5;
  if (!tb_moving) {meBBCaloCryOnBeam_->Fill(xie,xip);}

  edm::Handle<EBDigiCollection> digis;
  e.getByLabel(EBDigiCollection_, digis);
  int nebd = digis->size();

  meBBNumCaloCryRead_->Fill(nebd);

  //matrix 7x7 around cry in beam
  int cry_to_beRead[49]; //0 or -1 for non existing crystals (eg 1702)
  for(int u=0;u<49;u++){cry_to_beRead[u]=0;}
  // chech that all the crystals in the 7x7 exist
  for(int de=-3; de<4; de++){
    for(int dp=-3; dp<4; dp++){
      int u = de -7*dp + 24;
      bool existing_cry = (phi_c+dp) >= 0 && (phi_c+dp) <= 19 && (eta_c+de) >=0 && (eta_c+de) <= 84;
      if(!existing_cry){cry_to_beRead[u]=-1;}
    }
  }


  meEBBCaloEntriesVsCry_->Fill(cry_in_beam);

  for ( EBDigiCollection::const_iterator digiItr = digis->begin(); digiItr != digis->end(); ++digiItr ) {

    EBDetId id = digiItr->id();

    int ic = id.ic();
    int ie = (ic-1)/20;
    int ip = (ic-1)%20;

    int deta_c= ie - eta_c;
    int dphi_c= ip - phi_c;
    if (! tb_moving){meBBCaloCryRead_->Fill(deta_c, dphi_c);}

    if(std::abs(deta_c) > 3 || std::abs(dphi_c) > 3){continue;}
    int i_toBeRead = deta_c -7*dphi_c + 24;
    if( i_toBeRead > -1 &&  i_toBeRead <49){
      cry_to_beRead[i_toBeRead]++;
    }

    if(std::abs(deta_c) > 1 || std::abs(dphi_c) > 1){continue;}
    int i_in_array = deta_c -3*dphi_c + 4;

    if( i_in_array < 0 || i_in_array > 8 ){continue;}

    EBDataFrame dataframe = (*digiItr);

    for (int i = 0; i < 10; i++) {
      int adc = dataframe.sample(i).adc();
      int gainid = dataframe.sample(i).gainId();

      if ( gainid == 1 ){// gain 12
	if(! tb_moving){
	  meBBCaloPulseProfG12_[i_in_array]->Fill(i,float(adc));
	  meBBCaloPulseProf_[i_in_array]->Fill(i,float(adc));
	  meBBCaloGains_[i_in_array]->Fill(12);
	}

      }
      else if ( gainid == 2 ){// gain 6
	float val = (float(adc)-defaultPede_)*2 + defaultPede_;
	if(! tb_moving){
	  meBBCaloPulseProf_[i_in_array]->Fill(i,val);
	  meBBCaloGains_[i_in_array]->Fill(6);
	}

      }
      else if ( gainid == 3 ){// gain 1
	float val = (float(adc)-defaultPede_)*12 + defaultPede_;
	if(! tb_moving){
	meBBCaloPulseProf_[i_in_array]->Fill(i,val);
	meBBCaloGains_[i_in_array]->Fill(1);
	}

      }
    }// end of loop over samples
  }// end of loop over digis

  //now  if everything was correct cry_to_beRead should be filled with 1 or -1 but not 0
  bool all_cry_readout = true;

  if(all_cry_readout){ meBBCaloAllNeededCry_->Fill(1.5);}//bin3
  else {
    meBBCaloAllNeededCry_->Fill(-0.5);//bin1
    if( tb_moving ) {meEBBCaloReadCryErrors_->Fill( crystal_step_+0.5 );}
    else {meEBBCaloReadCryErrors_->Fill( crystal_step_ );}
  }

  //the part involving rechits

  edm::Handle<EcalUncalibratedRecHitCollection> hits;
  e.getByLabel(EcalUncalibratedRecHitCollection_, hits);
  int neh = hits->size();
  LogDebug("EBBeamCaloTask") << "event " << event <<" analyzed: "<< ievt_ << " hits collection size " << neh;
  float ene3x3=0;
  float maxEne = 0;
  int ieM =-1, ipM = -1;//for the crystal with maximum energy deposition
  float cryInBeamEne =0;
  for ( EcalUncalibratedRecHitCollection::const_iterator hitItr = hits->begin(); hitItr != hits->end(); ++hitItr ) {

    EBDetId id = hitItr->id();

    int ic = id.ic();
    int ie = (ic-1)/20;
    int ip = (ic-1)%20;

    int deta_c= ie - eta_c;
    int dphi_c= ip - phi_c;

    int i_in_array = deta_c -3*dphi_c + 4;

    float R_ene = hitItr->amplitude();
    if ( R_ene <= 0. ) R_ene = 0.0;
    if(R_ene > maxEne){
      maxEne=R_ene;
      ieM =ie; ipM = ip;
    }
    if(std::abs(deta_c) > 1 || std::abs(dphi_c) > 1){continue;}
    meEBBCaloBeamCentered_->Fill(deta_c,dphi_c,R_ene);

    if( i_in_array < 0 || i_in_array > 8 ){continue;}

    if(i_in_array == 4){cryInBeamEne = R_ene;}
    if(! tb_moving){meBBCaloEne_[i_in_array]->Fill(R_ene);}
    ene3x3 += R_ene;

  }//end of loop over rechits

  if (!tb_moving){
    meBBCaloE3x3_->Fill(ene3x3);
    meEBBCaloE1vsCry_->Fill(cry_in_beam , cryInBeamEne );
    meEBBCaloE3x3vsCry_->Fill(cry_in_beam, ene3x3 );

    meBBCaloMaxEneCry_->Fill(ieM,ipM);
    meEBBCaloE1MaxCry_->Fill(maxEne);
  }
  else{meBBCaloE3x3Moving_->Fill(ene3x3);}
  /////////////
}

