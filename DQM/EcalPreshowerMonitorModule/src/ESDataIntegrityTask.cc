#include "DQM/EcalPreshowerMonitorModule/interface/ESDataIntegrityTask.h"

#include <iostream>

using namespace cms;
using namespace edm;
using namespace std;

ESDataIntegrityTask::ESDataIntegrityTask(const ParameterSet& ps) {

  label_        = ps.getUntrackedParameter<string>("Label");
  instanceName_ = ps.getUntrackedParameter<string>("InstanceES");
  sta_          = ps.getUntrackedParameter<bool>("RunStandalone", false);

  init_ = false;

  meCRCError_ = 0;

  meDCCError_ = 0;  

  fedIds_=0;
  DCCfedId1_=0;
  DCCfedId4_=0;
  DCCfedId10_=0;
  DCCfedId40_=0;
  Kchip_=0;

  dbe_ = Service<DaqMonitorBEInterface>().operator->();

}

ESDataIntegrityTask::~ESDataIntegrityTask(){
}

void ESDataIntegrityTask::beginJob(const EventSetup& c) {
  
  ievt_ = 0;
  
  if ( dbe_ ) {
    dbe_->setCurrentFolder("ES/ESDataIntegrityTask");
    dbe_->rmdir("ES/ESDataIntegrityTask");
  }

}

void ESDataIntegrityTask::setup(void){
  
  init_ = true;
  
  Char_t hist[200];
  
  if ( dbe_ ) {   
    dbe_->setCurrentFolder("ES/ESDataIntegrityTask");
    
    sprintf(hist, "ES CRC Errors");
    meCRCError_ = dbe_->book2D(hist, hist, 36, 0, 36, 2, 0, 2);

    sprintf(hist, "ES DCC Errors");      
    meDCCError_ = dbe_->book1D(hist, hist, 256, 0, 256);  
    
    sprintf(hist, "ES DCC FedId");
    fedIds_= dbe_->book1D(hist, hist, 50, 0, 50);
    sprintf(hist, "ES DCC FedId=1");
    DCCfedId1_=dbe_->book1D(hist, hist, 12, 0, 12);
    sprintf(hist, "ES DCC FedId=4");
    DCCfedId4_=dbe_->book1D(hist, hist, 12, 0, 12);
    sprintf(hist, "ES DCC FedId=10 to 13");
    DCCfedId10_=dbe_->book1D(hist, hist, 12, 0, 12);
    sprintf(hist, "ES DCC FedId=40 to 43");
    DCCfedId40_=dbe_->book1D(hist, hist, 12, 0, 12);
    sprintf(hist, "ES Kchip");
    Kchip_=dbe_->book1D(hist, hist, 12, 0, 12);

  }

}

void ESDataIntegrityTask::cleanup(void){

  if (sta_) return;

  if ( dbe_ ) {
    dbe_->setCurrentFolder("ES/ESDataIntegrityTask");
    
    if (meCRCError_) dbe_->removeElement( meCRCError_->getName() );
    meCRCError_ = 0; 

    if ( meDCCError_ )dbe_->removeElement( meDCCError_->getName() );
    meDCCError_ = 0;

    if ( fedIds_)dbe_->removeElement(fedIds_->getName() );
    fedIds_ = 0;

    if ( DCCfedId1_)dbe_->removeElement(DCCfedId1_->getName() );
    DCCfedId1_= 0;

    if ( DCCfedId4_)dbe_->removeElement(DCCfedId4_->getName() );
    DCCfedId4_= 0;

    if ( DCCfedId10_)dbe_->removeElement(DCCfedId10_->getName() );
    DCCfedId10_= 0;

    if ( DCCfedId40_)dbe_->removeElement(DCCfedId40_->getName() );
    DCCfedId40_= 0;

  }

  init_ = false;

}

void ESDataIntegrityTask::endJob(void) {

  LogInfo("ESDataIntegrityTask") << "analyzed " << ievt_ << " events";

  if ( init_ ) this->cleanup();

}

void ESDataIntegrityTask::analyze(const Event& e, const EventSetup& c){

  if ( ! init_ ) this->setup();
  ievt_++;
  
  Handle<ESRawDataCollection> dccs;
  try {
    e.getByLabel(label_, instanceName_, dccs);
  } catch ( cms::Exception &e ) {
    LogDebug("") << "ESDataIntegrity : Error! can't get ES raw data collection !" << std::endl;
  }  

  Handle<ESLocalRawDataCollection> kchips;
  try {
    e.getByLabel(label_, instanceName_, kchips);
  } catch ( cms::Exception &e ) {
    LogDebug("") << "ESDataIntegrity : Error! can't get ES local raw data collection !" << std::endl;
  }  

  // DCC
  for ( ESRawDataCollection::const_iterator dccItr = dccs->begin(); dccItr != dccs->end(); ++dccItr ) {

    ESDCCHeaderBlock dcc = (*dccItr);
     
    fedIds_->Fill(dcc.fedId());

    if (dcc.fedId()==1) {   
      if(dcc.getPacketLength()!=-1)        DCCfedId1_->Fill(1);
      if(dcc.getBMMeasurements()!=-1)      DCCfedId1_->Fill(2);
      if(dcc.getRunNumber()!=-1)           DCCfedId1_->Fill(3);
      if(dcc.getBeginOfSpillSec()!=-1)     DCCfedId1_->Fill(4);
      if(dcc.getBeginOfSpillMiliSec()!=-1) DCCfedId1_->Fill(5);
      if(dcc.getEndOfSpillSec()!=-1)       DCCfedId1_->Fill(6);
      if(dcc.getEndOfSpillMiliSec()!=-1)   DCCfedId1_->Fill(7);
      if(dcc.getBeginOfSpillLV1()!=-1)     DCCfedId1_->Fill(8);
      if(dcc.getEndOfSpillLV1()!=-1)       DCCfedId1_->Fill(9);
    }


    if (dcc.fedId()==4) {       
        if(dcc.getPacketLength()!=-1)        DCCfedId4_->Fill(1);
        if(dcc.getMajorVersion()!=-1)        DCCfedId4_->Fill(2);
        if(dcc.getMinorVersion()!=-1)        DCCfedId4_->Fill(3);
        if(dcc.getTimeStampSec()!=-1)        DCCfedId4_->Fill(4);
        if(dcc.getTimeStampUSec()!=-1)       DCCfedId4_->Fill(5);
        if(dcc.getLV1()!=-1)                 DCCfedId4_->Fill(6);
        if(dcc.getRunNumber()!=-1)           DCCfedId4_->Fill(7);
        if(dcc.getSpillNumber()!=-1)         DCCfedId4_->Fill(8);
        if(dcc.getEventInSpill()!=-1)        DCCfedId4_->Fill(9);
        if(dcc.getVMEError()!=-1)            DCCfedId4_->Fill(10);
    }

    if ((dcc.fedId()>=10)&&(dcc.fedId()<=13)) {   
        if(dcc.getPacketLength()!=-1)     DCCfedId10_->Fill(1);
        if(dcc.getRunNumber()!=-1)        DCCfedId10_->Fill(2);
        if(dcc.getLV1()!=-1)              DCCfedId10_->Fill(3);
        if(dcc.getMajorVersion()!=-1)     DCCfedId10_->Fill(4);
        if(dcc.getMinorVersion()!=-1)     DCCfedId10_->Fill(5);
        if(dcc.getBC()!=-1)               DCCfedId10_->Fill(6);
        if(dcc.getEV()!=-1)               DCCfedId10_->Fill(7);
    }


    if ((dcc.fedId()>=40)&&(dcc.fedId()<=43)){   
      
      meDCCError_->Fill(dcc.getDCCErrors());
      
    }

  }

  //Kchip
  for ( ESLocalRawDataCollection::const_iterator kItr = kchips->begin(); kItr != kchips->end(); ++kItr ) {

    ESKCHIPBlock kchip = (*kItr);

    meCRCError_->Fill(kchip.fiberId(), kchip.getCRC(), 1);

    if(kchip.id()!=-1)        Kchip_->Fill(1);
    if(kchip.dccdId()!=-1)    Kchip_->Fill(2);
    if(kchip.fedId()!=-1)     Kchip_->Fill(3);
    if(kchip.getBC()!=-1)     Kchip_->Fill(4);
    if(kchip.getEC()!=-1)     Kchip_->Fill(5);
    if(kchip.getFlag1()!=-1)  Kchip_->Fill(6);
    if(kchip.getFlag2()!=-1)  Kchip_->Fill(7);
    if(kchip.getFlag2()!=-1)  Kchip_->Fill(8);

  }

}
