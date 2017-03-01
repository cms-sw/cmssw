#include "DQM/RPCMonitorClient/interface/RPCDaqInfo.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
#include "CondFormats/RunInfo/interface/RunInfo.h"
#include "CondFormats/RunInfo/interface/RunSummary.h"
#include "CondFormats/DataRecord/interface/RunSummaryRcd.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"



RPCDaqInfo::RPCDaqInfo(const edm::ParameterSet& ps) {
 
  FEDRange_.first  = ps.getUntrackedParameter<unsigned int>("MinimumRPCFEDId", 790);
  FEDRange_.second = ps.getUntrackedParameter<unsigned int>("MaximumRPCFEDId", 792);
  
  NumberOfFeds_ =FEDRange_.second -  FEDRange_.first +1;

  numberOfDisks_ = ps.getUntrackedParameter<int>("NumberOfEndcapDisks", 4);

  init_=false;
}

RPCDaqInfo::~RPCDaqInfo(){}
void RPCDaqInfo::beginJob(){}
void RPCDaqInfo::dqmEndLuminosityBlock(DQMStore::IBooker & ibooker, DQMStore::IGetter & igetter, edm::LuminosityBlock const & LB, edm::EventSetup const& iSetup){
  
  edm::eventsetup::EventSetupRecordKey recordKey(edm::eventsetup::EventSetupRecordKey::TypeTag::findType("RunInfoRcd"));
  
  if(!init_){this->myBooker(ibooker);}

  if(0 != iSetup.find( recordKey ) ) {
    
    //get fed summary information
    edm::ESHandle<RunInfo> sumFED;
    iSetup.get<RunInfoRcd>().get(sumFED);    
    std::vector<int> FedsInIds= sumFED->m_fed_in;   

    int FedCount=0;

    //loop on all active feds
    for(unsigned int fedItr=0;fedItr<FedsInIds.size(); ++fedItr) {
      int fedID=FedsInIds[fedItr];
      //make sure fed id is in allowed range  

      if(fedID>=FEDRange_.first && fedID<=FEDRange_.second) ++FedCount;
    }   

    //Fill active fed fraction ME
    if(NumberOfFeds_>0) DaqFraction_->Fill( FedCount/NumberOfFeds_);
    else  DaqFraction_->Fill(-1);
 
  }else{      
    DaqFraction_->Fill(-1);               
    return; 
  }
}


void RPCDaqInfo::dqmEndJob(DQMStore::IBooker &, DQMStore::IGetter &){}


void RPCDaqInfo::myBooker(DQMStore::IBooker & ibooker){

  //fraction of alive FEDs
  ibooker.setCurrentFolder("RPC/EventInfo/DAQContents");
  
  int limit = numberOfDisks_;
  if(numberOfDisks_ < 2) limit = 2;
  
  for (int i = -1 * limit; i<= limit;i++ ){//loop on wheels and disks
    if (i>-3 && i<3){//wheels
      std::stringstream streams;
      streams << "RPC_Wheel" << i;
      daqWheelFractions[i+2] = ibooker.bookFloat(streams.str());
      daqWheelFractions[i+2]->Fill(-1);
    }
    
    if (i == 0  || i > numberOfDisks_ || i< (-1 * numberOfDisks_))continue;
    
    int offset = numberOfDisks_;
    if (i>0) offset --; //used to skip case equale to zero
    
    std::stringstream streams;
    streams << "RPC_Disk" << i;
    daqDiskFractions[i+2] = ibooker.bookFloat(streams.str());
    daqDiskFractions[i+2]->Fill(-1);
  }


  //daq summary for RPCs
  ibooker.setCurrentFolder("RPC/EventInfo");
    
  DaqFraction_ = ibooker.bookFloat("DAQSummary");

  DaqMap_ = ibooker.book2D( "DAQSummaryMap","RPC DAQ Summary Map",15, -7.5, 7.5, 12, 0.5 ,12.5);

 //customize the 2d histo
  std::stringstream BinLabel;
  for (int i= 1 ; i<=15; i++){
    BinLabel.str("");
    if(i<13){
      BinLabel<<"Sec"<<i;
      DaqMap_->setBinLabel(i,BinLabel.str(),2);
    } 

    BinLabel.str("");
    if(i<5)
      BinLabel<<"Disk"<<i-5;
    else if(i>11)
      BinLabel<<"Disk"<<i-11;
    else if(i==11 || i==5)
      BinLabel.str("");
    else
      BinLabel<<"Wheel"<<i-8;
 
     DaqMap_->setBinLabel(i,BinLabel.str(),1);
  }

  init_=true;

}




