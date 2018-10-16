#include "DQM/RPCMonitorClient/interface/RPCDCSSummary.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

//#include "DQMServices/Core/interface/DQMStore.h"
//#include "DQMServices/Core/interface/MonitorElement.h"

//CondFormats
#include "CondFormats/RunInfo/interface/RunInfo.h"
#include "CondFormats/DataRecord/interface/RunSummaryRcd.h"


RPCDCSSummary::RPCDCSSummary(const edm::ParameterSet& ps) {

 numberOfDisks_ = ps.getUntrackedParameter<int>("NumberOfEndcapDisks", 4);

 FEDRange_.first  = ps.getUntrackedParameter<unsigned int>("MinimumRPCFEDId", 790);
 FEDRange_.second = ps.getUntrackedParameter<unsigned int>("MaximumRPCFEDId", 792);
 
 offlineDQM_ = ps.getUntrackedParameter<bool> ("OfflineDQM",true); 

 NumberOfFeds_ =FEDRange_.second -  FEDRange_.first +1;
 init_=false;
 defaultValue_ = 1.;
}

RPCDCSSummary::~RPCDCSSummary() {}

void RPCDCSSummary::beginJob() {}

void RPCDCSSummary::dqmEndLuminosityBlock(DQMStore::IBooker & ibooker, DQMStore::IGetter & igetter, edm::LuminosityBlock const & lumiB, edm::EventSetup const& setup){
  
  if(!init_){
    this->checkDCSbit(setup);        
    if(!offlineDQM_ ){this->myBooker(ibooker);}
  }
}


void  RPCDCSSummary::dqmEndJob(DQMStore::IBooker & ibooker, DQMStore::IGetter & igetter) {
 
  if( offlineDQM_){this->myBooker(ibooker);}
}



void  RPCDCSSummary::checkDCSbit( edm::EventSetup const& setup ){

 defaultValue_ = 1.; 
 
 if(auto runInfoRec = setup.tryToGet<RunInfoRcd>()) {
   defaultValue_ = -1.;
   //get fed summary information
   edm::ESHandle<RunInfo> sumFED;
   runInfoRec->get(sumFED);
   std::vector<int> FedsInIds= sumFED->m_fed_in;   
   unsigned int f = 0;
   bool flag = false;
   while(!flag && f < FedsInIds.size()) {
     int fedID=FedsInIds[f];
     //make sure fed id is in allowed range  
     if(fedID>=FEDRange_.first && fedID<=FEDRange_.second) {
       defaultValue_ = 1.;
       flag = true;
     } 
     f++;
   }   
 }   
 init_ = true;
}

void  RPCDCSSummary::myBooker(DQMStore::IBooker & ibooker ){

  ibooker.setCurrentFolder("RPC/EventInfo");
  // global fraction
  totalDCSFraction = ibooker.bookFloat("DCSSummary");  
  totalDCSFraction->Fill(defaultValue_);

  DCSMap_ = ibooker.book2D( "DCSSummaryMap","RPC DCS Summary Map",15, -7.5, 7.5, 12, 0.5 ,12.5);
  
  //customize the 2d histo
  std::stringstream BinLabel;
  for (int i= 1 ; i<13; i++){
    BinLabel.str("");
    BinLabel<<"Sec"<<i;
    DCSMap_->setBinLabel(i,BinLabel.str(),2);
  } 

  for(int i = -2; i<=2; i++){
    BinLabel.str("");
    BinLabel<<"Wheel"<<i;
    DCSMap_->setBinLabel((i+8),BinLabel.str(),1);
  }

  for(int i = 1; i<=numberOfDisks_; i++){
    BinLabel.str("");
    BinLabel<<"Disk"<<i;
    DCSMap_->setBinLabel((i+11),BinLabel.str(),1);
    BinLabel.str("");
    BinLabel<<"Disk"<<-i;
    DCSMap_->setBinLabel((-i+5),BinLabel.str(),1);
  }

  //fill the histo with "1" --- just for the moment
  for(int i=1; i<=15; i++){
     for (int j=1; j<=12; j++ ){
       if(i==5 || i==11 || (j>6 && (i<6 || i>10)))    
	 DCSMap_->setBinContent(i,j,-1);//bins that not correspond to subdetector parts
       else
	 DCSMap_->setBinContent(i,j,defaultValue_);
     }
   }


  if(numberOfDisks_ < 4){
    for (int j=1; j<=12; j++ ){
      DCSMap_->setBinContent(1,j,-1);//bins that not correspond to subdetector parts
      DCSMap_->setBinContent(15,j,-1);
    }
 }

  // book the ME
  ibooker.setCurrentFolder("RPC/EventInfo/DCSContents");

  int limit = numberOfDisks_;
  if(numberOfDisks_ < 2) limit = 2;
  

  for (int i = -1 * limit; i<= limit;i++ ){//loop on wheels and disks
    if (i>-3 && i<3){//wheels
      std::stringstream streams;
      streams << "RPC_Wheel" << i;
      dcsWheelFractions[i+2] = ibooker.bookFloat(streams.str());
      dcsWheelFractions[i+2]->Fill(defaultValue_);
   }
    
    if (i == 0  || i > numberOfDisks_ || i< (-1 * numberOfDisks_))continue;
    
    int offset = numberOfDisks_;
    if (i>0) offset --; //used to skip case equale to zero
    std::stringstream streams;
    streams << "RPC_Disk" << i;
    dcsDiskFractions[i+2] = ibooker.bookFloat(streams.str());
    dcsDiskFractions[i+2]->Fill(defaultValue_);
  }
}
