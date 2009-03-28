#include "DQM/RPCMonitorClient/interface/RPCDCSSummary.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"


using namespace std;
using namespace edm;



RPCDCSSummary::RPCDCSSummary(const ParameterSet& ps) {

 numberOfDisks_ = ps.getUntrackedParameter<int>("NumberOfEndcapDisks", 3);
}

RPCDCSSummary::~RPCDCSSummary() {}


void RPCDCSSummary::beginJob(const EventSetup& setup){
  // get the DQMStore
  theDbe = Service<DQMStore>().operator->();
  
  theDbe->setCurrentFolder("RPC/EventInfo");
  // global fraction
  totalDCSFraction = theDbe->bookFloat("DCSSummary");  
  totalDCSFraction->Fill(-1);

  DCSMap_ = theDbe->book2D( "DCSSummaryMap","RPC DCS Summary Map",15, -7.5, 7.5, 12, 0.5 ,12.5);
  
  //customize the 2d histo
  stringstream BinLabel;
  for (int i= 1 ; i<=15; i++){
    BinLabel.str("");
    if(i<13){
      BinLabel<<"Sec"<<i;
      DCSMap_->setBinLabel(i,BinLabel.str(),2);
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
 
     DCSMap_->setBinLabel(i,BinLabel.str(),1);
  }


  // book the ME
  theDbe->setCurrentFolder("RPC/EventInfo/DCSContents");

  int limit = numberOfDisks_;
  if(numberOfDisks_ < 2) limit = 2;
  

  for (int i = -1 * limit; i<= limit;i++ ){//loop on wheels and disks
    if (i>-3 && i<3){//wheels
      stringstream streams;
      streams << "RPC_Wheel" << i;
      dcsWheelFractions[i+2] = theDbe->bookFloat(streams.str());
      dcsWheelFractions[i+2]->Fill(-1);
   }
    
    if (i == 0  || i > numberOfDisks_ || i< (-1 * numberOfDisks_))continue;
    
    int offset = numberOfDisks_;
    if (i>0) offset --; //used to skip case equale to zero
    stringstream streams;
    streams << "RPC_Disk" << i;
    dcsDiskFractions[i+2] = theDbe->bookFloat(streams.str());
    dcsDiskFractions[i+2]->Fill(-1);
  }
}



void RPCDCSSummary::beginLuminosityBlock(const LuminosityBlock& lumi, const  EventSetup& setup) {
}




void RPCDCSSummary::endLuminosityBlock(const LuminosityBlock&  lumi, const  EventSetup& setup){}



void RPCDCSSummary::endJob() {}



void RPCDCSSummary::analyze(const Event& event, const EventSetup& setup){}





