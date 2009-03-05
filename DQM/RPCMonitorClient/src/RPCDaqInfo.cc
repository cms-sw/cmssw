#include "DQM/RPCMonitorClient/interface/RPCDaqInfo.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
#include "CondFormats/RunInfo/interface/RunInfo.h"
#include "CondFormats/RunInfo/interface/RunSummary.h"
#include "CondFormats/DataRecord/interface/RunSummaryRcd.h"

using namespace std;
using namespace edm;
RPCDaqInfo::RPCDaqInfo(const edm::ParameterSet& ps) {
 
  FEDRange_.first  = ps.getUntrackedParameter<unsigned int>("MinimumRPCFEDId", 790);
  FEDRange_.second = ps.getUntrackedParameter<unsigned int>("MaximumRPCFEDId", 792);
  
  NumberOfFeds_ =FEDRange_.second -  FEDRange_.first +1;

}

RPCDaqInfo::~RPCDaqInfo(){}

void RPCDaqInfo::beginLuminosityBlock(const LuminosityBlock& lumiBlock, const  EventSetup& iSetup){
  
  edm::eventsetup::EventSetupRecordKey recordKey(edm::eventsetup::EventSetupRecordKey::TypeTag::findType("RunInfoRcd"));


  if(0 != iSetup.find( recordKey ) ) {
  
    //get fed summary information
    ESHandle<RunInfo> sumFED;
    iSetup.get<RunInfoRcd>().get(sumFED);    
    vector<int> FedsInIds= sumFED->m_fed_in;   

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


void RPCDaqInfo::endLuminosityBlock(const edm::LuminosityBlock&  lumiBlock, const  edm::EventSetup& iSetup){}


void RPCDaqInfo::beginJob(const edm::EventSetup& iSetup){

  dbe_ = 0;
  dbe_ = Service<DQMStore>().operator->();
  
 
  dbe_->setCurrentFolder("RPC/EventInfo/DAQContents");
  DaqFraction_= dbe_->bookFloat("RPCDaqFraction");  
}


void RPCDaqInfo::endJob() {}



void RPCDaqInfo::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup){}

