#include "DQM/SiPixelMonitorClient/interface/SiPixelDaqInfo.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
#include "CondFormats/RunInfo/interface/RunInfo.h"
#include "CondFormats/RunInfo/interface/RunSummary.h"
#include "CondFormats/DataRecord/interface/RunSummaryRcd.h"

using namespace std;
using namespace edm;
SiPixelDaqInfo::SiPixelDaqInfo(const edm::ParameterSet& ps) {
 
  FEDRange_.first  = ps.getUntrackedParameter<unsigned int>("MinimumPixelFEDId", 0);
  FEDRange_.second = ps.getUntrackedParameter<unsigned int>("MaximumPixelFEDId", 39);
  
  NumberOfFeds_ =FEDRange_.second -  FEDRange_.first +1;

}

SiPixelDaqInfo::~SiPixelDaqInfo(){}

void SiPixelDaqInfo::beginLuminosityBlock(const LuminosityBlock& lumiBlock, const  EventSetup& iSetup){
  
  edm::eventsetup::EventSetupRecordKey recordKey(edm::eventsetup::EventSetupRecordKey::TypeTag::findType("RunInfoRcd"));


  if(0 != iSetup.find( recordKey ) ) {
    // cout<<"record key found"<<endl;
    //get fed summary information
    ESHandle<RunInfo> sumFED;
    iSetup.get<RunInfoRcd>().get(sumFED);    
    vector<int> FedsInIds= sumFED->m_fed_in;   

    int FedCount=0;

    //loop on all active feds
    for(unsigned int fedItr=0;fedItr<FedsInIds.size(); ++fedItr) {
      int fedID=FedsInIds[fedItr];
      //make sure fed id is in allowed range  
      //cout<<fedID<<endl;   
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


void SiPixelDaqInfo::endLuminosityBlock(const edm::LuminosityBlock&  lumiBlock, const  edm::EventSetup& iSetup){}


void SiPixelDaqInfo::beginJob(const edm::EventSetup& iSetup){

  dbe_ = 0;
  dbe_ = Service<DQMStore>().operator->();
  
 
  dbe_->setCurrentFolder("Pixel/EventInfo/DAQContents");
  DaqFraction_= dbe_->bookFloat("PixelDaqFraction");  
}


void SiPixelDaqInfo::endJob() {}



void SiPixelDaqInfo::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup){}

