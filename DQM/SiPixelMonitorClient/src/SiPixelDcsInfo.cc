#include "DQM/SiPixelMonitorClient/interface/SiPixelDcsInfo.h"
//#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
#include "CondFormats/RunInfo/interface/RunInfo.h"
#include "CondFormats/RunInfo/interface/RunSummary.h"
#include "CondFormats/DataRecord/interface/RunSummaryRcd.h"

using namespace std;
using namespace edm;
SiPixelDcsInfo::SiPixelDcsInfo(const edm::ParameterSet& ps) {
 
//  DCSPixelRange_.first  = ps.getUntrackedParameter<unsigned int>("MinimumPixelDcsChannel", 0);
//  DCSPixelRange_.second = ps.getUntrackedParameter<unsigned int>("MaximumPixelDcsChannel", 39);
  
//  NumberOfDcsChannels_ =DCSPixelRange_.second -  DCSPixelRange_.first +1;
  NumberOfDcsChannels_ = 100;
}

SiPixelDcsInfo::~SiPixelDcsInfo(){}

void SiPixelDcsInfo::beginLuminosityBlock(const LuminosityBlock& lumiBlock, const  EventSetup& iSetup){
  
  edm::eventsetup::EventSetupRecordKey recordKey(edm::eventsetup::EventSetupRecordKey::TypeTag::findType("RunInfoRcd"));


  if(0 != iSetup.find( recordKey ) ) {
    // cout<<"record key found"<<endl;
    //get DCS channel information
//    ESHandle<RunInfo> sumDCS;
//    iSetup.get<RunInfoRcd>().get(sumDCS);    
//    vector<int> DcsChannelsInIds= sumDCS->m_dcs_in;   

/*    int DcsCount=0;
    int DcsCountBarrel=0;
    int DcsCountEndcap=0;
*/
    //loop on all active DCS channels
/*    for(unsigned int fedItr=0;fedItr<FedsInIds.size(); ++fedItr) {
      int fedID=FedsInIds[fedItr];
      //make sure fed id is in allowed range  
      //cout<<fedID<<endl;   
      if(fedID>=FEDRange_.first && fedID<=FEDRange_.second){
        ++FedCount;
	if(fedID>=0 && fedID<=31){
	  ++FedCountBarrel;
	}else if(fedID>=32 && fedID<=39){
	  ++FedCountEndcap;
	}
      }
    }   
*/
    //Fill active dcs fraction ME's
//    if(NumberOfFeds_>0){
      //all Pixel:
      DcsFraction_->Fill(-1.);
      //Barrel:
      DcsFractionBarrel_->Fill(-1.);
      //Endcap:
      DcsFractionEndcap_->Fill(-1.);
/*    }else{
      DaqFraction_->Fill(-1);
      DaqFractionBarrel_->Fill(-1);
      DaqFractionEndcap_->Fill(-1);
    }

  }else{      
    DaqFraction_->Fill(-1);    
    DaqFractionBarrel_->Fill(-1);
    DaqFractionEndcap_->Fill(-1);
*/ 
    return; 
  }
}


void SiPixelDcsInfo::endLuminosityBlock(const edm::LuminosityBlock&  lumiBlock, const  edm::EventSetup& iSetup){}


void SiPixelDcsInfo::beginJob(const edm::EventSetup& iSetup){

  dbe_ = 0;
  dbe_ = Service<DQMStore>().operator->();
  
 
  dbe_->setCurrentFolder("Pixel/EventInfo/DCSContents");
  DcsFraction_= dbe_->bookFloat("PixelDcsFraction");  
  DcsFractionBarrel_= dbe_->bookFloat("PixelBarrelDcsFraction");  
  DcsFractionEndcap_= dbe_->bookFloat("PixelEndcapDcsFraction");  
}


void SiPixelDcsInfo::endJob() {}



void SiPixelDcsInfo::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup){}

