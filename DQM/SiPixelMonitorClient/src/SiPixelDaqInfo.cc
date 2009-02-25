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
    int FedCountBarrel=0;
    int FedCountEndcap=0;
    int FedCountShellmI=0;
    int FedCountShellmO=0;
    int FedCountShellpI=0;
    int FedCountShellpO=0;
    int FedCountHalfCylindermI=0;
    int FedCountHalfCylindermO=0;
    int FedCountHalfCylinderpI=0;
    int FedCountHalfCylinderpO=0;

    //loop on all active feds
    for(unsigned int fedItr=0;fedItr<FedsInIds.size(); ++fedItr) {
      int fedID=FedsInIds[fedItr];
      //make sure fed id is in allowed range  
      //cout<<fedID<<endl;   
      if(fedID>=FEDRange_.first && fedID<=FEDRange_.second){
        ++FedCount;
	if(fedID>=0 && fedID<=31){
	  ++FedCountBarrel;
	  if((fedID>=0 && fedID<=3) || (fedID>=12 && fedID<=15)) ++FedCountShellpI;
	  if(fedID>=4 && fedID<=11) ++FedCountShellpO;
	  if((fedID>=16 && fedID<=19) || (fedID>=28 && fedID<=31)) ++FedCountShellmO;
	  if(fedID>=20 && fedID<=27) ++FedCountShellmI;
	}else if(fedID>=32 && fedID<=39){
	  ++FedCountEndcap;
	  if(fedID==32 || fedID==35) ++FedCountHalfCylinderpI;
	  if(fedID==33 || fedID==34) ++FedCountHalfCylinderpO;
	  if(fedID==36 || fedID==39) ++FedCountHalfCylindermO;
	  if(fedID==37 || fedID==38) ++FedCountHalfCylindermI;
	}
      }
    }   

    //Fill active fed fraction ME
    if(NumberOfFeds_>0){
      //all Pixel:
      DaqFraction_->Fill( FedCount/NumberOfFeds_);
      //Barrel:
      DaqFractionBarrel_->Fill( FedCountBarrel/32.);
      //ShellmI:
      DaqFractionShellmI_->Fill( FedCountShellmI/8.);
      //ShellmO:
      DaqFractionShellmO_->Fill( FedCountShellmO/8.);
      //ShellpI:
      DaqFractionShellpI_->Fill( FedCountShellpI/8.);
      //ShellpO:
      DaqFractionShellpO_->Fill( FedCountShellpO/8.);
      //Endcap:
      DaqFractionEndcap_->Fill( FedCountEndcap/8.);
      //HalfCylindermI:
      DaqFractionHalfCylindermI_->Fill( FedCountHalfCylindermI/2.);
      //HalfCylindermO:
      DaqFractionHalfCylindermO_->Fill( FedCountHalfCylindermO/2.);
      //HalfCylinderpI:
      DaqFractionHalfCylinderpI_->Fill( FedCountHalfCylinderpI/2.);
      //HalfCylinderpO:
      DaqFractionHalfCylinderpO_->Fill( FedCountHalfCylinderpO/2.);
    }else{
      DaqFraction_->Fill(-1);
      DaqFractionBarrel_->Fill(-1);
      DaqFractionShellmI_->Fill(-1);
      DaqFractionShellmO_->Fill(-1);
      DaqFractionShellpI_->Fill(-1);
      DaqFractionShellpO_->Fill(-1);
      DaqFractionEndcap_->Fill(-1);
      DaqFractionHalfCylindermI_->Fill(-1);
      DaqFractionHalfCylindermO_->Fill(-1);
      DaqFractionHalfCylinderpI_->Fill(-1);
      DaqFractionHalfCylinderpO_->Fill(-1);
    } 
    
  }else{      
    DaqFraction_->Fill(-1);               
    DaqFractionBarrel_->Fill(-1);
    DaqFractionShellmI_->Fill(-1);
    DaqFractionShellmO_->Fill(-1);
    DaqFractionShellpI_->Fill(-1);
    DaqFractionShellpO_->Fill(-1);
    DaqFractionEndcap_->Fill(-1);
    DaqFractionHalfCylindermI_->Fill(-1);
    DaqFractionHalfCylindermO_->Fill(-1);
    DaqFractionHalfCylinderpI_->Fill(-1);
    DaqFractionHalfCylinderpO_->Fill(-1);
    return; 
  }
}


void SiPixelDaqInfo::endLuminosityBlock(const edm::LuminosityBlock&  lumiBlock, const  edm::EventSetup& iSetup){}


void SiPixelDaqInfo::beginJob(const edm::EventSetup& iSetup){

  dbe_ = 0;
  dbe_ = Service<DQMStore>().operator->();
  
 
  dbe_->setCurrentFolder("Pixel/EventInfo/DAQContents");
  DaqFraction_= dbe_->bookFloat("PixelDaqFraction");  
  DaqFractionBarrel_= dbe_->bookFloat("PixelBarrelDaqFraction");  
  DaqFractionShellmI_= dbe_->bookFloat("PixelShellmIDaqFraction");  
  DaqFractionShellmO_= dbe_->bookFloat("PixelShellmODaqFraction");  
  DaqFractionShellpI_= dbe_->bookFloat("PixelShellpIDaqFraction");  
  DaqFractionShellpO_= dbe_->bookFloat("PixelShellpODaqFraction");  
  DaqFractionEndcap_= dbe_->bookFloat("PixelEndcapDaqFraction");  
  DaqFractionHalfCylindermI_= dbe_->bookFloat("PixelHalfCylindermIDaqFraction");  
  DaqFractionHalfCylindermO_= dbe_->bookFloat("PixelHalfCylindermODaqFraction");  
  DaqFractionHalfCylinderpI_= dbe_->bookFloat("PixelHalfCylinderpIDaqFraction");  
  DaqFractionHalfCylinderpO_= dbe_->bookFloat("PixelHalfCylinderpODaqFraction");  
}


void SiPixelDaqInfo::endJob() {}



void SiPixelDaqInfo::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup){}

