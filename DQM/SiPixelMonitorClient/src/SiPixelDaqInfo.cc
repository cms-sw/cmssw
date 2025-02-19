#include "DQM/SiPixelMonitorClient/interface/SiPixelDaqInfo.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/FEDRawData/interface/FEDRawData.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
#include "CondFormats/RunInfo/interface/RunInfo.h"
#include "CondFormats/RunInfo/interface/RunSummary.h"
#include "CondFormats/DataRecord/interface/RunSummaryRcd.h"

using namespace std;
using namespace edm;
SiPixelDaqInfo::SiPixelDaqInfo(const edm::ParameterSet& ps) {
 
  FEDRange_.first  = ps.getUntrackedParameter<unsigned int>("MinimumPixelFEDId", 0);
  FEDRange_.second = ps.getUntrackedParameter<unsigned int>("MaximumPixelFEDId", 39);
  daqSource_       = ps.getUntrackedParameter<string>("daqSource",  "source");

  NumberOfFeds_ =FEDRange_.second -  FEDRange_.first +1;
  
  NEvents_ = 0;
  for(int i=0; i!=40; i++) FEDs_[i] = 0;

}

SiPixelDaqInfo::~SiPixelDaqInfo(){}

void SiPixelDaqInfo::beginLuminosityBlock(const LuminosityBlock& lumiBlock, const  EventSetup& iSetup){}


void SiPixelDaqInfo::endLuminosityBlock(const edm::LuminosityBlock&  lumiBlock, const  edm::EventSetup& iSetup){
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

    //loop on all active feds
    for(unsigned int fedItr=0;fedItr<FedsInIds.size(); ++fedItr) {
      int fedID=FedsInIds[fedItr];
      //make sure fed id is in allowed range  
      //cout<<fedID<<endl;   
      if(fedID>=FEDRange_.first && fedID<=FEDRange_.second){
        ++FedCount;
	if(fedID>=0 && fedID<=31) ++FedCountBarrel;
	else if(fedID>=32 && fedID<=39) ++FedCountEndcap;
      }
    }   
    
    //Fill active fed fraction ME
    if(FedCountBarrel<=32){
      FedCountBarrel = 0; FedCountEndcap = 0; FedCount = 0; NumberOfFeds_ = 40;
      for(int i=0; i!=40; i++){
        if(i<=31 && FEDs_[i]>0) FedCountBarrel++;
	if(i>=32 && FEDs_[i]>0) FedCountEndcap++;
	if(FEDs_[i]>0) FedCount++;
      }
    }
    if(NumberOfFeds_>0){
      //all Pixel:
      Fraction_->Fill( FedCount/NumberOfFeds_);
      //Barrel:
      FractionBarrel_->Fill( FedCountBarrel/32.);
      //Endcap:
      FractionEndcap_->Fill( FedCountEndcap/8.);
    }else{
      Fraction_->Fill(-1);
      FractionBarrel_->Fill(-1);
      FractionEndcap_->Fill(-1);
    } 
    
  }else{      
    Fraction_->Fill(-1);	       
    FractionBarrel_->Fill(-1);
    FractionEndcap_->Fill(-1);
    return; 
  }
}

void SiPixelDaqInfo::endRun(const edm::Run&  r, const  edm::EventSetup& iSetup){
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

    //loop on all active feds
    for(unsigned int fedItr=0;fedItr<FedsInIds.size(); ++fedItr) {
      int fedID=FedsInIds[fedItr];
      //make sure fed id is in allowed range  
      //cout<<fedID<<endl;   
      if(fedID>=FEDRange_.first && fedID<=FEDRange_.second){
        ++FedCount;
	if(fedID>=0 && fedID<=31) ++FedCountBarrel;
	else if(fedID>=32 && fedID<=39) ++FedCountEndcap;
      }
    }   

    if(FedCountBarrel>32){
      FedCountBarrel = nFEDsBarrel_;
      FedCountEndcap = nFEDsEndcap_;
      FedCount = FedCountBarrel + FedCountEndcap;
      NumberOfFeds_ = 40;
    }

    //Fill active fed fraction ME
    if(FedCountBarrel<=32){
      FedCountBarrel = 0; FedCountEndcap = 0; FedCount = 0; NumberOfFeds_ = 40;
      for(int i=0; i!=40; i++){
        if(i<=31 && FEDs_[i]>0) FedCountBarrel++;
	if(i>=32 && FEDs_[i]>0) FedCountEndcap++;
	if(FEDs_[i]>0) FedCount++;
      }
    }
    if(NumberOfFeds_>0){
      //all Pixel:
      Fraction_->Fill( FedCount/NumberOfFeds_);
      //Barrel:
      FractionBarrel_->Fill( FedCountBarrel/32.);
      //Endcap:
      FractionEndcap_->Fill( FedCountEndcap/8.);
    }else{
      Fraction_->Fill(-1);
      FractionBarrel_->Fill(-1);
      FractionEndcap_->Fill(-1);
    } 
    
  }else{      
    Fraction_->Fill(-1);	       
    FractionBarrel_->Fill(-1);
    FractionEndcap_->Fill(-1);
    return; 
  }
}


void SiPixelDaqInfo::beginJob(){

  dbe_ = 0;
  dbe_ = Service<DQMStore>().operator->();
  
 
  dbe_->setCurrentFolder("Pixel/EventInfo");
  Fraction_= dbe_->bookFloat("DAQSummary");  
  dbe_->setCurrentFolder("Pixel/EventInfo/DAQContents");
  FractionBarrel_= dbe_->bookFloat("PixelBarrelFraction");  
  FractionEndcap_= dbe_->bookFloat("PixelEndcapFraction");  
}


void SiPixelDaqInfo::endJob() {}



void SiPixelDaqInfo::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup){
  NEvents_++;  
  //cout<<"in SiPixelDaqInfo::analyze now!"<<endl;
  if(NEvents_>=1 && NEvents_<=100){
    // check if any Pixel FED is in readout:
    edm::Handle<FEDRawDataCollection> rawDataHandle;
    iEvent.getByLabel(daqSource_, rawDataHandle);
    if(!rawDataHandle.isValid()){
      edm::LogInfo("SiPixelDaqInfo") << daqSource_ << " is empty!";
      return;
    }
    const FEDRawDataCollection& rawDataCollection = *rawDataHandle;
    nFEDsBarrel_ = 0; nFEDsEndcap_ = 0;
    for(int i = 0; i != 40; i++){
      if(rawDataCollection.FEDData(i).size() > 208 ) FEDs_[i]++;
    }
  }

}

