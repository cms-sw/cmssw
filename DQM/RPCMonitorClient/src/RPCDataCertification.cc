#include "DQM/RPCMonitorClient/interface/RPCDataCertification.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
//CondFormats
#include "CondFormats/RunInfo/interface/RunInfo.h"
#include "CondFormats/DataRecord/interface/RunSummaryRcd.h"


RPCDataCertification::RPCDataCertification(const edm::ParameterSet& ps) {

 numberOfDisks_ = ps.getUntrackedParameter<int>("NumberOfEndcapDisks", 4);

 FEDRange_.first  = ps.getUntrackedParameter<unsigned int>("MinimumRPCFEDId", 790);
 FEDRange_.second = ps.getUntrackedParameter<unsigned int>("MaximumRPCFEDId", 792);
 
 NumberOfFeds_ =FEDRange_.second -  FEDRange_.first +1;
}

RPCDataCertification::~RPCDataCertification() {}


void RPCDataCertification::beginJob(){}


  void RPCDataCertification::beginRun(const edm::Run& r, const edm::EventSetup& setup){
 edm::eventsetup::EventSetupRecordKey recordKey(edm::eventsetup::EventSetupRecordKey::TypeTag::findType("RunInfoRcd"));

 int defaultValue = 1;

 if(0 != setup.find( recordKey ) ) {
    defaultValue = -1;
    //get fed summary information
    edm::ESHandle<RunInfo> sumFED;
    setup.get<RunInfoRcd>().get(sumFED);    
    std::vector<int> FedsInIds= sumFED->m_fed_in;   
    unsigned int f = 0;
    bool flag = false;
    while(!flag && f < FedsInIds.size()) {
      int fedID=FedsInIds[f];
      //make sure fed id is in allowed range  
      if(fedID>=FEDRange_.first && fedID<=FEDRange_.second) {
	defaultValue = 1;
	flag = true;
      } 
      f++;
    }   
  }   


  // get the DQMStore
 theDbe = edm::Service<DQMStore>().operator->();
  
  theDbe->setCurrentFolder("RPC/EventInfo");
  // global fraction
  totalCertFraction = theDbe->bookFloat("CertificationSummary");  
  totalCertFraction->Fill(defaultValue);

  CertMap_ = theDbe->book2D( "CertificationSummaryMap","RPC Certification Summary Map",15, -7.5, 7.5, 12, 0.5 ,12.5);
  
  //customize the 2d histo
  std::stringstream BinLabel;
  for (int i= 1 ; i<13; i++){
    BinLabel.str("");
    BinLabel<<"Sec"<<i;
    CertMap_->setBinLabel(i,BinLabel.str(),2);
  } 

  for(int i = -2 ; i <=2; i++){
    BinLabel.str("");
    BinLabel<<"Wheel"<<i;
    CertMap_->setBinLabel(i+8,BinLabel.str(),1);
  }


 for(int i = 1; i<=numberOfDisks_; i++){
    BinLabel.str("");
    BinLabel<<"Disk"<<i;
    CertMap_->setBinLabel((i+11),BinLabel.str(),1);
    BinLabel.str("");
    BinLabel<<"Disk"<<-i;
    CertMap_->setBinLabel((-i+5),BinLabel.str(),1);
  }
  //fill the histo with "1" --- just for the moment
  for(int i=1; i<=15; i++){
     for (int j=1; j<=12; j++ ){
       if(i==5 || i==11 || (j>6 && (i<6 || i>10)))    
	 CertMap_->setBinContent(i,j,-1);//bins that not correspond to subdetector parts
       else
	 CertMap_->setBinContent(i,j,defaultValue);
     }
   }


  if(numberOfDisks_ < 4){
    for (int j=1; j<=12; j++ ){
      CertMap_->setBinContent(1,j,-1);//bins that not correspond to subdetector parts
      CertMap_->setBinContent(15,j,-1);
    }
 }
  // book the ME
  theDbe->setCurrentFolder("RPC/EventInfo/CertificationContents");

  int limit = numberOfDisks_;
  if(numberOfDisks_ < 2) limit = 2;
  

  for (int i = -1 * limit; i<= limit;i++ ){//loop on wheels and disks
    if (i>-3 && i<3){//wheels
      std::stringstream streams;
      streams << "RPC_Wheel" << i;
      certWheelFractions[i+2] = theDbe->bookFloat(streams.str());
      certWheelFractions[i+2]->Fill(defaultValue);
   }
    
    if (i == 0  || i > numberOfDisks_ || i< (-1 * numberOfDisks_))continue;
    
    int offset = numberOfDisks_;
    if (i>0) offset --; //used to skip case equale to zero
    std::stringstream streams;
    streams << "RPC_Disk" << i;
    certDiskFractions[i+2] = theDbe->bookFloat(streams.str());
    certDiskFractions[i+2]->Fill(defaultValue);
  }
}

void RPCDataCertification::beginLuminosityBlock(const edm::LuminosityBlock& lumi, const  edm::EventSetup& setup) {}


void RPCDataCertification::endLuminosityBlock(const edm::LuminosityBlock&  lumi, const  edm::EventSetup& setup){}


void RPCDataCertification::endJob() {}



void RPCDataCertification::analyze(const edm::Event& event, const edm::EventSetup& setup){}





