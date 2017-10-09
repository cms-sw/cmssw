#include "DQMServices/Components/src/DQMDaqInfo.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"

DQMDaqInfo::DQMDaqInfo(const edm::ParameterSet& iConfig)  
{   
}

DQMDaqInfo::~DQMDaqInfo()
{  
}

void DQMDaqInfo::beginLuminosityBlock(const edm::LuminosityBlock& lumiBlock, const  edm::EventSetup& iSetup){
  
  edm::eventsetup::EventSetupRecordKey recordKey(edm::eventsetup::EventSetupRecordKey::TypeTag::findType("RunInfoRcd"));
  
  if( 0 != iSetup.find( recordKey ) ) {
    
    edm::ESHandle<RunInfo> sumFED;
    iSetup.get<RunInfoRcd>().get(sumFED);    
   
    //const RunInfo* summaryFED=sumFED.product();
  
    std::vector<int> FedsInIds= sumFED->m_fed_in;   

    float  FedCount[9]={0., 0., 0., 0., 0., 0., 0., 0., 0.};
    
    for(unsigned int fedItr=0;fedItr<FedsInIds.size(); ++fedItr) {
      int fedID=FedsInIds[fedItr];     

      if(fedID>=PixelRange.first             &&  fedID<=PixelRange.second)        ++FedCount[Pixel];
      if(fedID>=TrackerRange.first           &&  fedID<=TrackerRange.second)      ++FedCount[SiStrip];
      if(fedID>=CSCRange.first               &&  fedID<=CSCRange.second)          ++FedCount[CSC];
      if(fedID>=RPCRange.first               &&  fedID<=RPCRange.second)          ++FedCount[RPC];
      if(fedID>=DTRange.first                &&  fedID<=DTRange.second)           ++FedCount[DT];
      if(fedID>=HcalRange.first              &&  fedID<=HcalRange.second)         ++FedCount[Hcal];       
      if(fedID>=ECALBarrRange.first          &&  fedID<=ECALBarrRange.second)     ++FedCount[EcalBarrel];      
      if((fedID>=ECALEndcapRangeLow.first    && fedID<=ECALEndcapRangeLow.second) ||
	 (fedID>=ECALEndcapRangeHigh.first && fedID<=ECALEndcapRangeHigh.second)) ++FedCount[EcalEndcap];
      if(fedID>=L1TRange.first               &&  fedID<=L1TRange.second)          ++FedCount[L1T];
    
    }   
    
    for(int detIndex=0; detIndex<9; ++detIndex) { 
      DaqFraction[detIndex]->Fill( FedCount[detIndex]/NumberOfFeds[detIndex]);
    }

  }else{    
  
    for(int detIndex=0; detIndex<9; ++detIndex)  DaqFraction[detIndex]->Fill(-1);               
    return; 
  }
  
 
}


void DQMDaqInfo::endLuminosityBlock(const edm::LuminosityBlock&  lumiBlock, const  edm::EventSetup& iSetup){
}


void 
DQMDaqInfo::beginJob()
{
  dbe_ = 0;
  dbe_ = edm::Service<DQMStore>().operator->();
  
  std::string commonFolder = "/EventInfo/DAQContents";  
  std::string subsystFolder;
  std::string curentFolder;
  
  subsystFolder="Pixel";  
  curentFolder= subsystFolder+commonFolder;
  dbe_->setCurrentFolder(curentFolder.c_str());
  DaqFraction[Pixel]   = dbe_->bookFloat("PixelDaqFraction");
  

  subsystFolder="SiStrip";  
  curentFolder=subsystFolder+commonFolder;
  dbe_->setCurrentFolder(curentFolder.c_str());
  DaqFraction[SiStrip]    = dbe_->bookFloat("SiStripDaqFraction");
  
  subsystFolder="RPC";  
  curentFolder=subsystFolder+commonFolder;
  dbe_->setCurrentFolder(curentFolder.c_str());
  DaqFraction[RPC]        = dbe_->bookFloat("RPCDaqFraction");
  
  subsystFolder="CSC";  
  curentFolder=subsystFolder+commonFolder;
  dbe_->setCurrentFolder(curentFolder.c_str());
  DaqFraction[CSC]       = dbe_->bookFloat("CSCDaqFraction");

  subsystFolder="DT";  
  curentFolder=subsystFolder+commonFolder;
  dbe_->setCurrentFolder(curentFolder.c_str());
  DaqFraction[DT]         = dbe_->bookFloat("DTDaqFraction");

  subsystFolder="Hcal";  
  curentFolder=subsystFolder+commonFolder;
  dbe_->setCurrentFolder(curentFolder.c_str());
  DaqFraction[Hcal]       = dbe_->bookFloat("HcalDaqFraction");

  subsystFolder="EcalBarrel";  
  curentFolder=subsystFolder+commonFolder;
  dbe_->setCurrentFolder(curentFolder.c_str());
  DaqFraction[EcalBarrel]       = dbe_->bookFloat("EcalBarrDaqFraction");

  subsystFolder="EcalEndcap";  
  curentFolder=subsystFolder+commonFolder;
  dbe_->setCurrentFolder(curentFolder.c_str());
  DaqFraction[EcalEndcap]       = dbe_->bookFloat("EcalEndDaqFraction");

  subsystFolder="L1T";  
  curentFolder=subsystFolder+commonFolder;
  dbe_->setCurrentFolder(curentFolder.c_str());
  DaqFraction[L1T]       = dbe_->bookFloat("L1TDaqFraction");


  PixelRange.first  = FEDNumbering::MINSiPixelFEDID;
  PixelRange.second = FEDNumbering::MAXSiPixelFEDID;
  TrackerRange.first = FEDNumbering::MINSiStripFEDID;
  TrackerRange.second = FEDNumbering::MAXSiStripFEDID;
  CSCRange.first  = FEDNumbering::MINCSCFEDID;
  CSCRange.second = FEDNumbering::MAXCSCFEDID;
  RPCRange.first  = 790;
  RPCRange.second = 792;
  DTRange.first   = 770;
  DTRange.second  = 774;
  HcalRange.first  = FEDNumbering::MINHCALFEDID;
  HcalRange.second = FEDNumbering::MAXHCALFEDID;
  L1TRange.first  = FEDNumbering::MINTriggerGTPFEDID;
  L1TRange.second = FEDNumbering::MAXTriggerGTPFEDID;
  ECALBarrRange.first  = 610;    
  ECALBarrRange.second = 645;
  ECALEndcapRangeLow.first   = 601;
  ECALEndcapRangeLow.second  = 609;
  ECALEndcapRangeHigh.first  = 646;
  ECALEndcapRangeHigh.second = 654;

  NumberOfFeds[Pixel]   = PixelRange.second-PixelRange.first +1;
  NumberOfFeds[SiStrip] = TrackerRange.second-TrackerRange.first +1;
  NumberOfFeds[CSC]     = CSCRange.second-CSCRange.first  +1;
  NumberOfFeds[RPC]     = RPCRange.second-RPCRange.first  +1;
  NumberOfFeds[DT]      = DTRange.second-DTRange.first +1;
  NumberOfFeds[Hcal]    = HcalRange.second-HcalRange.first +1;  
  NumberOfFeds[EcalBarrel]    = ECALBarrRange.second-ECALBarrRange.first +1 ;
  NumberOfFeds[EcalEndcap]    = (ECALEndcapRangeLow.second-ECALEndcapRangeLow.first +1)+(ECALEndcapRangeHigh.second-ECALEndcapRangeHigh.first +1) ;
  NumberOfFeds[L1T]    = L1TRange.second-L1TRange.first +1;

}


void 
DQMDaqInfo::endJob() {
}



void
DQMDaqInfo::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{ 
 

}

