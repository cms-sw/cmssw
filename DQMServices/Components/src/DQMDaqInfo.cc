#include "DQMServices/Components/src/DQMDaqInfo.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"

DQMDaqInfo::DQMDaqInfo(const edm::ParameterSet& iConfig)  
{
  
  /// Temporary  txt file for cross checks, will be removed
  saveDCFile_=iConfig.getUntrackedParameter("saveDCFile",false);
  if(saveDCFile_){
    outputFile_=iConfig.getParameter<std::string>("outputFile");
    dataCertificationFile.open(outputFile_.c_str());
    dataCertificationFile<<" Run Number  |  Luminosity Section "<<std::endl;
  }
  FedGranularity = iConfig.getUntrackedParameter("FedGranularityInput",false);
  
  /// Standard output root file 
   
}


DQMDaqInfo::~DQMDaqInfo()
{  
  dataCertificationFile.close();
}


void DQMDaqInfo::beginLuminosityBlock(const edm::LuminosityBlock& lumiBlock, const  edm::EventSetup& iSetup){
  
  if(saveDCFile_)  dataCertificationFile<<"\n"<<  lumiBlock.id().run() <<"  |  "<<lumiBlock.luminosityBlock()  <<std::endl;
  
 
   
  if(FedGranularity){
    
    edm::eventsetup::EventSetupRecordKey recordKey2(edm::eventsetup::EventSetupRecordKey::TypeTag::findType("RunInfoRcd"));
    if( recordKey2.type() == edm::eventsetup::EventSetupRecordKey::TypeTag() ) {

     for(int detIndex=0; detIndex<7; ++detIndex) { 
    	DaqFraction[detIndex]->Fill(-1);           
      }
      return;
    }
    
    
    edm::ESHandle<RunInfo> sumFED;
    iSetup.get<RunInfoRcd>().get(sumFED);    
	
    if (!sumFED.isValid()) {
      for(int detIndex=0; detIndex<7; ++detIndex) { 
	DaqFraction[detIndex]->Fill(-1);           
      }
      return;
    }

    //const RunInfo* summaryFED=sumFED.product();

    std::vector<int> FedsInIds= sumFED->m_fed_in;   
    
    
    if(saveDCFile_)  dataCertificationFile<<"\n Number of FEDS in:"<<  FedsInIds.size()   <<std::endl;
    
    std::pair<int,int> PixelRange   = FEDNumbering::getSiPixelFEDIds();;
    std::pair<int,int> TrackerRange = FEDNumbering::getSiStripFEDIds();
    std::pair<int,int> CSCRange     = FEDNumbering::getCSCFEDIds();
    std::pair<int,int> RPCRange     = FEDNumbering::getRPCFEDIds();
    std::pair<int,int> DTRange      = FEDNumbering::getDTFEDIds();
    std::pair<int,int> ECALRange    = FEDNumbering::getEcalFEDIds();
    std::pair<int,int> HCALRange    = FEDNumbering::getHcalFEDIds();

    float  FedCount[7]={0.,0.,0.,0.,0.,0.,0.};
    float  NumberOfFeds[7];
    NumberOfFeds[Pixel]   = PixelRange.second-PixelRange.first;
    NumberOfFeds[SiStrip] = TrackerRange.second-TrackerRange.first;
    NumberOfFeds[CSC]     = CSCRange.second-CSCRange.first ;
    NumberOfFeds[RPC]     = RPCRange.second-RPCRange.first ;
    NumberOfFeds[DT]      = DTRange.second-DTRange.first;
    NumberOfFeds[ECAL]    = ECALRange.second-ECALRange.first ;
    NumberOfFeds[HCAL]    = HCALRange.second-HCALRange.first;
   
    for(unsigned int fedItr=0;fedItr<FedsInIds.size(); ++fedItr) {
      int fedID=FedsInIds[fedItr];
      if(fedID>PixelRange.first   &  fedID<=PixelRange.second)    ++FedCount[Pixel]  ;
      if(fedID>TrackerRange.first &  fedID<=TrackerRange.second)  ++FedCount[SiStrip];
      if(fedID>CSCRange.first     &  fedID<=CSCRange.second)      ++FedCount[CSC]    ;
      if(fedID>RPCRange.first     &  fedID<=RPCRange.second)      ++FedCount[RPC]    ;
      if(fedID>DTRange.first      &  fedID<=DTRange.second)       ++FedCount[DT]     ;
      if(fedID>ECALRange.first    &  fedID<=ECALRange.second)     ++FedCount[ECAL]   ;
      if(fedID>HCALRange.first    &  fedID<=HCALRange.second)     ++FedCount[HCAL]	; 
	
    }   
    
    for(int detIndex=0; detIndex<7; ++detIndex) { 
      DaqFraction[detIndex]->Fill( FedCount[detIndex]/NumberOfFeds[detIndex]);
      if(saveDCFile_)  dataCertificationFile<<"subdet " << detIndex<< " "<<FedCount[detIndex]/NumberOfFeds[detIndex]<<std::endl;
       
      
    }

    
  }else{
    
    
    edm::eventsetup::EventSetupRecordKey recordKey(edm::eventsetup::EventSetupRecordKey::TypeTag::findType("RunSummaryRcd"));
    if( recordKey.type() == edm::eventsetup::EventSetupRecordKey::TypeTag()  ) {
      for(int detIndex=0; detIndex<7; ++detIndex) { 
    	DaqFraction[detIndex]->Fill(-1);           
      }
      return;
    }
    
    edm::ESHandle<RunSummary> sum;
    iSetup.get<RunSummaryRcd>().get(sum);
    
    if (!sum.isValid()) {
      for(int detIndex=0; detIndex<7; ++detIndex) { 
    	DaqFraction[detIndex]->Fill(-1);           
      }
      return;
    }
    const RunSummary* summary=sum.product();
    std::vector<int> SubDetId= summary->m_subdt_in;    
    std::vector<std::string> subdet = summary->getSubdtIn();
    
    
    for(int det=0;det<7;++det) DaqFraction[det]->Fill(0.);
    
    
    for (size_t itrSubDet=0; itrSubDet<subdet.size(); itrSubDet++){
      if(saveDCFile_)  dataCertificationFile<<SubDetId[itrSubDet]<< " "<<subdet[itrSubDet] << std::endl;   
      DaqFraction[SubDetId[itrSubDet]]->Fill(1.);
    } 
    
    
    
  }
}


void DQMDaqInfo::endLuminosityBlock(const edm::LuminosityBlock&  lumiBlock, const  edm::EventSetup& iSetup){
}


void 
DQMDaqInfo::beginJob(const edm::EventSetup& iSetup)
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

  subsystFolder="ECAL";  
  curentFolder=subsystFolder+commonFolder;
  dbe_->setCurrentFolder(curentFolder.c_str());
  DaqFraction[ECAL]       = dbe_->bookFloat("ECALDaqFraction");

  subsystFolder="HCAL";  
  curentFolder=subsystFolder+commonFolder;
  dbe_->setCurrentFolder(curentFolder.c_str());
  DaqFraction[HCAL]       = dbe_->bookFloat("HCALDaqFraction");


}


void 
DQMDaqInfo::endJob() {
}



void
DQMDaqInfo::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{ 
 

}


//define this as a plug-in
//DEFINE_FWK_MODULE(DQMDaqInfo);
