#include "DQMServices/Components/src/DQMDaqInfo.h"


DQMDaqInfo::DQMDaqInfo(const edm::ParameterSet& iConfig)  
{
  
  /// Temporary  txt file for cross checks, will be removed
  saveDCFile_=iConfig.getUntrackedParameter("saveDCFile",false);
  if(saveDCFile_){
    outputFile_=iConfig.getParameter<std::string>("outputFile");
    dataCertificationFile.open(outputFile_.c_str());
    dataCertificationFile<<" Run Number  |  Luminosity Section "<<std::endl;
  }
  
  /// Standard output root file 
  saveData = iConfig.getParameter<bool>("saveRootFile");
  if(saveData) outputFileName = iConfig.getParameter<std::string>("OutputFileName");
   
}


DQMDaqInfo::~DQMDaqInfo()
{  
  dataCertificationFile.close();
}


void DQMDaqInfo::beginLuminosityBlock(const edm::LuminosityBlock& lumiBlock, const  edm::EventSetup& iSetup){
  
  edm::eventsetup::EventSetupRecordKey recordKey(edm::eventsetup::EventSetupRecordKey::TypeTag::findType("RunSummaryRcd"));
    
  edm::ESHandle<RunSummary> sum;
  iSetup.get<RunSummaryRcd>().get(sum);
  const RunSummary* summary=sum.product();
  std::vector<int> SubDetId= summary->m_subdt_in;    
  std::vector<std::string> subdet = summary->getSubdtIn();
    
  for(int det=0;det<7;++det) DaqFraction[det]->Fill(0.);
  
  if(saveDCFile_)  dataCertificationFile<<"\n"<<  lumiBlock.id().run() <<"  |  "<<lumiBlock.luminosityBlock()  <<std::endl;
  
  for (size_t itrSubDet=0; itrSubDet<subdet.size(); itrSubDet++){
    if(saveDCFile_)  dataCertificationFile<<SubDetId[itrSubDet]<< " "<<subdet[itrSubDet] << std::endl;
    DaqFraction[SubDetId[itrSubDet]]->Fill(1.);
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
  if(saveData) dbe_->save(outputFileName);
}



void
DQMDaqInfo::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{ 
}


//define this as a plug-in
//DEFINE_FWK_MODULE(DQMDaqInfo);
