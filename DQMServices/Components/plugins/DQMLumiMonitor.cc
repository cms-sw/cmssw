/*
 * \file DQMLumiMonitor.cc
 * \author S. Dutta
 * Last Update:
 * $Date: 2012/04/11 07:16:50 $
 * $Revision: 1.2 $
 * $Author: dutta $
 *
 * Description: Pixel Luminosity Monitoring 
 *
*/
#include "DQMServices/Components/plugins/DQMLumiMonitor.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/SiPixelCluster/interface/SiPixelCluster.h"
#include "RecoLuminosity/LumiProducer/interface/DIPLuminosityRcd.h" 
#include "RecoLuminosity/LumiProducer/interface/DIPLumiSummary.h" 
#include "TPRegexp.h"


// -----------------------------
//  constructors and destructor
// -----------------------------

DQMLumiMonitor::DQMLumiMonitor( const edm::ParameterSet& ps ) : parameters_(ps) {


  moduleName_          = parameters_.getParameter<std::string>("ModuleName");
  folderName_          = parameters_.getParameter<std::string>("FolderName");
  pixelClusterInputTag_= parameters_.getParameter<edm::InputTag>("PixelClusterInputTag");
  lumiRecordName_      = parameters_.getParameter<std::string>("LumiRecordName");

  nClusME_ = 0;
  nClusVsLSME_ = 0;
  intLumiVsLSME_= 0;  
  corrIntLumiAndClusVsLSME_ = 0;
}

DQMLumiMonitor::~DQMLumiMonitor() {

}

void DQMLumiMonitor::bookHistograms() {
    
  edm::ParameterSet ClusHistoPar =  parameters_.getParameter<edm::ParameterSet>("TH1ClusPar");
  edm::ParameterSet LumiHistoPar =  parameters_.getParameter<edm::ParameterSet>("TH1LumiPar");
  edm::ParameterSet LumiSecHistoPar =  parameters_.getParameter<edm::ParameterSet>("TH1LSPar");

  std::string currentFolder = moduleName_ + "/" + folderName_ ;
  dbe_->setCurrentFolder(currentFolder.c_str());

  if (nClusME_ == 0) nClusME_ = dbe_->book1D("nPlxClus", " Number of Pixel Clusters ",
					     ClusHistoPar.getParameter<int32_t>("Xbins"),
					     ClusHistoPar.getParameter<double>("Xmin"),
					     ClusHistoPar.getParameter<double>("Xmax"));
  else nClusME_->Reset();
  if (nClusVsLSME_ == 0) nClusVsLSME_ = dbe_->bookProfile("nClusVsLS", " Number of Pixel Cluster Vs LS number",
							      LumiSecHistoPar.getParameter<int32_t>("Xbins"),
							      LumiSecHistoPar.getParameter<double>("Xmin"),
							      LumiSecHistoPar.getParameter<double>("Xmax"),
							  0.0, 0.0, "");
  else nClusVsLSME_->Reset();
  if (intLumiVsLSME_ == 0) intLumiVsLSME_ = dbe_->bookProfile("intLumiVsLS", " Integrated Luminosity Vs LS number",
							      LumiSecHistoPar.getParameter<int32_t>("Xbins"),
							      LumiSecHistoPar.getParameter<double>("Xmin"),
							      LumiSecHistoPar.getParameter<double>("Xmax"),
							      0.0, 0.0, "");
  else intLumiVsLSME_->Reset();

  if (corrIntLumiAndClusVsLSME_== 0) corrIntLumiAndClusVsLSME_ = dbe_->bookProfile2D("corrIntLumiAndClusVsLS", " Correlation of nCluster and Integrated Luminosity Vs LS number",
										     LumiSecHistoPar.getParameter<int32_t>("Xbins"),
										     LumiSecHistoPar.getParameter<double>("Xmin"),
										     LumiSecHistoPar.getParameter<double>("Xmax"),
										     LumiHistoPar.getParameter<int32_t>("Xbins"),
										     LumiHistoPar.getParameter<double>("Xmin"),
										     LumiHistoPar.getParameter<double>("Xmax"),
										     0.0, 0.0);
  else corrIntLumiAndClusVsLSME_->Reset();
}
void DQMLumiMonitor::beginJob() {
  dbe_ = edm::Service<DQMStore>().operator->();
  intLumi_  = -1.0; 
  nLumi_ = -1;
}

void DQMLumiMonitor::beginRun(edm::Run const& iRun, edm::EventSetup const& iSetup) {
  bookHistograms();

}
void DQMLumiMonitor::analyze(edm::Event const& iEvent, edm::EventSetup const& iSetup)  {

  //Access Pixel Clusters
  edm::Handle< SiPixelClusterCollectionNew > siPixelClusters;
  iEvent.getByLabel(pixelClusterInputTag_, siPixelClusters);
  
  if(!siPixelClusters.isValid()) {
    edm::LogError("PixelLumiMonotor") << "Could not find Cluster Collection " << pixelClusterInputTag_;
    return;
  }
  nClusME_->Fill(siPixelClusters->size());
  if (nLumi_ != -1) nClusVsLSME_->Fill(nLumi_, siPixelClusters->size());
  if (intLumi_ != -1 || nLumi_ != -1) corrIntLumiAndClusVsLSME_->Fill(nLumi_, intLumi_, siPixelClusters->size());
}

void DQMLumiMonitor::endLuminosityBlock(edm::LuminosityBlock const& lumiBlock, edm::EventSetup const& eSetup){
  std::cout <<" Run Number "<<lumiBlock.run() <<" Lumi Section Numnber "<< lumiBlock.luminosityBlock()<<std::endl;
  nLumi_ = lumiBlock.luminosityBlock();
  edm::eventsetup::EventSetupRecordKey recordKey(edm::eventsetup::EventSetupRecordKey::TypeTag::findType(lumiRecordName_));
  
  if( recordKey.type() == edm::eventsetup::EventSetupRecordKey::TypeTag()) {
    std::cout <<"Record \"DIPLuminosityRcd"<<"\" does not exist "<<std::endl;
    return;
  }
  edm::ESHandle<DIPLumiSummary> datahandle;
  eSetup.getData(datahandle);
  if(datahandle.isValid()){
    const DIPLumiSummary* dipLumiSummary_=datahandle.product();
    if (!dipLumiSummary_->isNull()) {
      intLumi_ = dipLumiSummary_->intgDelLumiByLS();
      edm::LogInfo("PixelLumiMonotor") <<" Luminosity in this Lumi Section " << intLumi_ ;
      intLumiVsLSME_->Fill(nLumi_, intLumi_);
    }else{  
      edm::LogError("PixelLumiMonotor") << "Empty data found !";
    } 
  } else {
    edm::LogError("PixelLumiMonotor") << "No valid data found!";
  }
}

void DQMLumiMonitor::endRun(edm::Run const& iRun, edm::EventSetup const& iSetup) {

}


void DQMLumiMonitor::endJob() {

}
// Define this as a plug-in
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(DQMLumiMonitor);
