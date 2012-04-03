/*
 * \file DQMLumiMonitor.cc
 * \author S. Dutta
 * Last Update:
 * $Date: 2012/03/29 06:21:16 $
 * $Revision: 1.1 $
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
  intLumiVsLSME_-= 0;  
}

DQMLumiMonitor::~DQMLumiMonitor() {

}

void DQMLumiMonitor::bookHistograms() {
    
  edm::ParameterSet ClusHistoPar =  parameters_.getParameter<edm::ParameterSet>("TH1ClusPar");

  std::string currentFolder = moduleName_ + "/" + folderName_ ;
  dbe_->setCurrentFolder(currentFolder.c_str());

  if (nClusME_ == 0) {
    nClusME_ = dbe_->book1D("nPlxClus", " Number of Pixel Clusters ",
			    ClusHistoPar.getParameter<int32_t>("Xbins"),
			    ClusHistoPar.getParameter<double>("Xmin"),
			    ClusHistoPar.getParameter<double>("Xmax"));
    intLumiVsLSME_ = dbe_->bookProfile("intLumiVsLS", " Integrated Luminosity Vs LS number",
				       2001, -0.5, 2000.5, 
				       ClusHistoPar.getParameter<int32_t>("Xbins"),
				       ClusHistoPar.getParameter<double>("Xmin"),
				       ClusHistoPar.getParameter<double>("Xmax"));
  } else {
    nClusME_->Reset();
    intLumiVsLSME_->Reset();
  }
}
void DQMLumiMonitor::beginJob() {
  dbe_ = edm::Service<DQMStore>().operator->();
  _intLumi  = -1.0; 
}

void DQMLumiMonitor::beginRun(edm::Run const& iRun, edm::EventSetup const& iSetup) {
  bookHistograms();

}
void DQMLumiMonitor::analyze(edm::Event const& iEvent, edm::EventSetup const& iSetup)  {

  //Access Pixel Clusters
  /*  edm::Handle< SiPixelClusterCollectionNew > siPixelClusters;
  iEvent.getByLabel(pixelClusterInputTag_, siPixelClusters);
  
  if(!siPixelClusters.isValid()) {
    edm::LogError("PixelLumiMonotor") << "Could not find Cluster Collection " << pixelClusterInputTag_;
    return;
  }
  nClusME_->Fill(siPixelClusters->size());  */
}

void DQMLumiMonitor::endLuminosityBlock(edm::LuminosityBlock const& lumiBlock, edm::EventSetup const& eSetup){
  std::cout <<" Run Number "<<lumiBlock.run() <<" Lumi Section Numnber "<< lumiBlock.luminosityBlock()<<std::endl;
 
  edm::eventsetup::EventSetupRecordKey recordKey(edm::eventsetup::EventSetupRecordKey::TypeTag::findType(lumiRecordName_));
  
  if( recordKey.type() == edm::eventsetup::EventSetupRecordKey::TypeTag()) {
    std::cout <<"Record \"DIPLuminosityRcd"<<"\" does not exist "<<std::endl;
    return;
  }
  edm::ESHandle<DIPLumiSummary> datahandle;
  eSetup.getData(datahandle);
  if(datahandle.isValid()){
    const DIPLumiSummary* _dipLumiSummary=datahandle.product();
    if (!_dipLumiSummary->isNull()) {
      std::cout<<" Luminosity in this Lumi Section " << _dipLumiSummary->intgDelLumiByLS() <<std::endl;
    }else{
      std::cout<<"empty data found"<<std::endl;
    } 
  } else {
    std::cout<<"no valid data found"<<std::endl;
  }
}

void DQMLumiMonitor::endRun(edm::Run const& iRun, edm::EventSetup const& iSetup) {

}


void DQMLumiMonitor::endJob() {

}
// Define this as a plug-in
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(DQMLumiMonitor);
