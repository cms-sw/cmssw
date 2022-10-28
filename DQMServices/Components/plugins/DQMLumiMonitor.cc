/*
 * \file DQMLumiMonitor.cc
 * \author S. Dutta
 * Last Update:
 *
 * Description: Pixel Luminosity Monitoring 
 *
*/
#include <string>
#include <vector>
#include <map>

#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Luminosity/interface/LumiDetails.h"
#include "DataFormats/Luminosity/interface/LumiSummary.h"
#include "DataFormats/SiPixelCluster/interface/SiPixelCluster.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"

#include "TPRegexp.h"

//
// class declaration
//

class DQMLumiMonitor : public edm::one::EDAnalyzer<edm::one::WatchRuns, edm::one::WatchLuminosityBlocks> {
public:
  typedef dqm::legacy::DQMStore DQMStore;
  typedef dqm::legacy::MonitorElement MonitorElement;
  DQMLumiMonitor(const edm::ParameterSet&);
  ~DQMLumiMonitor() override = default;

protected:
  void beginJob() override;
  void beginRun(edm::Run const& iRun, edm::EventSetup const& iSetup) override;
  void analyze(edm::Event const& iEvent, edm::EventSetup const& iSetup) override;
  void beginLuminosityBlock(edm::LuminosityBlock const& lumiSeg, edm::EventSetup const& eSetup) override;
  void endLuminosityBlock(edm::LuminosityBlock const& lumiSeg, edm::EventSetup const& eSetup) override;
  void endRun(edm::Run const& iRun, edm::EventSetup const& iSetup) override;

private:
  void bookHistograms();

  edm::ParameterSet parameters_;

  std::string moduleName_;
  std::string folderName_;
  edm::EDGetTokenT<edmNew::DetSetVector<SiPixelCluster> > pixelClusterInputTag_;
  edm::EDGetTokenT<LumiSummary> lumiRecordName_;

  DQMStore* dbe_;

  MonitorElement* nClusME_;
  MonitorElement* intLumiVsLSME_;
  MonitorElement* nClusVsLSME_;
  MonitorElement* corrIntLumiAndClusVsLSME_;

  float intLumi_;
  int nLumi_;
  unsigned long long m_cacheID_;
};

// -----------------------------
//  constructors and destructor
// -----------------------------

DQMLumiMonitor::DQMLumiMonitor(const edm::ParameterSet& ps) : parameters_(ps) {
  moduleName_ = parameters_.getParameter<std::string>("ModuleName");
  folderName_ = parameters_.getParameter<std::string>("FolderName");
  pixelClusterInputTag_ =
      consumes<edmNew::DetSetVector<SiPixelCluster> >(parameters_.getParameter<edm::InputTag>("PixelClusterInputTag"));
  lumiRecordName_ = consumes<LumiSummary, edm::InLumi>(parameters_.getParameter<std::string>("LumiRecordName"));

  nClusME_ = nullptr;
  nClusVsLSME_ = nullptr;
  intLumiVsLSME_ = nullptr;
  corrIntLumiAndClusVsLSME_ = nullptr;
}

void DQMLumiMonitor::bookHistograms() {
  edm::ParameterSet ClusHistoPar = parameters_.getParameter<edm::ParameterSet>("TH1ClusPar");
  edm::ParameterSet LumiHistoPar = parameters_.getParameter<edm::ParameterSet>("TH1LumiPar");
  edm::ParameterSet LumiSecHistoPar = parameters_.getParameter<edm::ParameterSet>("TH1LSPar");

  std::string currentFolder = moduleName_ + "/" + folderName_;
  dbe_->setCurrentFolder(currentFolder);

  if (nClusME_ == nullptr)
    nClusME_ = dbe_->book1D("nPlxClus",
                            " Number of Pixel Clusters ",
                            ClusHistoPar.getParameter<int32_t>("Xbins"),
                            ClusHistoPar.getParameter<double>("Xmin"),
                            ClusHistoPar.getParameter<double>("Xmax"));
  else
    nClusME_->Reset();
  if (nClusVsLSME_ == nullptr)
    nClusVsLSME_ = dbe_->bookProfile("nClusVsLS",
                                     " Number of Pixel Cluster Vs LS number",
                                     LumiSecHistoPar.getParameter<int32_t>("Xbins"),
                                     LumiSecHistoPar.getParameter<double>("Xmin"),
                                     LumiSecHistoPar.getParameter<double>("Xmax"),
                                     0.0,
                                     0.0,
                                     "");
  else
    nClusVsLSME_->Reset();
  if (intLumiVsLSME_ == nullptr)
    intLumiVsLSME_ = dbe_->bookProfile("intLumiVsLS",
                                       " Integrated Luminosity Vs LS number",
                                       LumiSecHistoPar.getParameter<int32_t>("Xbins"),
                                       LumiSecHistoPar.getParameter<double>("Xmin"),
                                       LumiSecHistoPar.getParameter<double>("Xmax"),
                                       0.0,
                                       0.0,
                                       "");
  else
    intLumiVsLSME_->Reset();

  if (corrIntLumiAndClusVsLSME_ == nullptr)
    corrIntLumiAndClusVsLSME_ = dbe_->bookProfile2D("corrIntLumiAndClusVsLS",
                                                    " Correlation of nCluster and Integrated Luminosity Vs LS number",
                                                    LumiSecHistoPar.getParameter<int32_t>("Xbins"),
                                                    LumiSecHistoPar.getParameter<double>("Xmin"),
                                                    LumiSecHistoPar.getParameter<double>("Xmax"),
                                                    LumiHistoPar.getParameter<int32_t>("Xbins"),
                                                    LumiHistoPar.getParameter<double>("Xmin"),
                                                    LumiHistoPar.getParameter<double>("Xmax"),
                                                    0.0,
                                                    0.0);
  else
    corrIntLumiAndClusVsLSME_->Reset();
}
void DQMLumiMonitor::beginJob() {
  dbe_ = edm::Service<DQMStore>().operator->();
  intLumi_ = -1.0;
  nLumi_ = -1;
}

void DQMLumiMonitor::beginRun(edm::Run const& iRun, edm::EventSetup const& iSetup) { bookHistograms(); }
void DQMLumiMonitor::analyze(edm::Event const& iEvent, edm::EventSetup const& iSetup) {
  //Access Pixel Clusters
  edm::Handle<edmNew::DetSetVector<SiPixelCluster> > siPixelClusters;
  iEvent.getByToken(pixelClusterInputTag_, siPixelClusters);

  if (!siPixelClusters.isValid()) {
    edm::LogError("PixelLumiMonotor") << "Could not find Cluster Collection ";
    return;
  }
  unsigned int nClusterPix = (*siPixelClusters).dataSize();
  nClusME_->Fill(nClusterPix);
  if (nLumi_ != -1)
    nClusVsLSME_->Fill(nLumi_, nClusterPix);
  if (intLumi_ != -1 || nLumi_ != -1)
    corrIntLumiAndClusVsLSME_->Fill(nLumi_, intLumi_, nClusterPix);
}

void DQMLumiMonitor::beginLuminosityBlock(edm::LuminosityBlock const& lumiSeg, edm::EventSetup const& eSetup) {}

void DQMLumiMonitor::endLuminosityBlock(edm::LuminosityBlock const& lumiBlock, edm::EventSetup const& eSetup) {
  edm::LogInfo("PixelLumiMonotor") << " Run Number " << lumiBlock.run() << " Lumi Section Numnber "
                                   << lumiBlock.luminosityBlock();

  nLumi_ = lumiBlock.luminosityBlock();

  // Access Lumi Summary
  edm::Handle<LumiSummary> lumiSummary_;
  lumiBlock.getByToken(lumiRecordName_, lumiSummary_);
  if (lumiSummary_->isValid()) {
    intLumi_ = lumiSummary_->intgDelLumi();
    edm::LogInfo("PixelLumiMonotor") << " Luminosity in this Lumi Section " << intLumi_;
    intLumiVsLSME_->Fill(nLumi_, intLumi_);
  } else {
    edm::LogError("PixelLumiMonotor") << "No valid data found!";
  }
  /*
  // Access Lumi Details
  Handle<LumiDetails> lumiDetails;
  lumiBlock.getByLabel("expressLumiProducer", lumiDetails);
  if(lumiDetails->isValid()){
    std::cout<<"valid detail"<<std::endl;
  }else{
    std::cout << "no valid lumi detail data" <<std::endl;
  }  */
}

void DQMLumiMonitor::endRun(edm::Run const& iRun, edm::EventSetup const& iSetup) {}

// Define this as a plug-in
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(DQMLumiMonitor);
