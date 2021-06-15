#ifndef RecoLocalTracker_SiStripApproximatedClustersDump_h
#define RecoLocalTracker_SiStripApproximatedClustersDump_h

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Common/interface/DetSet.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "DataFormats/SiStripCluster/interface/SiStripApproximateCluster.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "CommonTools/Utils/interface/TFileDirectory.h"

#include <memory>
#include <iostream>

//ROOT inclusion
#include "TROOT.h"
#include "TFile.h"
#include "TNtuple.h"
#include "TTree.h"
#include "TMath.h"
#include "TList.h"
#include "TString.h"

//
// class decleration
//

class SiStripApproximatedClustersDump : public edm::one::EDAnalyzer<edm::one::SharedResources> {
public:
  explicit SiStripApproximatedClustersDump(const edm::ParameterSet&);
  ~SiStripApproximatedClustersDump() override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void analyze(const edm::Event&, const edm::EventSetup&) override;

  edm::InputTag inputTagClusters;
  edm::EDGetTokenT<edmNew::DetSetVector<SiStripApproximateCluster> > clusterToken;

  TTree* outNtuple;
  edm::Service<TFileService> fs;

  uint32_t detId;
  uint16_t barycenter;
  uint16_t width;
  uint8_t avCharge;
  edm::EventNumber_t eventN;
};
#endif
