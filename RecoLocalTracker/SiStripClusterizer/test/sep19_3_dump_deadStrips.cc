/*
 * Dump the dead strip info based on GT
 *    - deadStripTree: dead strip (channel #) on each detId
 */
// system includes
#include <memory>
#include <iostream>

// user include files
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "CommonTools/Utils/interface/TFileDirectory.h"
#include "DataFormats/Common/interface/DetSet.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "DataFormats/SiStripCluster/interface/SiStripApproximateCluster.h"
#include "DataFormats/SiStripCluster/interface/SiStripApproximateClusterCollection.h"
#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/StripGeomDetUnit.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/CommonTopologies/interface/StripTopology.h"
#include "CalibFormats/SiStripObjects/interface/SiStripQuality.h"
#include "CalibTracker/Records/interface/SiStripQualityRcd.h"

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

class sep19_3_dump_deadStrips : public edm::one::EDAnalyzer<edm::one::SharedResources> {
public:
  explicit sep19_3_dump_deadStrips(const edm::ParameterSet&);
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void analyze(const edm::Event&, const edm::EventSetup&) override;

  const edm::ESGetToken<SiStripQuality, SiStripQualityRcd> qualESToken;
  edm::ESHandle<SiStripQuality> qualityHandle;

  // output
  edm::Service<TFileService> fs;
  TTree* deadStripTree;
  edm::EventNumber_t eventN;
  int runN;
  int lumi;
  int    detId;
  uint16_t    size;
  uint16_t    channel[800];


  // loading tracker modules' detId
  TFile* f_trk;
  TTree* tracker;

  int part; // 1: Pixel, 2: Strip

};

sep19_3_dump_deadStrips::sep19_3_dump_deadStrips(const edm::ParameterSet& conf) : qualESToken(esConsumes()) {
  usesResource("TFileService");


  deadStripTree = fs->make<TTree>("deadStripTree", "deadStripTree");
  deadStripTree->Branch("event", &eventN, "event/i");
  deadStripTree->Branch("run",   &runN, "run/I");
  deadStripTree->Branch("lumi",  &lumi, "lumi/I");

  deadStripTree->Branch("detId", &detId, "detId/I");
  deadStripTree->Branch("size", &size, "size/s");

  deadStripTree->Branch("channel", channel, "channel[size]/s");

  std::cout << "read tracker.root" << std::endl;
  f_trk    = TFile::Open("tracker.root", "read");
  tracker  = (TTree*) f_trk->Get("tracker");
  
  tracker->SetBranchAddress("part",  &part);
  tracker->SetBranchAddress("detId",  &detId);
}

void sep19_3_dump_deadStrips::analyze(const edm::Event& event, const edm::EventSetup& es) {
  qualityHandle = es.getHandle(qualESToken);

  eventN = event.id().event();
  runN   = (int) event.id().run();
  lumi   = (int) event.id().luminosityBlock();

  for (int i = 0; i < tracker->GetEntries(); ++i)
  {
    tracker->GetEntry(i);
    if (part==1) continue; // 1: Pixel, 2: Strip

    size = 0;
    for (int strip = 0; strip < 769; ++strip) 
    {
      if (qualityHandle->IsStripBad(detId, strip)) 
      {
        channel[size] = strip;
        size ++;
      }
    }
    deadStripTree->Fill();    
  }
}

void sep19_3_dump_deadStrips::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;  
  descriptions.add("sep19_3_dump_deadStrips", desc);
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(sep19_3_dump_deadStrips);
