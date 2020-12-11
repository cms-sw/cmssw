// File: ReadPixClusters.cc
// Description: To test the pixel clusters.
// Author: Danek Kotlinski
// Creation Date:  Initial version. 3/06
// Modify to work with CMSSW620, 8/13, CMSSW700, 10/13 d.k.
// Add ByToken data access.
//--------------------------------------------
#include <memory>
#include <string>
#include <iostream>

#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
//#include "DataFormats/Common/interface/Handle.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/InputTag.h"
//#include "FWCore/Utilities/interface/EDGetToken.h"  // not needed

#include "DataFormats/SiPixelCluster/interface/SiPixelCluster.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/DetId/interface/DetId.h"

#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "DataFormats/TrackerCommon/interface/PixelBarrelName.h"

#include "Geometry/CommonDetUnit/interface/PixelGeomDetUnit.h"
#include "Geometry/CommonDetUnit/interface/PixelGeomDetType.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/CommonDetUnit/interface/GeomDetType.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "Geometry/CommonTopologies/interface/PixelTopology.h"

// For L1
#include "L1Trigger/GlobalTriggerAnalyzer/interface/L1GtUtils.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutSetupFwd.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutSetup.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutRecord.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerObjectMapRecord.h"

// For HLT
#include "DataFormats/HLTReco/interface/TriggerEvent.h"
#include "DataFormats/HLTReco/interface/TriggerTypeDefs.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"
#include "FWCore/Common/interface/TriggerNames.h"

// To use root histos
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

// For ROOT
#include <TROOT.h>
#include <TChain.h>
#include <TFile.h>
#include <TF1.h>
#include <TH2F.h>
#include <TH1F.h>

#define NEW_ID
#ifdef NEW_ID
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"
#else
#include "DataFormats/SiPixelDetId/interface/PXBDetId.h"
#include "DataFormats/SiPixelDetId/interface/PXFDetId.h"
#endif

#define HISTOS

#include "DataFormats/Provenance/interface/RunLumiEventNumber.h"

using namespace std;

//=============================================================================

class ReadPixClusters : public edm::EDAnalyzer {
public:
  explicit ReadPixClusters(const edm::ParameterSet &conf);
  virtual ~ReadPixClusters();
  virtual void analyze(const edm::Event &e, const edm::EventSetup &c) override;
  virtual void beginRun(edm::Run const &, edm::EventSetup const &) override;
  virtual void beginJob() override;
  virtual void endJob() override;

private:
  edm::ParameterSet conf_;
  edm::InputTag src_;
  edm::EDGetTokenT<edmNew::DetSetVector<SiPixelCluster>> tPixelCluster;
  edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> trackerTopoToken_;
  edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> trackerGeomToken_;
  bool printLocal;
  int countEvents, countAllEvents;
  double sumClusters;

  //TFile* hFile;
  TH1F *hdetunit;
  TH1F *hpixid, *hpixsubid, *hlayerid, *hladder1id, *hladder2id, *hladder3id, *hz1id, *hz2id, *hz3id;

  TH1F *hcharge1, *hcharge2, *hcharge3, *hcharge4, *hcharge5;
  TH1F *hpixcharge1, *hpixcharge2, *hpixcharge3, *hpixcharge4, *hpixcharge5;
  TH1F *hcols1, *hcols2, *hcols3, *hrows1, *hrows2, *hrows3;
  TH1F *hpcols1, *hpcols2, *hpcols3, *hprows1, *hprows2, *hprows3;
  TH1F *hsize1, *hsize2, *hsize3, *hsize4, *hsize5, *hsizex1, *hsizex2, *hsizex3, *hsizex4, *hsizex5, *hsizey1,
      *hsizey2, *hsizey3, *hsizey4, *hsizey5;

  TH1F *hclusPerDet1, *hclusPerDet2, *hclusPerDet3;
  TH1F *hpixPerDet1, *hpixPerDet2, *hpixPerDet3;
  TH1F *hpixPerLink1, *hpixPerLink2, *hpixPerLink3;
  TH1F *hclusPerLay1, *hclusPerLay2, *hclusPerLay3;
  TH1F *hpixPerLay1, *hpixPerLay2, *hpixPerLay3;
  TH1F *hdetsPerLay1, *hdetsPerLay2, *hdetsPerLay3;
  TH1F *hclus, *hclusBPix, *hclusFPix, *hdigis, *hdigisB, *hdigisF;

  TH1F *hdetr, *hdetz;
  //   TH1F *hcolsB,  *hrowsB,  *hcolsF,  *hrowsF;
  //   TH2F *htest1, *htest2;
  TH2F *hDetMap1, *hDetMap2, *hDetMap3;
  TH2F *hpDetMap1, *hpDetMap2, *hpDetMap3;
  TH2F *hpixDetMap1, *hpixDetMap2, *hpixDetMap3;
  TH2F *hcluDetMap1, *hcluDetMap2, *hcluDetMap3;

  TH1F *hevent, *horbit, *hlumi, *hlumi0, *hlumi1;
  TH1F *hbx, *hbx0, *hbx1;
  TH1F *hdets, *hmbits1, *hmbits2, *hmbits3, *hmaxPixPerDet;

  TH1F *hclusPerDisk1, *hclusPerDisk2, *hclusPerDisk3, *hclusPerDisk4;
  TH1F *htest;
};

/////////////////////////////////////////////////////////////////
// Contructor, empty.
ReadPixClusters::ReadPixClusters(edm::ParameterSet const &conf)
    : conf_(conf), src_(conf.getParameter<edm::InputTag>("src")) {
  printLocal = conf.getUntrackedParameter<bool>("Verbosity", false);
  //src_ =  conf.getParameter<edm::InputTag>( "src" );

  cout << " Construct " << printLocal << endl;
  tPixelCluster = consumes<edmNew::DetSetVector<SiPixelCluster>>(src_);
  trackerTopoToken_ = esConsumes<TrackerTopology, TrackerTopologyRcd>();
  trackerGeomToken_ = esConsumes<TrackerGeometry, TrackerDigiGeometryRecord>();
}
// Virtual destructor needed.
ReadPixClusters::~ReadPixClusters() {}

// ------------ method called at the begining   ------------
void ReadPixClusters::beginRun(edm::Run const &, const edm::EventSetup &iSetup) {
  cout << "beginRun -  PixelClusterTest " << printLocal << endl;
}

// ------------ method called at the begining   ------------
void ReadPixClusters::beginJob() {
  cout << "Initialize PixelClusterTest " << printLocal << endl;

#ifdef HISTOS

  // NEW way to use root (from 2.0.0?)
  edm::Service<TFileService> fs;

  //=====================================================================

  hladder1id = fs->make<TH1F>("hladder1id", "Ladder L1 id", 23, -11.5, 11.5);
  hladder2id = fs->make<TH1F>("hladder2id", "Ladder L2 id", 35, -17.5, 17.5);
  hladder3id = fs->make<TH1F>("hladder3id", "Ladder L3 id", 47, -23.5, 23.5);
  hz1id = fs->make<TH1F>("hz1id", "Z-index id L1", 11, -5.5, 5.5);
  hz2id = fs->make<TH1F>("hz2id", "Z-index id L2", 11, -5.5, 5.5);
  hz3id = fs->make<TH1F>("hz3id", "Z-index id L3", 11, -5.5, 5.5);

  int sizeH = 200;
  float lowH = -0.5;
  float highH = 199.5;

  hclusPerDet1 = fs->make<TH1F>("hclusPerDet1", "Clus per det l1", sizeH, lowH, highH);
  hclusPerDet2 = fs->make<TH1F>("hclusPerDet2", "Clus per det l2", sizeH, lowH, highH);
  hclusPerDet3 = fs->make<TH1F>("hclusPerDet3", "Clus per det l3", sizeH, lowH, highH);

  sizeH = 1000;
  highH = 1999.5;
  hpixPerDet1 = fs->make<TH1F>("hpixPerDet1", "Pix per det l1", sizeH, lowH, highH);
  hpixPerDet2 = fs->make<TH1F>("hpixPerDet2", "Pix per det l2", sizeH, lowH, highH);
  hpixPerDet3 = fs->make<TH1F>("hpixPerDet3", "Pix per det l3", sizeH, lowH, highH);

  sizeH = 1000;
  highH = 999.5;
  hpixPerLink1 = fs->make<TH1F>("hpixPerLink1", "Pix per link l1", sizeH, lowH, highH);
  hpixPerLink2 = fs->make<TH1F>("hpixPerLink2", "Pix per link l2", sizeH, lowH, highH);
  hpixPerLink3 = fs->make<TH1F>("hpixPerLink3", "Pix per link l3", sizeH, lowH, highH);

  sizeH = 2000;
  highH = 1999.5;
  hclusPerLay1 = fs->make<TH1F>("hclusPerLay1", "Clus per layer l1", sizeH, lowH, highH);
  hclusPerLay2 = fs->make<TH1F>("hclusPerLay2", "Clus per layer l2", sizeH, lowH, highH);
  hclusPerLay3 = fs->make<TH1F>("hclusPerLay3", "Clus per layer l3", sizeH, lowH, highH);

  hclusPerDisk1 = fs->make<TH1F>("hclusPerDisk1", "Clus per disk1", sizeH, lowH, highH);
  hclusPerDisk2 = fs->make<TH1F>("hclusPerDisk2", "Clus per disk2", sizeH, lowH, highH);
  hclusPerDisk3 = fs->make<TH1F>("hclusPerDisk3", "Clus per disk3", sizeH, lowH, highH);
  hclusPerDisk4 = fs->make<TH1F>("hclusPerDisk4", "Clus per disk4", sizeH, lowH, highH);

  sizeH = 2000;
  highH = 9999.5;
  hpixPerLay1 = fs->make<TH1F>("hpixPerLay1", "Pix per layer l1", sizeH, lowH, highH);
  hpixPerLay2 = fs->make<TH1F>("hpixPerLay2", "Pix per layer l2", sizeH, lowH, highH);
  hpixPerLay3 = fs->make<TH1F>("hpixPerLay3", "Pix per layer l3", sizeH, lowH, highH);

  hclus = fs->make<TH1F>("hclus", "Clus per event", sizeH, lowH, highH);
  hdigis = fs->make<TH1F>("hdigis", "All Digis in clus per event", 2000, lowH, 19999.5);
  hdigisB = fs->make<TH1F>("hdigisB", "BPix Digis in clus per event", 8000, lowH, 19999.5);
  hdigisF = fs->make<TH1F>("hdigisF", "FPix Digis in clus per event", 2000, lowH, 7999.5);
  hclusBPix = fs->make<TH1F>("hclusBPix", "Bpix Clus per event", 2000, 0., 2000.);
  hclusFPix = fs->make<TH1F>("hclusFPix", "Fpix Clus per event", 2000, 0., 2000.);
  hdets = fs->make<TH1F>("hdets", "Dets per event", 2000, -0.5, 1999.5);

  hmaxPixPerDet = fs->make<TH1F>("hmaxPixPerDet", "Max pixels per det", 1000, -0.5, 999.5);

  sizeH = 1000;
  highH = 1999.5;

  hdetsPerLay1 = fs->make<TH1F>("hdetsPerLay1", "Full dets per layer l1", 161, -0.5, 160.5);
  hdetsPerLay3 = fs->make<TH1F>("hdetsPerLay3", "Full dets per layer l3", 353, -0.5, 352.5);
  hdetsPerLay2 = fs->make<TH1F>("hdetsPerLay2", "Full dets per layer l2", 257, -0.5, 256.5);

  sizeH = 120;
  lowH = 0.;
  highH = 121.0;                                                             // charge limit in kelec
  hcharge1 = fs->make<TH1F>("hcharge1", "Clu charge l1", sizeH, 0., highH);  //in ke
  hcharge2 = fs->make<TH1F>("hcharge2", "Clu charge l2", sizeH, 0., highH);
  hcharge3 = fs->make<TH1F>("hcharge3", "Clu charge l3", sizeH, 0., highH);
  hcharge4 = fs->make<TH1F>("hcharge4", "Clu charge d1", sizeH, 0., highH);
  hcharge5 = fs->make<TH1F>("hcharge5", "Clu charge d2", sizeH, 0., highH);

  sizeH = 90;
  highH = 61.0;                                                                    // charge limit in kelec
  hpixcharge1 = fs->make<TH1F>("hpixcharge1", "Pix charge l1", sizeH, 0., highH);  //in ke
  hpixcharge2 = fs->make<TH1F>("hpixcharge2", "Pix charge l2", sizeH, 0., highH);
  hpixcharge3 = fs->make<TH1F>("hpixcharge3", "Pix charge l3", sizeH, 0., highH);
  hpixcharge4 = fs->make<TH1F>("hpixcharge4", "Pix charge d1", sizeH, 0., highH);
  hpixcharge5 = fs->make<TH1F>("hpixcharge5", "Pix charge d2", sizeH, 0., highH);

  hcols1 = fs->make<TH1F>("hcols1", "Layer 1 cols", 500, -0.5, 499.5);
  hcols2 = fs->make<TH1F>("hcols2", "Layer 2 cols", 500, -0.5, 499.5);
  hcols3 = fs->make<TH1F>("hcols3", "Layer 3 cols", 500, -0.5, 499.5);

  hrows1 = fs->make<TH1F>("hrows1", "Layer 1 rows", 200, -0.5, 199.5);
  hrows2 = fs->make<TH1F>("hrows2", "Layer 2 rows", 200, -0.5, 199.5);
  hrows3 = fs->make<TH1F>("hrows3", "layer 3 rows", 200, -0.5, 199.5);

  hpcols1 = fs->make<TH1F>("hpcols1", "Layer 1 pix cols", 500, -0.5, 499.5);
  hpcols2 = fs->make<TH1F>("hpcols2", "Layer 2 pix cols", 500, -0.5, 499.5);
  hpcols3 = fs->make<TH1F>("hpcols3", "Layer 3 pix cols", 500, -0.5, 499.5);

  hprows1 = fs->make<TH1F>("hprows1", "Layer 1 pix rows", 200, -0.5, 199.5);
  hprows2 = fs->make<TH1F>("hprows2", "Layer 2 pix rows", 200, -0.5, 199.5);
  hprows3 = fs->make<TH1F>("hprows3", "layer 3 pix rows", 200, -0.5, 199.5);

  sizeH = 1000;
  highH = 999.5;  // charge limit in kelec
  hsize1 = fs->make<TH1F>("hsize1", "layer 1 clu size", sizeH, -0.5, highH);
  hsize2 = fs->make<TH1F>("hsize2", "layer 2 clu size", sizeH, -0.5, highH);
  hsize3 = fs->make<TH1F>("hsize3", "layer 3 clu size", sizeH, -0.5, highH);
  hsize4 = fs->make<TH1F>("hsize4", "disk 1 clu size", sizeH, -0.5, highH);
  hsize5 = fs->make<TH1F>("hsize5", "disk 2 clu size", sizeH, -0.5, highH);

  hsizex1 = fs->make<TH1F>("hsizex1", "lay1 clu size in x", 10, -0.5, 9.5);
  hsizex2 = fs->make<TH1F>("hsizex2", "lay2 clu size in x", 10, -0.5, 9.5);
  hsizex3 = fs->make<TH1F>("hsizex3", "lay3 clu size in x", 10, -0.5, 9.5);
  hsizex4 = fs->make<TH1F>("hsizex4", "d1 clu size in x", 10, -0.5, 9.5);
  hsizex5 = fs->make<TH1F>("hsizex5", "d2 clu size in x", 10, -0.5, 9.5);

  hsizey1 = fs->make<TH1F>("hsizey1", "lay1 clu size in y", 20, -0.5, 19.5);
  hsizey2 = fs->make<TH1F>("hsizey2", "lay2 clu size in y", 20, -0.5, 19.5);
  hsizey3 = fs->make<TH1F>("hsizey3", "lay3 clu size in y", 20, -0.5, 19.5);
  hsizey4 = fs->make<TH1F>("hsizey4", "d1 clu size in y", 20, -0.5, 19.5);
  hsizey5 = fs->make<TH1F>("hsizey5", "d2 clu size in y", 20, -0.5, 19.5);

  hevent = fs->make<TH1F>("hevent", "event", 1000, 0, 10000000.);
  horbit = fs->make<TH1F>("horbit", "orbit", 100, 0, 100000000.);

  hlumi1 = fs->make<TH1F>("hlumi1", "lumi", 2000, 0, 2000.);
  hlumi0 = fs->make<TH1F>("hlumi0", "lumi", 2000, 0, 2000.);
  hlumi = fs->make<TH1F>("hlumi", "lumi", 2000, 0, 2000.);
  hbx1 = fs->make<TH1F>("hbx1", "bx", 4000, 0, 4000.);
  hbx0 = fs->make<TH1F>("hbx0", "bx", 4000, 0, 4000.);
  hbx = fs->make<TH1F>("hbx", "bx", 4000, 0, 4000.);

  hDetMap1 = fs->make<TH2F>("hDetMap1", " ", 9, -4.5, 4.5, 21, -10.5, 10.5);
  hDetMap1->SetOption("colz");
  hDetMap2 = fs->make<TH2F>("hDetMap2", " ", 9, -4.5, 4.5, 33, -16.5, 16.5);
  hDetMap2->SetOption("colz");
  hDetMap3 = fs->make<TH2F>("hDetMap3", " ", 9, -4.5, 4.5, 45, -22.5, 22.5);
  hDetMap3->SetOption("colz");

  hpDetMap1 = fs->make<TH2F>("hpDetMap1", " ", 9, -4.5, 4.5, 21, -10.5, 10.5);
  hpDetMap1->SetOption("colz");
  hpDetMap2 = fs->make<TH2F>("hpDetMap2", " ", 9, -4.5, 4.5, 33, -16.5, 16.5);
  hpDetMap2->SetOption("colz");
  hpDetMap3 = fs->make<TH2F>("hpDetMap3", " ", 9, -4.5, 4.5, 45, -22.5, 22.5);
  hpDetMap3->SetOption("colz");

  hpixDetMap1 = fs->make<TH2F>("hpixDetMap1", "pix det layer 1", 416, 0., 416., 160, 0., 160.);
  hpixDetMap2 = fs->make<TH2F>("hpixDetMap2", "pix det layer 2", 416, 0., 416., 160, 0., 160.);
  hpixDetMap3 = fs->make<TH2F>("hpixDetMap3", "pix det layer 3", 416, 0., 416., 160, 0., 160.);

  hcluDetMap1 = fs->make<TH2F>("hcluDetMap1", "clu det layer 1", 416, 0., 416., 160, 0., 160.);
  hcluDetMap2 = fs->make<TH2F>("hcluDetMap2", "clu det layer 1", 416, 0., 416., 160, 0., 160.);
  hcluDetMap3 = fs->make<TH2F>("hcluDetMap3", "clu det layer 1", 416, 0., 416., 160, 0., 160.);
  htest = fs->make<TH1F>("htest", "FPix R", 300, -15., 15.);

#endif

  countEvents = 0;
  countAllEvents = 0;
  sumClusters = 0.;
}
// ------------ method called to at the end of the job  ------------
void ReadPixClusters::endJob() {
  sumClusters = sumClusters / float(countEvents);
  cout << " End PixelClusTest, events all/with hits=  " << countAllEvents << "/" << countEvents << " " << sumClusters
       << " " << printLocal << endl;
}
//////////////////////////////////////////////////////////////////
// Functions that gets called by framework every event
void ReadPixClusters::analyze(const edm::Event &e, const edm::EventSetup &es) {
  using namespace edm;

  // Get event setup
  edm::ESHandle<TrackerGeometry> geom = es.getHandle(trackerGeomToken_);
  const TrackerGeometry &theTracker(*geom);

#ifdef NEW_ID
  //Retrieve tracker topology from geometry
  edm::ESHandle<TrackerTopology> tTopo = es.getHandle(trackerTopoToken_);
#endif

  countAllEvents++;
  RunNumber_t const run = e.id().run();
  EventNumber_t const event = e.id().event();
  LuminosityBlockNumber_t const lumiBlock = e.luminosityBlock();

  int bx = e.bunchCrossing();
  int orbit = e.orbitNumber();

  // Get Cluster Collection from InputTag
  edm::Handle<edmNew::DetSetVector<SiPixelCluster>> clusters;
  e.getByToken(tPixelCluster, clusters);

  const edmNew::DetSetVector<SiPixelCluster> &input = *clusters;
  int numOf = input.size();

  hbx0->Fill(float(bx));
  hlumi0->Fill(float(lumiBlock));

  if (printLocal)
    cout << "run " << run << " event " << event << " bx " << bx << " lumi " << lumiBlock << " orbit " << orbit << " "
         << numOf << endl;

  hdets->Fill(float(numOf));  // number of modules with pix

  // Select events with pixels
  //if(numOf<1) return; // skip events with  pixel dets
  //if(numOf<4) return; // skip events with few pixel dets

  hevent->Fill(float(event));
  hlumi->Fill(float(lumiBlock));
  hbx->Fill(float(bx));
  horbit->Fill(float(orbit));

  countEvents++;
  int numberOfDetUnits = 0;
  int numberOfClusters = 0;
  int numberOfPixels = 0;
  int numberOfDetUnits1 = 0;
  int numOfClustersPerDet1 = 0;
  int numOfClustersPerLay1 = 0;
  int numberOfDetUnits2 = 0;
  int numOfClustersPerDet2 = 0;
  int numOfClustersPerLay2 = 0;
  int numberOfDetUnits3 = 0;
  int numOfClustersPerDet3 = 0;
  int numOfClustersPerLay3 = 0;

  int numOfPixPerLay1 = 0;
  int numOfPixPerLay2 = 0;
  int numOfPixPerLay3 = 0;

  int numOfPixPerDet1 = 0;
  int numOfPixPerDet2 = 0;
  int numOfPixPerDet3 = 0;

  int numOfPixPerLink11 = 0;
  int numOfPixPerLink12 = 0;
  int numOfPixPerLink21 = 0;
  int numOfPixPerLink22 = 0;
  //SK:unused  int numOfPixPerLink3=0;

  int maxClusPerDet = 0;
  int maxPixPerDet = 0;
  unsigned int maxPixPerClu = 0;

  int numOfClustersPerDisk1 = 0;
  int numOfClustersPerDisk2 = 0;
  int numOfClustersPerDisk3 = 0;
  int numOfClustersPerDisk4 = 0;
  int numOfPixPerDisk1 = 0;
  int numOfPixPerDisk2 = 0;
  int numOfPixPerDisk3 = 0;
  int numOfPixPerDisk4 = 0;

  float aveCharge1 = 0., aveCharge2 = 0., aveCharge3 = 0., aveCharge4 = 0., aveCharge5 = 0.;

  static int module1[416][160] = {{0}};
  static int module2[416][160] = {{0}};
  static int module3[416][160] = {{0}};

  // get vector of detunit ids
  //--- Loop over detunits.
  edmNew::DetSetVector<SiPixelCluster>::const_iterator DSViter = input.begin();
  for (; DSViter != input.end(); DSViter++) {
    bool valid = false;
    unsigned int detid = DSViter->detId();
    // Det id
    DetId detId = DetId(detid);             // Get the Detid object
    unsigned int detType = detId.det();     // det type, pixel=1
    unsigned int subid = detId.subdetId();  //subdetector type, barrel=1

    if (printLocal)
      cout << "Det: " << detId.rawId() << " " << detId.null() << " " << detType << " " << subid << endl;

#ifdef HISTOS
      //hdetunit->Fill(float(detid));
      //hpixid->Fill(float(detType));
      //hpixsubid->Fill(float(subid));
#endif  // HISTOS

    if (detType != 1)
      continue;  // look only at pixels
    ++numberOfDetUnits;

    //const GeomDetUnit * genericDet = geom->idToDet(detId);
    //const PixelGeomDetUnit * pixDet =
    //dynamic_cast<const PixelGeomDetUnit*>(genericDet);

    // Get the geom-detector
    const PixelGeomDetUnit *theGeomDet = dynamic_cast<const PixelGeomDetUnit *>(theTracker.idToDet(detId));
    double detZ = theGeomDet->surface().position().z();
    double detR = theGeomDet->surface().position().perp();

    //const BoundPlane& plane = theGeomDet->surface(); //for transf.

    //double detThick = theGeomDet->specificSurface().bounds().thickness();
    //int cols = theGeomDet->specificTopology().ncolumns();
    //int rows = theGeomDet->specificTopology().nrows();

    const PixelTopology *topol = &(theGeomDet->specificTopology());

    // barrel ids
    unsigned int layerC = 0;
    unsigned int ladderC = 0;
    unsigned int zindex = 0;
    int shell = 0;      // shell id // Shell { mO = 1, mI = 2 , pO =3 , pI =4 };
    int sector = 0;     // 1-8
    int ladder = 0;     // 1-22
    int layer = 0;      // 1-3
    int module = 0;     // 1-4
    bool half = false;  //

    // Endcap ids
    unsigned int disk = 0;     //1,2,3
    unsigned int blade = 0;    //1-24
    unsigned int moduleF = 0;  // plaquette 1,2,3,4
    unsigned int side = 0;     //size=1 for -z, 2 for +z
    unsigned int panel = 0;    //panel=1

    edmNew::DetSet<SiPixelCluster>::const_iterator clustIt;

    // Subdet id, pix barrel=1, forward=2
    if (subid == 2) {  // forward

#ifdef NEW_ID
      disk = tTopo->pxfDisk(detid);      //1,2,3
      blade = tTopo->pxfBlade(detid);    //1-24
      zindex = tTopo->pxfModule(detid);  //
      side = tTopo->pxfSide(detid);      //size=1 for -z, 2 for +z
      panel = tTopo->pxfPanel(detid);    //panel=1
#else
      PXFDetId pdetId = PXFDetId(detid);
      disk = pdetId.disk();       //1,2,3
      blade = pdetId.blade();     //1-24
      moduleF = pdetId.module();  // plaquette
      side = pdetId.side();       //size=1 for -z, 2 for +z
      panel = pdetId.panel();     //panel=1
#endif

      if (printLocal)
        cout << " forward det, disk " << disk << ", blade " << blade << ", module " << moduleF << ", side " << side
             << ", panel " << panel << " pos = " << detZ << " " << detR << endl;

      bool fpixInner = (((panel == 1) && (moduleF <= 2)) || ((panel == 2) && (moduleF <= 1)));  // make split at 10cm
      if (fpixInner)
        htest->Fill(detR);

    } else if (subid == 1) {  // barrel

#ifdef HISTOS
      //hdetr->Fill(detR);
      //hdetz->Fill(detZ);
#endif  // HISTOS

      //hcolsB->Fill(float(cols));
      //hrowsB->Fill(float(rows));

#ifdef NEW_ID
      layerC = tTopo->pxbLayer(detid);
      ladderC = tTopo->pxbLadder(detid);
      zindex = tTopo->pxbModule(detid);
      PixelBarrelName pbn(detid);
#else
      PXBDetId pdetId = PXBDetId(detid);
      //unsigned int detTypeP=pdetId.det();
      //unsigned int subidP=pdetId.subdetId();
      // Barell layer = 1,2,3
      layerC = pdetId.layer();
      // Barrel ladder id 1-20,32,44.
      ladderC = pdetId.ladder();
      // Barrel Z-index=1,8
      zindex = pdetId.module();
      // Convert to online
      PixelBarrelName pbn(pdetId);
#endif

      // Shell { mO = 1, mI = 2 , pO =3 , pI =4 };
      PixelBarrelName::Shell sh = pbn.shell();  //enum
      sector = pbn.sectorName();
      ladder = pbn.ladderName();
      layer = pbn.layerName();
      module = pbn.moduleName();
      half = pbn.isHalfModule();
      shell = int(sh);
      // change the module sign for z<0
      if (shell == 1 || shell == 2)
        module = -module;
      // change ladeer sign for Outer )x<0)
      if (shell == 1 || shell == 3)
        ladder = -ladder;

      if (printLocal) {
        cout << " Barrel layer, ladder, module " << layerC << " " << ladderC << " " << zindex << " " << sh << "("
             << shell << ") " << sector << " " << layer << " " << ladder << " " << module << " " << half << endl;
        //cout<<" Barrel det, thick "<<detThick<<" "
        //  <<" layer, ladder, module "
        //  <<layer<<" "<<ladder<<" "<<zindex<<endl;
        //cout<<" col/row, pitch "<<cols<<" "<<rows<<" "
        //  <<pitchX<<" "<<pitchY<<endl;
      }

    }  // if subid

    if (printLocal) {
      cout << "List clusters : " << endl;
      cout << "Num Charge Size SizeX SizeY X Y Xmin Xmax Ymin Ymax Edge" << endl;
    }

    // Loop over clusters
    for (clustIt = DSViter->begin(); clustIt != DSViter->end(); clustIt++) {
      sumClusters++;
      numberOfClusters++;
      float ch = float(clustIt->charge()) / 1000.;  // convert ke to electrons
      int size = clustIt->size();
      int sizeX = clustIt->sizeX();  //x=row=rfi, starting from CMSSW6 this is limited to 64
      int sizeY = clustIt->sizeY();  //y=col=z_global, starting from CMSSW6 this is limited to 64
      float x = clustIt->x();        // cluster position as float (int+0.5)
      float y = clustIt->y();        // analog average

      // Returns int index of the cluster min/max
      // This parameters are affected by the 64*64 limit on the cluster grid, so max-min<=64.
      int minPixelRow = clustIt->minPixelRow();  //x
      int maxPixelRow = clustIt->maxPixelRow();
      int minPixelCol = clustIt->minPixelCol();  //y
      int maxPixelCol = clustIt->maxPixelCol();

      //unsigned int geoId = clustIt->geographicalId(); // always 0?!

      // edge method moved to topologu class
      bool edgeHitX = (topol->isItEdgePixelInX(minPixelRow)) || (topol->isItEdgePixelInX(maxPixelRow));
      bool edgeHitY = (topol->isItEdgePixelInY(minPixelCol)) || (topol->isItEdgePixelInY(maxPixelCol));

      if (printLocal)
        cout << numberOfClusters << " " << ch << " " << size << " " << sizeX << " " << sizeY << " " << x << " " << y
             << " " << minPixelRow << " " << maxPixelRow << " " << minPixelCol << " " << maxPixelCol << " " << edgeHitX
             << " " << edgeHitY << endl;

      // Get the pixels in the Cluster
      const vector<SiPixelCluster::Pixel> &pixelsVec = clustIt->pixels();
      if (printLocal)
        cout << " Pixels in this cluster " << endl;

      bool bigInX = false, bigInY = false;
      bool edgeHitX2 = false;  // edge method moved
      bool edgeHitY2 = false;  // to topologu class
      //bool edgeInX = false; // edge method moved
      //bool edgeInY = false; // to topology class
      //SK:unused      bool cluBigInX = false; // does this clu include a big pixel
      //SK:unused      bool cluBigInY = false; // does this clu include a big pixel
      //int noisy = 0;

      if (pixelsVec.size() > maxPixPerClu)
        maxPixPerClu = pixelsVec.size();

      // Look at pixels in this cluster. ADC is calibrated, in electrons
      for (unsigned int i = 0; i < pixelsVec.size(); ++i) {  // loop over pixels
        numberOfPixels++;
        float pixx = pixelsVec[i].x;  // index as float=iteger, row index, 0-159
        float pixy = pixelsVec[i].y;  // same, col index, 0-415
        float adc = (float(pixelsVec[i].adc) / 1000.);

        //int chan = PixelChannelIdentifier::pixelToChannel(int(pixx),int(pixy));
        //bool binInX = (RectangularPixelTopology::isItBigPixelInX(int(pixx)));
        //bool bigInY = (RectangularPixelTopology::isItBigPixelInY(int(pixy)));
        //int roc = rocId(int(pixy),int(pixx));  // column, row

#ifdef HISTOS
        // Pixel histos
        if (subid == 1) {  // barrel
          if (layer == 1) {
            numOfPixPerDet1++;
            numOfPixPerLay1++;
            valid = valid || true;
            hpixcharge1->Fill(adc);
            hpixDetMap1->Fill(pixy, pixx);
            hpDetMap1->Fill(float(module), float(ladder));
            module1[int(pixx)][int(pixy)]++;

            hpcols1->Fill(pixy);
            hprows1->Fill(pixx);

            if (pixx < 80.)
              numOfPixPerLink11++;
            else
              numOfPixPerLink12++;

          } else if (layer == 2) {
            numOfPixPerDet2++;
            numOfPixPerLay2++;

            hpcols2->Fill(pixy);
            hprows2->Fill(pixx);

            if (pixx < 80.)
              numOfPixPerLink21++;
            else
              numOfPixPerLink22++;

            hpixcharge2->Fill(adc);
            hpixDetMap2->Fill(pixy, pixx);
            hpDetMap2->Fill(float(module), float(ladder));
            module2[int(pixx)][int(pixy)]++;

          } else if (layer == 3) {
            numOfPixPerDet3++;
            numOfPixPerLay3++;
            valid = valid || true;
            hpixcharge3->Fill(adc);
            hpixDetMap3->Fill(pixy, pixx);
            hpDetMap3->Fill(float(module), float(ladder));
            module3[int(pixx)][int(pixy)]++;

            hpcols3->Fill(pixy);
            hprows3->Fill(pixx);

          }  // if layer

        } else if (subid == 2) {  // endcap
          // pixels

          if (disk == 1) {  // disk1 -+z
            if (side == 1)
              numOfPixPerDisk2++;  // d1,-z
            else if (side == 2)
              numOfPixPerDisk3++;  // d1, +z
            else
              cout << " unknown side " << side << endl;

            hpixcharge4->Fill(adc);

          } else if (disk == 2) {  // disk2 -+z

            if (side == 1)
              numOfPixPerDisk1++;  // d2, -z
            else if (side == 2)
              numOfPixPerDisk4++;  // d2, +z
            else
              cout << " unknown side " << side << endl;

            hpixcharge5->Fill(adc);

          } else
            cout << " unknown disk " << disk << endl;

        }  // end if subdet (pixel loop)

#endif  // HISTOS

        // This looks if there is an edge pixel in the cluster
        bool edgeInX = topol->isItEdgePixelInX(int(pixx));
        bool edgeInY = topol->isItEdgePixelInY(int(pixy));
        if (edgeInX)
          edgeHitX2 = true;
        if (edgeInY)
          edgeHitY2 = true;

        if (printLocal)
          cout << i << " " << pixx << " " << pixy << " " << adc << " " << bigInX << " " << bigInY << " " << edgeInX
               << " " << edgeInY << endl;

        //SK:unused	if(bigInX) cluBigInX=true;
        //SK:unused	if(bigInY) cluBigInY=true;

      }  // pixel loop

      // if(edgeHitX ||  edgeHitX2 || edgeHitY ||  edgeHitY2)
      // 	cout<<" egde "<<edgeHitX<<" "<<edgeHitX2<<" "<<edgeHitY<<" "<<edgeHitY2<<" "
      // 	    <<size<<" "<<sizeX<<" "<<sizeY<<" "
      // 	    <<x<<" "<<y<<" "<<minPixelRow<<" "<<maxPixelRow<<" "<<minPixelCol<<" "
      // 	    <<maxPixelCol<<" "<<edgeHitX<<" "<<edgeHitY<<endl;

      // This will happen  for clusters wider than 64
      if ((edgeHitX != edgeHitX2) && sizeX < 64)
        cout << " wrong egdeX " << edgeHitX << " "
             << edgeHitX2
             //<<" "<<event<<" "<<detid<<" "<<numberOfClusters
             << " size = " << size << " " << sizeX << " " << sizeY << " " << x << " " << y << " " << minPixelRow << " "
             << maxPixelRow << " " << minPixelCol << " " << maxPixelCol << " " << edgeHitX << " " << edgeHitY << endl;

      if ((edgeHitY != edgeHitY2) && sizeY < 64)
        cout << " wrong egdeY " << edgeHitY << " "
             << edgeHitY2
             //<<" "<<event<<" "<<detid<<" "<<numberOfClusters
             << " size = " << size << " " << sizeX << " " << sizeY << " " << x << " " << y << " " << minPixelRow << " "
             << maxPixelRow << " " << minPixelCol << " " << maxPixelCol << " " << edgeHitX << " " << edgeHitY << endl;

#ifdef HISTOS

      // Cluster histos
      if (subid == 1) {  // barrel
        //if (subid==1) {  // barrel
        if (layer == 1) {  // layer

          hDetMap1->Fill(float(module), float(ladder));
          hcluDetMap1->Fill(y, x);
          hcharge1->Fill(ch);
          hcols1->Fill(y);
          hrows1->Fill(x);
          hsize1->Fill(float(size));
          hsizex1->Fill(float(sizeX));
          hsizey1->Fill(float(sizeY));
          numOfClustersPerDet1++;
          numOfClustersPerLay1++;

          aveCharge1 += ch;

        } else if (layer == 2) {
          hDetMap2->Fill(float(module), float(ladder));
          hcluDetMap2->Fill(y, x);
          hcharge2->Fill(ch);
          hcols2->Fill(y);
          hrows2->Fill(x);
          hsize2->Fill(float(size));
          hsizex2->Fill(float(sizeX));
          hsizey2->Fill(float(sizeY));
          numOfClustersPerDet2++;
          numOfClustersPerLay2++;

          aveCharge2 += ch;

        } else if (layer == 3) {
          hDetMap3->Fill(float(module), float(ladder));
          hcluDetMap3->Fill(y, x);
          hcharge3->Fill(ch);
          hcols3->Fill(y);
          hrows3->Fill(x);
          hsize3->Fill(float(size));
          hsizex3->Fill(float(sizeX));
          hsizey3->Fill(float(sizeY));
          numOfClustersPerDet3++;
          numOfClustersPerLay3++;

          aveCharge3 += ch;

        }  // end if bpix layer

      } else if (subid == 2) {  // endcap

        //cout<<disk<<" "<<side<<endl;
        if (disk == 1) {  // disk1 -+z
          if (side == 1)
            numOfClustersPerDisk2++;  // d1,-z
          else if (side == 2)
            numOfClustersPerDisk3++;  // d1, +z
          else
            cout << " unknown side " << side << endl;

          hcharge4->Fill(ch);
          aveCharge4 += ch;
          hsize4->Fill(float(size));
          hsizex4->Fill(float(sizeX));
          hsizey4->Fill(float(sizeY));

        } else if (disk == 2) {  // disk2 -+z

          if (side == 1)
            numOfClustersPerDisk1++;  // d2, -z
          else if (side == 2)
            numOfClustersPerDisk4++;  // d2, +z
          else
            cout << " unknown side " << side << endl;

          hcharge5->Fill(ch);
          aveCharge5 += ch;
          hsize5->Fill(float(size));
          hsizex5->Fill(float(sizeX));
          hsizey5->Fill(float(sizeY));

        } else
          cout << " unknown disk " << disk << endl;  // end fpix disk

      }  // end if barrel/forward

#endif  // HISTOS

    }  // clusters

    if (numOfClustersPerDet1 > maxClusPerDet)
      maxClusPerDet = numOfClustersPerDet1;
    if (numOfClustersPerDet2 > maxClusPerDet)
      maxClusPerDet = numOfClustersPerDet2;
    if (numOfClustersPerDet3 > maxClusPerDet)
      maxClusPerDet = numOfClustersPerDet3;

    if (printLocal) {
      if (layer == 1)
        cout << "Lay1: number of clusters per det = " << numOfClustersPerDet1 << endl;
      else if (layer == 2)
        cout << "Lay2: number of clusters per det = " << numOfClustersPerDet1 << endl;
      else if (layer == 3)
        cout << "Lay3: number of clusters per det = " << numOfClustersPerDet1 << endl;
    }  // end if printLocal

#ifdef HISTOS
    if (subid == 1) {  // barrel

      // Det histos
      if (layer == 1) {
        hladder1id->Fill(float(ladder));
        hz1id->Fill(float(module));
        ++numberOfDetUnits1;
        hclusPerDet1->Fill(float(numOfClustersPerDet1));
        hpixPerDet1->Fill(float(numOfPixPerDet1));
        if (numOfPixPerDet1 > maxPixPerDet)
          maxPixPerDet = numOfPixPerDet1;
        numOfClustersPerDet1 = 0;
        numOfPixPerDet1 = 0;
        //if(numOfPixPerLink11>798 || numOfPixPerLink12>798) select=true;
        hpixPerLink1->Fill(float(numOfPixPerLink11));
        hpixPerLink1->Fill(float(numOfPixPerLink12));
        numOfPixPerLink11 = 0;
        numOfPixPerLink12 = 0;

      } else if (layer == 2) {
        hladder2id->Fill(float(ladder));
        hz2id->Fill(float(module));
        ++numberOfDetUnits2;
        hclusPerDet2->Fill(float(numOfClustersPerDet2));
        hpixPerDet2->Fill(float(numOfPixPerDet2));
        if (numOfPixPerDet2 > maxPixPerDet)
          maxPixPerDet = numOfPixPerDet2;
        numOfClustersPerDet2 = 0;
        numOfPixPerDet2 = 0;
        hpixPerLink2->Fill(float(numOfPixPerLink21));
        hpixPerLink2->Fill(float(numOfPixPerLink22));
        numOfPixPerLink21 = 0;
        numOfPixPerLink22 = 0;

      } else if (layer == 3) {
        hladder3id->Fill(float(ladder));
        hz3id->Fill(float(module));
        ++numberOfDetUnits3;
        hclusPerDet3->Fill(float(numOfClustersPerDet3));
        hpixPerDet3->Fill(float(numOfPixPerDet3));
        if (numOfPixPerDet3 > maxPixPerDet)
          maxPixPerDet = numOfPixPerDet3;
        numOfClustersPerDet3 = 0;
        numOfPixPerDet3 = 0;
        //SK:unused	numOfPixPerLink3=0;

      }  // layer

    }  // end barrel/forward

#endif  // HISTOS

  }  // detunits loop

  if (0) {
    cout << "run " << run << " event " << event << " bx " << bx << " lumi " << lumiBlock << " orbit " << orbit
         << " num " << countEvents << endl;
    cout << "Num of pix " << numberOfPixels << " num of clus " << numberOfClusters << " num of dets " << numOf
         << " max clus per det " << maxClusPerDet << " max pix per clu " << maxPixPerClu << " count " << countEvents
         << endl;
    cout << "Number of clusters per      Lay1,2,3: " << numOfClustersPerLay1 << " " << numOfClustersPerLay2 << " "
         << numOfClustersPerLay3 << endl;
    cout << "Number of pixels per        Lay1,2,3: " << numOfPixPerLay1 << " " << numOfPixPerLay2 << " "
         << numOfPixPerLay3 << endl;
    cout << "Number of dets with clus in Lay1,2,3: " << numberOfDetUnits1 << " " << numberOfDetUnits2 << " "
         << numberOfDetUnits3 << endl;
    cout << "Number of clus in disks 1,2,3,4     : " << numOfClustersPerDisk1 << " " << numOfClustersPerDisk2 << " "
         << numOfClustersPerDisk3 << " " << numOfClustersPerDisk4 << endl;
    aveCharge1 /= float(numOfClustersPerLay1);
    aveCharge2 /= float(numOfClustersPerLay2);
    aveCharge3 /= float(numOfClustersPerLay3);
    aveCharge4 /= float(numOfClustersPerDisk2 + numOfClustersPerDisk3);
    aveCharge5 /= float(numOfClustersPerDisk1 + numOfClustersPerDisk4);
    cout << " Average charge " << aveCharge1 << " " << aveCharge2 << " " << aveCharge3 << " " << aveCharge4 << " "
         << aveCharge5 << endl;
  }

#ifdef HISTOS

  hdigis->Fill(float(numberOfPixels));  // all pix
  int tmp1 = numOfPixPerLay1 + numOfPixPerLay2 + numOfPixPerLay3;
  hdigisB->Fill(float(tmp1));  // pix in bpix
  tmp1 = numOfPixPerDisk1 + numOfPixPerDisk2 + numOfPixPerDisk3 + numOfPixPerDisk4;
  hdigisF->Fill(float(tmp1));  // pix in fpix

  hclus->Fill(float(numberOfClusters));  // clusters fpix+bpix
  tmp1 = numOfClustersPerLay1 + numOfClustersPerLay2 + numOfClustersPerLay3;
  hclusBPix->Fill(float(tmp1));  // clusters in bpix
  int tmp2 = numOfClustersPerDisk1 + numOfClustersPerDisk2 + numOfClustersPerDisk3 + numOfClustersPerDisk4;
  hclusFPix->Fill(float(tmp2));  // clusters in fpix

  hclusPerLay1->Fill(float(numOfClustersPerLay1));
  hclusPerLay2->Fill(float(numOfClustersPerLay2));
  hclusPerLay3->Fill(float(numOfClustersPerLay3));
  hpixPerLay1->Fill(float(numOfPixPerLay1));
  hpixPerLay2->Fill(float(numOfPixPerLay2));
  hpixPerLay3->Fill(float(numOfPixPerLay3));
  if (numOfClustersPerLay1 > 0)
    hdetsPerLay1->Fill(float(numberOfDetUnits1));
  if (numOfClustersPerLay2 > 0)
    hdetsPerLay2->Fill(float(numberOfDetUnits2));
  if (numOfClustersPerLay3 > 0)
    hdetsPerLay3->Fill(float(numberOfDetUnits3));

  hclusPerDisk1->Fill(float(numOfClustersPerDisk1));
  hclusPerDisk2->Fill(float(numOfClustersPerDisk2));
  hclusPerDisk3->Fill(float(numOfClustersPerDisk3));
  hclusPerDisk4->Fill(float(numOfClustersPerDisk4));

  hmaxPixPerDet->Fill(float(maxPixPerDet));

  // Select only events with more tha 3 clusters
  if (numberOfClusters > 3) {
    hbx1->Fill(float(bx));
    hlumi1->Fill(float(lumiBlock));
  }  // if num of clusters

#endif  // HISTOS

}  // end

//define this as a plug-in
DEFINE_FWK_MODULE(ReadPixClusters);
