// File: ReadPixClusters.cc
// Description: TO test the pixel clusters with tracks (full)
// Author: Danek Kotlinski 
// Creation Date:  Initial version. 3/06
//
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

#include "DataFormats/Common/interface/EDProduct.h"

//#include "DataFormats/SiPixelCluster/interface/SiPixelClusterCollection.h"
#include "DataFormats/SiPixelCluster/interface/SiPixelCluster.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHitCollection.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/DetId/interface/DetId.h"

#include "DataFormats/SiPixelDetId/interface/PXBDetId.h" 
#include "DataFormats/SiPixelDetId/interface/PXFDetId.h" 
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"

#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetUnit.h"
#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetType.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/CommonDetUnit/interface/GeomDetType.h"
#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"
//#include "Geometry/CommonTopologies/interface/PixelTopology.h"

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


// For tracks 
#include "DataFormats/TrackReco/interface/Track.h"
#include "TrackingTools/PatternTools/interface/Trajectory.h"

//#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"

#include "TrackingTools/TransientTrack/interface/TransientTrack.h"

#include "TrackingTools/PatternTools/interface/TrajTrackAssociation.h"
//#include "TrackingTools/PatternTools/interface/TrajectoryFitter.h"

#include "TrackingTools/TrackFitters/interface/TrajectoryStateCombiner.h"

//#include "TrackingTools/TrackAssociator/interface/TrackDetectorAssociator.h"
//#include "TrackingTools/TrackAssociator/interface/TrackAssociatorParameters.h"

//#include "Alignment/OfflineValidation/interface/TrackerValidationVariables.h"


//#include "TrackingTools/TrackAssociator/interface/TrackDetectorAssociator.h"
//#include "TrackingTools/PatternTools/interface/Trajectory.h"
//#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
//#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
//#include "TrackingTools/PatternTools/interface/TrajTrackAssociation.h"

#include "RecoVertex/VertexPrimitives/interface/TransientVertex.h"
#include <DataFormats/VertexReco/interface/VertexFwd.h>

// For luminisoty
#include "DataFormats/Luminosity/interface/LumiSummary.h"
#include "DataFormats/Common/interface/ConditionsInEdm.h"

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
#include <TProfile.h>
#include <TVector3.h>


#define HISTOS
//#define L1
//#define HLT

using namespace std;

class TestPixTracks : public edm::EDAnalyzer {
 public:
  
  explicit TestPixTracks(const edm::ParameterSet& conf);  
  virtual ~TestPixTracks();
  virtual void analyze(const edm::Event& e, const edm::EventSetup& c);
  virtual void beginRun(const edm::EventSetup& iSetup);
  virtual void beginJob();
  virtual void endJob();
  
 private:
  edm::ParameterSet conf_;
  edm::InputTag src_;
  //const static bool PRINT = false;
  bool PRINT;
  float countTracks, countGoodTracks, countTracksInPix, countPVs, countEvents, countLumi;  

  //TFile* hFile;
  //TH1D *hdetunit;
  //TH1D *hpixid,*hpixsubid,
  //*hlayerid,
  //*hladder1id,*hladder2id,*hladder3id,
  //*hz1id,*hz2id,*hz3id;

  TH1D *hcharge1,*hcharge2, *hcharge3, *hcharge;
  TH1D *hpixcharge1,*hpixcharge2, *hpixcharge3, *hpixcharge;
  TH1D *hcols1,*hcols2,*hcols3,*hrows1,*hrows2,*hrows3;
  TH1D *hsize1,*hsize2,*hsize3,
    *hsizex1,*hsizex2,*hsizex3,
    *hsizey1,*hsizey2,*hsizey3;

  TH1D *hclusPerTrk1,*hclusPerTrk2,*hclusPerTrk3;
  TH1D *hclusPerLay1,*hclusPerLay2,*hclusPerLay3;
  TH1D *hpixPerLay1,*hpixPerLay2,*hpixPerLay3;
  //TH1D *hdetsPerLay1,*hdetsPerLay2,*hdetsPerLay3;

  //TH1D *hdetr, *hdetz;
  //   TH1D *hcolsB,  *hrowsB,  *hcolsF,  *hrowsF;
  TH2F *hDetMap1, *hDetMap2, *hDetMap3;  // clusters 
  //TH2F *hpixDetMap1, *hpixDetMap2, *hpixDetMap3;
  TH2F *hcluDetMap1, *hcluDetMap2, *hcluDetMap3;

  TH2F *hpvxy, *hclusMap1, *hclusMap2, *hclusMap3;

  TH1D *hpvz, *hpvr, *hNumPv, *hNumPvClean;
  TH1D *hPt, *hEta, *hDz, *hD0,*hzdiff;
 
  //TH1D *hncharge1,*hncharge2, *hncharge3;
  //TH1D *hchargeMonoPix1,*hchargeMonoPix2, *hchargeMonoPix3;
  // TH1D *hnpixcharge1,*hnpixcharge2, *hnpixcharge3;
  //TH1D *htest1,*htest2,*htest3,*htest4,*htest5,*htest6,*htest7,*htest8,*htest9;
  TH1D *hl1a, *hl1t, *hlt1;

  TH1D *hclusBpix, *hpixBpix;
  TH1D *htracks, *htracksGood, *htracksGoodInPix;

  TProfile *hclumult1,  *hclumult2,  *hclumult3;
  TProfile *hclumultx1, *hclumultx2, *hclumultx3;
  TProfile *hclumulty1, *hclumulty2, *hclumulty3;
  TProfile *hcluchar1,  *hcluchar2,  *hcluchar3;
  TProfile *hpixchar1,  *hpixchar2,  *hpixchar3;

  TProfile *htracksls,  *hpvsls, *htrackslsn,  *hpvslsn, *hintgl, *hinstl, *hbeam1, *hbeam2;

  TH1D *hlumi, *hlumi0, *hbx, *hbx0;
 
};
/////////////////////////////////////////////////////////////////
// Contructor,
TestPixTracks::TestPixTracks(edm::ParameterSet const& conf) 
//  : conf_(conf), src_(conf.getParameter<edm::InputTag>( "src" )) { }
  : conf_(conf) { 

  PRINT = conf.getUntrackedParameter<bool>("Verbosity",false);
  src_ =  conf.getParameter<edm::InputTag>( "src" );
  //if(PRINT) cout<<" Construct "<<endl;

}



// Virtual destructor needed.
TestPixTracks::~TestPixTracks() { }  

// ------------ method called at the begining   ------------
void TestPixTracks::beginRun(const edm::EventSetup& iSetup) {
  cout << "BeginRun, Verbosity =  " <<PRINT<<endl;
}

// ------------ method called at the begining   ------------
void TestPixTracks::beginJob() {
  cout << "BeginJob, Verbosity " <<PRINT<<endl;

  countTracks=0.; countGoodTracks=0.; countTracksInPix=0.; countPVs=0.; countEvents=0.; countLumi=0.;  

#ifdef HISTOS

  // NEW way to use root (from 2.0.0?)
  edm::Service<TFileService> fs;

  // put here whatever you want to do at the beginning of the job
  //hFile = new TFile ( "histo.root", "RECREATE" );

  //hladder1id = fs->make<TH1D>( "hladder1id", "Ladder L1 id", 50, 0., 50.);
  //hladder2id = fs->make<TH1D>( "hladder2id", "Ladder L2 id", 50, 0., 50.);
  //hladder3id = fs->make<TH1D>( "hladder3id", "Ladder L3 id", 50, 0., 50.);
  //hz1id = fs->make<TH1D>( "hz1id", "Z-index id L1", 10, 0., 10.);
  //hz2id = fs->make<TH1D>( "hz2id", "Z-index id L2", 10, 0., 10.);
  //hz3id = fs->make<TH1D>( "hz3id", "Z-index id L3", 10, 0., 10.);

  int sizeH=20;
  float lowH = -0.5;
  float highH = 19.5;

  hclusPerTrk1 = fs->make<TH1D>( "hclusPerTrk1", "Clus per track l1",
			    sizeH, lowH, highH);
  hclusPerTrk2 = fs->make<TH1D>( "hclusPerTrk2", "Clus per track l2",
			    sizeH, lowH, highH);
  hclusPerTrk3 = fs->make<TH1D>( "hclusPerTrk3", "Clus per track l3",
			    sizeH, lowH, highH);

  sizeH=2000;
  highH = 1999.5;
  hclusPerLay1 = fs->make<TH1D>( "hclusPerLay1", "Clus per layer l1",
			    sizeH, lowH, highH);
  hclusPerLay2 = fs->make<TH1D>( "hclusPerLay2", "Clus per layer l2",
			    sizeH, lowH, highH);
  hclusPerLay3 = fs->make<TH1D>( "hclusPerLay3", "Clus per layer l3",
			    sizeH, lowH, highH);

  highH = 9999.5;
  hpixPerLay1 = fs->make<TH1D>( "hpixPerLay1", "Pix per layer l1",
			    sizeH, lowH, highH);
  hpixPerLay2 = fs->make<TH1D>( "hpixPerLay2", "Pix per layer l2",
			    sizeH, lowH, highH);
  hpixPerLay3 = fs->make<TH1D>( "hpixPerLay3", "Pix per layer l3",
			    sizeH, lowH, highH);

  //hdetsPerLay1 = fs->make<TH1D>( "hdetsPerLay1", "Full dets per layer l1",
  //			 161, -0.5, 160.5);
  //hdetsPerLay3 = fs->make<TH1D>( "hdetsPerLay3", "Full dets per layer l3",
  //			 353, -0.5, 352.5);
  //hdetsPerLay2 = fs->make<TH1D>( "hdetsPerLay2", "Full dets per layer l2",
  //			 257, -0.5, 256.5);
 
  hcharge1 = fs->make<TH1D>( "hcharge1", "Clu charge l1", 400, 0.,200.); //in ke
  hcharge2 = fs->make<TH1D>( "hcharge2", "Clu charge l2", 400, 0.,200.);
  hcharge3 = fs->make<TH1D>( "hcharge3", "Clu charge l3", 400, 0.,200.);

  //hchargeMonoPix1 = fs->make<TH1D>( "hchargeMonoPix1", "Clu charge l1 MonPix", 200, 0.,100.); //in ke
  //hchargeMonoPix2 = fs->make<TH1D>( "hchargeMonoPix2", "Clu charge l2 MonPix", 200, 0.,100.);
  //hchargeMonoPix3 = fs->make<TH1D>( "hchargeMonoPix3", "Clu charge l3 MonPix", 200, 0.,100.);
 
  //hncharge1 = fs->make<TH1D>( "hncharge1", "Noise charge l1", 200, 0.,100.); //in ke
  //hncharge2 = fs->make<TH1D>( "hncharge2", "Noise charge l2", 200, 0.,100.);
  //hncharge3 = fs->make<TH1D>( "hncharge3", "Noise charge l3", 200, 0.,100.);
 
  //hpixcharge1 = fs->make<TH1D>( "hpixcharge1", "Pix charge l1", 100, 0.,50.); //in ke
  //hpixcharge2 = fs->make<TH1D>( "hpixcharge2", "Pix charge l2", 100, 0.,50.);
  //hpixcharge3 = fs->make<TH1D>( "hpixcharge3", "Pix charge l3", 100, 0.,50.);
 
  //hnpixcharge1 = fs->make<TH1D>( "hnpixcharge1", "Noise pix charge l1", 100, 0.,50.); //in ke
  //hnpixcharge2 = fs->make<TH1D>( "hnpixcharge2", "Noise pix charge l2", 100, 0.,50.);
  //hnpixcharge3 = fs->make<TH1D>( "hnpixcharge3", "Noise pix charge l3", 100, 0.,50.);

  //hpixcharge = fs->make<TH1D>( "hpixcharge", "Clu charge", 100, 0.,50.);
  //hcharge = fs->make<TH1D>( "hcharge", "Pix charge", 100, 0.,50.);
 
  hcols1 = fs->make<TH1D>( "hcols1", "Layer 1 cols", 500,-0.5,499.5);
  hcols2 = fs->make<TH1D>( "hcols2", "Layer 2 cols", 500,-0.5,499.5);
  hcols3 = fs->make<TH1D>( "hcols3", "Layer 3 cols", 500,-0.5,499.5);
  
  hrows1 = fs->make<TH1D>( "hrows1", "Layer 1 rows", 200,-0.5,199.5);
  hrows2 = fs->make<TH1D>( "hrows2", "Layer 2 rows", 200,-0.5,199.5);
  hrows3 = fs->make<TH1D>( "hrows3", "layer 3 rows", 200,-0.5,199.5);

  hsize1 = fs->make<TH1D>( "hsize1", "layer 1 clu size",100,-0.5,99.5);
  hsize2 = fs->make<TH1D>( "hsize2", "layer 2 clu size",100,-0.5,99.5);
  hsize3 = fs->make<TH1D>( "hsize3", "layer 3 clu size",100,-0.5,99.5);
  hsizex1 = fs->make<TH1D>( "hsizex1", "lay1 clu size in x",
		      20,-0.5,19.5);
  hsizex2 = fs->make<TH1D>( "hsizex2", "lay2 clu size in x",
		      20,-0.5,19.5);
  hsizex3 = fs->make<TH1D>( "hsizex3", "lay3 clu size in x",
		      20,-0.5,19.5);
  hsizey1 = fs->make<TH1D>( "hsizey1", "lay1 clu size in y",
		      30,-0.5,29.5);
  hsizey2 = fs->make<TH1D>( "hsizey2", "lay2 clu size in y",
		      30,-0.5,29.5);
  hsizey3 = fs->make<TH1D>( "hsizey3", "lay3 clu size in y",
		      30,-0.5,29.5);
  

  hDetMap1 = fs->make<TH2F>( "hDetMap1", "layer 1 clus map",
		      9,0.,9.,23,0.,23.);
  hDetMap2 = fs->make<TH2F>( "hDetMap2", "layer 2 clus map",
		      9,0.,9.,33,0.,33.);
  hDetMap3 = fs->make<TH2F>( "hDetMap3", "layer 3 clus map",
		      9,0.,9.,45,0.,45.);
  //hpixDetMap1 = fs->make<TH2F>( "hpixDetMap1", "pix det layer 1",
  //	      416,0.,416.,160,0.,160.);
  //hpixDetMap2 = fs->make<TH2F>( "hpixDetMap2", "pix det layer 2",
  //	      416,0.,416.,160,0.,160.);
  //hpixDetMap3 = fs->make<TH2F>( "hpixDetMap3", "pix det layer 3",
  //	      416,0.,416.,160,0.,160.);
  hcluDetMap1 = fs->make<TH2F>( "hcluDetMap1", "clu det layer 1",
				416,0.,416.,160,0.,160.);
  hcluDetMap2 = fs->make<TH2F>( "hcluDetMap2", "clu det layer 1",
				416,0.,416.,160,0.,160.);
  hcluDetMap3 = fs->make<TH2F>( "hcluDetMap3", "clu det layer 1",
				416,0.,416.,160,0.,160.);

  htracksGoodInPix = fs->make<TH1D>( "htracksGoodInPix", "count good tracks in pix",2000,-0.5,1999.5);
  htracksGood = fs->make<TH1D>( "htracksGood", "count good tracks",2000,-0.5,1999.5);
  htracks = fs->make<TH1D>( "htracks", "count tracks",2000,-0.5,1999.5);
  hclusBpix = fs->make<TH1D>( "hclusBpix", "count clus in bpix",200,-0.5,1999.5);
  hpixBpix = fs->make<TH1D>( "hpixBpix", "count pixels",200,-0.5,1999.5);

  hpvxy = fs->make<TH2F>( "hpvxy", "pv xy",100,-1.,1.,100,-1.,1.);
  hpvz = fs->make<TH1D>( "hpvz", "pv z",1000,-50.,50.);
  hpvr = fs->make<TH1D>( "hpvr", "pv r",100,0.,1.);
  hNumPv = fs->make<TH1D>( "hNumPv", "num of pv",100,0.,100.);
  hNumPvClean = fs->make<TH1D>( "hNumPvClean", "num of pv clean",100,0.,100.);

  hPt = fs->make<TH1D>( "hPt", "pt",100,0.,100.);
  hEta = fs->make<TH1D>( "hEta", "eta",50,-2.5,2.5);
  hD0 = fs->make<TH1D>( "hD0", "d0",500,0.,5.);
  hDz = fs->make<TH1D>( "hDz", "pt",250,-25.,25.);
  hzdiff = fs->make<TH1D>( "hzdiff", "PVz-Trackz",200,-10.,10.);

  hl1a    = fs->make<TH1D>("hl1a",   "l1a",   128,-0.5,127.5);
  hl1t    = fs->make<TH1D>("hl1t",   "l1t",   128,-0.5,127.5);
  hlt1    = fs->make<TH1D>("hlt1","hlt1",256,-0.5,255.5);

   hclumult1 = fs->make<TProfile>("hclumult1","cluster size layer 1",60,-3.,3.,0.0,100.);
   hclumult2 = fs->make<TProfile>("hclumult2","cluster size layer 2",60,-3.,3.,0.0,100.);
   hclumult3 = fs->make<TProfile>("hclumult3","cluster size layer 3",60,-3.,3.,0.0,100.);

   hclumultx1 = fs->make<TProfile>("hclumultx1","cluster x-size layer 1",60,-3.,3.,0.0,100.);
   hclumultx2 = fs->make<TProfile>("hclumultx2","cluster x-size layer 2",60,-3.,3.,0.0,100.);
   hclumultx3 = fs->make<TProfile>("hclumultx3","cluster x-size layer 3",60,-3.,3.,0.0,100.);

   hclumulty1 = fs->make<TProfile>("hclumulty1","cluster y-size layer 1",60,-3.,3.,0.0,100.);
   hclumulty2 = fs->make<TProfile>("hclumulty2","cluster y-size layer 2",60,-3.,3.,0.0,100.);
   hclumulty3 = fs->make<TProfile>("hclumulty3","cluster y-size layer 3",60,-3.,3.,0.0,100.);

   hcluchar1 = fs->make<TProfile>("hcluchar1","cluster char layer 1",60,-3.,3.,0.0,1000.);
   hcluchar2 = fs->make<TProfile>("hcluchar2","cluster char layer 2",60,-3.,3.,0.0,1000.);
   hcluchar3 = fs->make<TProfile>("hcluchar3","cluster char layer 3",60,-3.,3.,0.0,1000.);

   hpixchar1 = fs->make<TProfile>("hpixchar1","pix char layer 1",60,-3.,3.,0.0,1000.);
   hpixchar2 = fs->make<TProfile>("hpixchar2","pix char layer 2",60,-3.,3.,0.0,1000.);
   hpixchar3 = fs->make<TProfile>("hpixchar3","pix char layer 3",60,-3.,3.,0.0,1000.);

   hintgl  = fs->make<TProfile>("hintgl", "inst lumi vs ls ",1000,0.,3000.,0.0,10000.);
   hinstl  = fs->make<TProfile>("hinstl", "intg lumi vs ls ",1000,0.,3000.,0.0,100.);
   hbeam1  = fs->make<TProfile>("hbeam1", "beam1 vs ls ",1000,0.,3000.,0.0,1000.);
   hbeam2  = fs->make<TProfile>("hbeam2", "beam2 vs ls ",1000,0.,3000.,0.0,1000.);

   htracksls = fs->make<TProfile>("htracksls","tracks with pix hits  vs ls",1000,0.,3000.,0.0,10000.);
   hpvsls = fs->make<TProfile>("hpvsls","pvs  vs ls",1000,0.,3000.,0.0,1000.);
   htrackslsn = fs->make<TProfile>("htrackslsn","tracks with pix hits/lumi  vs ls",1000,0.,3000.,0.0,10000.);
   hpvslsn = fs->make<TProfile>("hpvslsn","pvs/lumi  vs ls",1000,0.,3000.,0.0,1000.);

   hlumi0  = fs->make<TH1D>("hlumi0", "lumi", 2000,0,2000.);
   hlumi   = fs->make<TH1D>("hlumi", "lumi",   2000,0,2000.);
   hbx0    = fs->make<TH1D>("hbx0",   "bx",   4000,0,4000.);  
   hbx    = fs->make<TH1D>("hbx",   "bx",     4000,0,4000.);  

   hclusMap1 = fs->make<TH2F>("hclusMap1","clus - lay1",260,-26.,26.,350,-3.5,3.5);
   hclusMap2 = fs->make<TH2F>("hclusMap2","clus - lay2",260,-26.,26.,350,-3.5,3.5);
   hclusMap3 = fs->make<TH2F>("hclusMap3","clus - lay3",260,-26.,26.,350,-3.5,3.5);

#endif

}
// ------------ method called to at the end of the job  ------------
void TestPixTracks::endJob(){
  cout << " End PixelTracksTest " << endl;

  if(countEvents>0.) {
    countTracks /= countEvents;
    countGoodTracks /= countEvents;
    countTracksInPix /= countEvents;
    countPVs /= countEvents;
    countLumi /= 1000.;
    cout<<" Average tracks/event "<< countTracks<<" good "<< countGoodTracks<<" in pix "<< countTracksInPix
	<<" PVs "<< countPVs<<" events " << countEvents<<" lumi pb-1 "<< countLumi<<"/10, bug!"<<endl;  
  }

}
//////////////////////////////////////////////////////////////////
// Functions that gets called by framework every event
void TestPixTracks::analyze(const edm::Event& e, 
			    const edm::EventSetup& es) {


  using namespace edm;
  using namespace reco;
  static int lumiBlockOld = -9999;

  const float CLU_SIZE_PT_CUT = 1.;

  int trackNumber = 0;
  int countNiceTracks=0;
  int countPixTracks=0;

  int numberOfDetUnits = 0;
  int numberOfClusters = 0;
  int numberOfPixels   = 0;

  int numberOfDetUnits1 = 0;
  int numOfClustersPerDet1=0;        
  int numOfClustersPerLay1=0;        
  int numOfPixelsPerLay1=0;        

  int numberOfDetUnits2 = 0;
  int numOfClustersPerDet2=0;        
  int numOfClustersPerLay2=0;        
  int numOfPixelsPerLay2=0;        

  int numberOfDetUnits3 = 0;
  int numOfClustersPerDet3=0;        
  int numOfClustersPerLay3=0;        
  int numOfPixelsPerLay3=0;     

  int run       = e.id().run();
  int event     = e.id().event();
  int lumiBlock = e.luminosityBlock();
  int bx        = e.bunchCrossing();
  int orbit     = e.orbitNumber();

  if(PRINT) cout<<"Run "<<run<<" Event "<<event<<" LS "<<lumiBlock<<endl;

  hbx0->Fill(float(bx));
  hlumi0->Fill(float(lumiBlock));

  edm::LuminosityBlock const& iLumi = e.getLuminosityBlock();
  edm::Handle<LumiSummary> lumi;
  iLumi.getByLabel("lumiProducer", lumi);
  edm::Handle<edm::ConditionsInLumiBlock> cond;
  float intlumi = 0, instlumi=0;
  int beamint1=0, beamint2=0;
  iLumi.getByLabel("conditionsInEdm", cond);
  // This will only work when running on RECO until (if) they fix it in the FW
  // When running on RAW and reconstructing, the LumiSummary will not appear
  // in the event before reaching endLuminosityBlock(). Therefore, it is not
  // possible to get this info in the event
  if (lumi.isValid()) {
    intlumi =(lumi->intgRecLumi())/1000.;  // 10^30 -> 10^33/cm2/sec  ->  1/nb/sec
    instlumi=(lumi->avgInsDelLumi())/1000.;
    beamint1=(cond->totalIntensityBeam1)/1000;
    beamint2=(cond->totalIntensityBeam2)/1000;
  } else {
    //std::cout << "** ERROR: Event does not get lumi info\n";
  }

  hinstl->Fill(float(lumiBlock),float(instlumi));
  hintgl->Fill(float(lumiBlock),float(intlumi));
  hbeam1->Fill(float(lumiBlock),float(beamint1));
  hbeam2->Fill(float(lumiBlock),float(beamint2));




#ifdef L1
  // Get L1
  Handle<L1GlobalTriggerReadoutRecord> L1GTRR;
  e.getByLabel("gtDigis",L1GTRR);

  if (L1GTRR.isValid()) {
    //bool l1a = L1GTRR->decision();  // global decission?
    //cout<<" L1 status :"<<l1a<<" "<<hex;
    for (unsigned int i = 0; i < L1GTRR->decisionWord().size(); ++i) {
      int l1flag = L1GTRR->decisionWord()[i]; 
      int t1flag = L1GTRR->technicalTriggerWord()[i]; 
    
      if( l1flag>0 ) hl1a->Fill(float(i));
      if( t1flag>0 && i<64) hl1t->Fill(float(i));
    } // for loop
  } // if l1a
#endif

#ifdef HLT

  bool hlt[256];
  for(int i=0;i<256;++i) hlt[i]=false;

  edm::TriggerNames TrigNames;
  edm::Handle<edm::TriggerResults> HLTResults;

  // Extract the HLT results
  e.getByLabel(edm::InputTag("TriggerResults","","HLT"),HLTResults);
  if ((HLTResults.isValid() == true) && (HLTResults->size() > 0)) {

    //TrigNames.init(*HLTResults);
    const edm::TriggerNames & TrigNames = e.triggerNames(*HLTResults);

    //cout<<TrigNames.triggerNames().size()<<endl;

    for (unsigned int i = 0; i < TrigNames.triggerNames().size(); i++) {  // loop over trigger
      //if(countAllEvents==1) cout<<i<<" "<<TrigNames.triggerName(i)<<endl;

      if ( 
           (HLTResults->wasrun(TrigNames.triggerIndex(TrigNames.triggerName(i))) == true) &&
           (HLTResults->accept(TrigNames.triggerIndex(TrigNames.triggerName(i))) == true) &&
           (HLTResults->error( TrigNames.triggerIndex(TrigNames.triggerName(i))) == false) ) {

        hlt[i]=true;
        hlt1->Fill(float(i));

      } // if hlt

    } // loop 
  } // if valid
#endif


  // -- Does this belong into beginJob()?
  //ESHandle<TrackerGeometry> TG;
  //iSetup.get<TrackerDigiGeometryRecord>().get(TG);
  //const TrackerGeometry* theTrackerGeometry = TG.product();
  //const TrackerGeometry& theTracker(*theTrackerGeometry);
  // Get event setup 
  edm::ESHandle<TrackerGeometry> geom;
  es.get<TrackerDigiGeometryRecord>().get( geom );
  const TrackerGeometry& theTracker(*geom);


  // -- Primary vertices
  // ----------------------------------------------------------------------
  edm::Handle<reco::VertexCollection> vertices;
  e.getByLabel("offlinePrimaryVertices", vertices);

  if(PRINT) cout<<" PV list "<<vertices->size()<<endl;
  int pvNotFake = 0, pvsTrue = 0;
  vector<float> pvzVector;
  for (reco::VertexCollection::const_iterator iv = vertices->begin(); iv != vertices->end(); ++iv) {

    if( (iv->isFake()) == 1 ) continue; 
      pvNotFake++;
      float pvx = iv->x();
      float pvy = iv->y();
      float pvz = iv->z();
      int numTracksPerPV = iv->tracksSize();
      //int numTracksPerPV = iv->nTracks();

      //float xe = iv->xError();
      //float ye = iv->yError();
      //float ze = iv->zError();      
      //int chi2 = iv->chi2(); 
      //int dof = iv->ndof(); 

      if(PRINT) cout<<" PV "<<pvNotFake<<" pos = "<<pvx<<"/"<<pvy<<"/"<<pvz
		    <<", Num of tracks "<<numTracksPerPV<<endl;

      hpvz->Fill(pvz);
      if(pvz>-22. && pvz<22.) {
	float pvr = sqrt(pvx*pvx + pvy*pvy); 
	hpvxy->Fill(pvx,pvy);
	hpvr->Fill(pvr);
	if(pvr<0.3) {
	  pvsTrue++;
	  pvzVector.push_back(pvz);
	  //if(PRINT) cout<<"PV "<<pvsTrue<<" "<<pvz<<endl;
	} //pvr
      } // pvz
     
      //if(pvsTrue<1) continue; // skip events with no PV

  } // loop pvs
  hNumPv->Fill(float(pvNotFake));
  hNumPvClean->Fill(float(pvsTrue));

  if(PRINT) cout<<" Not fake PVs = "<<pvNotFake<<" good position "<<pvsTrue<<endl;
   
  Handle<reco::TrackCollection> recTracks;
  // e.getByLabel("generalTracks", recTracks);
  // e.getByLabel("ctfWithMaterialTracksP5", recTracks);
  // e.getByLabel("splittedTracksP5", recTracks);
  //e.getByLabel("cosmictrackfinderP5", recTracks);
  e.getByLabel(src_ , recTracks);


  if(PRINT) cout<<" Tracks "<<recTracks->size()<<endl;
  for(reco::TrackCollection::const_iterator t=recTracks->begin();
      t!=recTracks->end(); ++t){

    trackNumber++;
    numOfClustersPerDet1=0;        
    numOfClustersPerDet2=0;        
    numOfClustersPerDet3=0;        
    int pixelHits=0;
    
    int size = t->recHitsSize();
    float pt = t->pt();
    float eta = t->eta();
    float phi = t->phi();
    float trackCharge = t->charge();
    float d0 = t->d0();
    float dz = t->dz();
    float tkvx = t->vx();
    float tkvy = t->vy();
    float tkvz = t->vz();

    if(PRINT) cout<<"Track "<<trackNumber<<" Pt "<<pt<<" Eta "<<eta<<" d0/dz "<<d0<<" "<<dz
		  <<" Hits "<<size<<endl;

    hEta->Fill(eta);
    hDz->Fill(dz);
    if(abs(eta)>2.8 || abs(dz)>25.) continue;  //  skip  
    
    hD0->Fill(d0);
    if(d0>1.0) continue; // skip 
    
    bool goodTrack = false;
    for(vector<float>::iterator m=pvzVector.begin(); m!=pvzVector.end();++m) {
      float z = *m;
      float tmp = abs(z-dz);
      hzdiff->Fill(tmp);
      if( tmp < 1.) goodTrack=true; 
    }
    
    if(!goodTrack) continue;
    countNiceTracks++;      
    hPt->Fill(pt);
        
    // Loop over rechits
    for ( trackingRecHit_iterator recHit = t->recHitsBegin();
	  recHit != t->recHitsEnd(); ++recHit ) {
      
      if ( !((*recHit)->isValid()) ) continue;

      if( (*recHit)->geographicalId().det() != DetId::Tracker ) continue; 

      const DetId & hit_detId = (*recHit)->geographicalId();
      uint IntSubDetID = (hit_detId.subdetId());

      // Select pixel rechits	
      if(IntSubDetID == 0 ) continue;  // Select ??
      if(IntSubDetID != PixelSubdetector::PixelBarrel) continue; // look only at bpix || IntSubDetID == PixelSubdetector::PixelEndcap) {

      // Pixel detector
      PXBDetId pdetId = PXBDetId(hit_detId);
      //unsigned int detTypeP=pdetId.det();
      //unsigned int subidP=pdetId.subdetId();
      // Barell layer = 1,2,3
      int layer=pdetId.layer();
      // Barrel ladder id 1-20,32,44.
      int ladder=pdetId.ladder();
      // Barrel Z-index=1,8
      int zindex=pdetId.module();

      if(PRINT) cout<<"barrel layer/ladder/module: "<<layer<<"/"<<ladder<<"/"<<zindex<<endl;

      // Get the geom-detector
      const PixelGeomDetUnit * theGeomDet =
	dynamic_cast<const PixelGeomDetUnit*> (theTracker.idToDet(hit_detId) );
      double detZ = theGeomDet->surface().position().z();
      double detR = theGeomDet->surface().position().perp();
      const PixelTopology * topol = &(theGeomDet->specificTopology());  // pixel topology
      
      //std::vector<SiStripRecHit2D*> output = getRecHitComponents((*recHit).get()); 
      //std::vector<SiPixelRecHit*> TrkComparison::getRecHitComponents(const TrackingRecHit* rechit){

      const SiPixelRecHit* hit = dynamic_cast<const SiPixelRecHit*>((*recHit).get());
      //edm::Ref<edmNew::DetSetVector<SiStripCluster> ,SiStripCluster> cluster = hit->cluster();
      // get the edm::Ref to the cluster

      if(hit) {

	// RecHit (recthits are transient, so not available without ttrack refit)
 	//double xloc = hit->localPosition().x();// 1st meas coord
 	//double yloc = hit->localPosition().y();// 2nd meas coord or zero
 	//double zloc = hit->localPosition().z();// up, always zero
	//cout<<" rechit loc "<<xloc<<" "<<yloc<<endl;
	
	edm::Ref<edmNew::DetSetVector<SiPixelCluster>, SiPixelCluster> const& clust = hit->cluster();
	//  check if the ref is not null
	if (!clust.isNonnull()) continue;

	  numberOfClusters++;
	  pixelHits++;
	  float charge = (clust->charge())/1000.0; // convert electrons to kilo-electrons
	  int size = clust->size();
	  int sizeX = clust->sizeX();
	  int sizeY = clust->sizeY();
	  float row = clust->x();
	  float col = clust->y();
	  numberOfPixels += size;

	  //cout<<" clus loc "<<row<<" "<<col<<endl;

	  if(PRINT) cout<<" cluster "<<numberOfClusters<<" charge = "<<charge<<" size = "<<size<<endl;


	  LocalPoint lp = topol->localPosition( MeasurementPoint( clust->x(), clust->y() ) );
	  //float x = lp.x();
	  //float y = lp.y();
	  //cout<<" clu loc "<<x<<" "<<y<<endl;

	  GlobalPoint clustgp = theGeomDet->surface().toGlobal( lp );
	  double gX = clustgp.x();
	  double gY = clustgp.y();
	  double gZ = clustgp.z();

	  //cout<<" CLU GLOBAL "<<gX<<" "<<gY<<" "<<gZ<<endl;

	  TVector3 v(gX,gY,gZ);
	  float phi = v.Phi();



	  //int maxPixelCol = clust->maxPixelCol();
	  //int maxPixelRow = clust->maxPixelRow();
	  //int minPixelCol = clust->minPixelCol();
	  //int minPixelRow = clust->minPixelRow();
	  //int geoId = PixGeom->geographicalId().rawId();
	  // Replace with the topology methods
	  // edge method moved to topologi class
	  //int edgeHitX = (int) ( topol->isItEdgePixelInX( minPixelRow ) || topol->isItEdgePixelInX( maxPixelRow ) );
	  //int edgeHitY = (int) ( topol->isItEdgePixelInY( minPixelCol ) || topol->isItEdgePixelInY( maxPixelCol ) );

	  // calculate alpha and beta from cluster position
	  //LocalTrajectoryParameters ltp = tsos.localParameters();
	  //LocalVector localDir = ltp.momentum()/ltp.momentum().mag();
	  //float locx = localDir.x();
	  //float locy = localDir.y();
	  //float locz = localDir.z();
	  //float loctheta = localDir.theta(); // currently unused
	  //float alpha = atan2( locz, locx );
	  //float beta = atan2( locz, locy );

	  if(layer==1) {
	
	    hDetMap1->Fill(float(zindex),float(ladder)); 
	    hcluDetMap1->Fill(col,row);
	    hcharge1->Fill(charge);
	    hcols1->Fill(col);
	    hrows1->Fill(row);

	    hclusMap1->Fill(gZ,phi);

	    if(pt>CLU_SIZE_PT_CUT) {
	      hsize1->Fill(float(size));
	      hsizex1->Fill(float(sizeX));
	      hsizey1->Fill(float(sizeY));
	      
	      hclumult1->Fill(eta,float(size));
	      hclumultx1->Fill(eta,float(sizeX));
	      hclumulty1->Fill(eta,float(sizeY));
	      hcluchar1->Fill(eta,float(charge));
	    }

	    numOfClustersPerDet1++;
	    numOfClustersPerLay1++;
	    numOfPixelsPerLay1 += size;     
   	
	  } else if(layer==2) {
	
	    hDetMap2->Fill(float(zindex),float(ladder));
	    hcluDetMap2->Fill(col,row);
	    hcharge2->Fill(charge);
	    hcols2->Fill(col);
	    hrows2->Fill(row);

	    hclusMap2->Fill(gZ,phi);

	    if(pt>CLU_SIZE_PT_CUT) {
	      hsize2->Fill(float(size));
	      hsizex2->Fill(float(sizeX));
	      hsizey2->Fill(float(sizeY));
	      
	      hclumult2->Fill(eta,float(size));
	      hclumultx2->Fill(eta,float(sizeX));
	      hclumulty2->Fill(eta,float(sizeY));
	      hcluchar2->Fill(eta,float(charge));
	    }

	    numOfClustersPerDet2++;
	    numOfClustersPerLay2++;
	    numOfPixelsPerLay2 += size;     
	    
	  } else if(layer==3) {
	
	    hDetMap3->Fill(float(zindex),float(ladder));
	    hcluDetMap3->Fill(col,row);
	    hcharge3->Fill(charge);
	    hcols3->Fill(col);
	    hrows3->Fill(row);

	    hclusMap3->Fill(gZ,phi);

	    if(pt>CLU_SIZE_PT_CUT) {
	      hsize3->Fill(float(size));
	      hsizex3->Fill(float(sizeX));
	      hsizey3->Fill(float(sizeY));
	      hclumult3->Fill(eta,float(size));
	      hclumultx3->Fill(eta,float(sizeX));
	      hclumulty3->Fill(eta,float(sizeY));
	      hcluchar3->Fill(eta,float(charge));
	    }

	    numOfClustersPerDet3++;
	    numOfClustersPerLay3++;
	    numOfPixelsPerLay3 += size;     
	  } // if layer
	  
      } // if valid

    } // clusters

    if(pixelHits>0) countPixTracks++;

    if(PRINT) cout<<" Clusters for track "<<trackNumber<<" num of clusters "<<numberOfClusters
		  <<" num of pixels "<<pixelHits<<endl;

#ifdef HISTOS
    if(numberOfClusters>0) { 
      hclusPerTrk1->Fill(float(numOfClustersPerDet1));
      if(PRINT) cout<<"Lay1: number of clusters per track = "<<numOfClustersPerDet1<<endl;
      hclusPerTrk2->Fill(float(numOfClustersPerDet2));
      if(PRINT) cout<<"Lay2: number of clusters per track = "<<numOfClustersPerDet1<<endl;
      hclusPerTrk3->Fill(float(numOfClustersPerDet3));
      if(PRINT) cout<<"Lay3: number of clusters per track = "<<numOfClustersPerDet1<<endl;
    }
#endif // HISTOS


  } // tracks


#ifdef HISTOS
  if(numberOfClusters>0) {
    hclusPerLay1->Fill(float(numOfClustersPerLay1));
    hclusPerLay2->Fill(float(numOfClustersPerLay2));
    hclusPerLay3->Fill(float(numOfClustersPerLay3));
    //hdetsPerLay1->Fill(float(numberOfDetUnits1));
    //hdetsPerLay2->Fill(float(numberOfDetUnits2));
    //hdetsPerLay3->Fill(float(numberOfDetUnits3));
    //int tmp = numberOfDetUnits1+numberOfDetUnits2+numberOfDetUnits3;
    hpixPerLay1->Fill(float(numOfPixelsPerLay1));
    hpixPerLay2->Fill(float(numOfPixelsPerLay2));
    hpixPerLay3->Fill(float(numOfPixelsPerLay3));
    //htest7->Fill(float(tmp));
    hclusBpix->Fill(float(numberOfClusters));
    hpixBpix->Fill(float(numberOfPixels));

  }
  htracksGood->Fill(float(countNiceTracks));
  htracksGoodInPix->Fill(float(countPixTracks));
  htracks->Fill(float(trackNumber));

  hbx->Fill(float(bx));
  hlumi->Fill(float(lumiBlock));

  htracksls->Fill(float(lumiBlock),float(countPixTracks));
  hpvsls->Fill(float(lumiBlock),float(pvsTrue));
  if(instlumi>0.) {
    float tmp = float(countPixTracks)/instlumi;
    htrackslsn->Fill(float(lumiBlock),tmp);
    tmp = float(pvsTrue)/instlumi;
    hpvslsn->Fill(float(lumiBlock),tmp);
  }

#endif // HISTOS

  //
  countTracks += float(trackNumber);
  countGoodTracks += float(countNiceTracks);
  countTracksInPix += float(countPixTracks);
  countPVs += float(pvsTrue);
  countEvents++;
  if(lumiBlock != lumiBlockOld) {
    countLumi += intlumi; 
    lumiBlockOld = lumiBlock;
  }


  if(PRINT) cout<<" event with tracks = "<<trackNumber<<" "<<countNiceTracks<<endl;

  return;





#ifdef USE_TRAJ

  //------------------------------------------------------------------------------------
  // Use Trajectories

  edm::Handle<TrajTrackAssociationCollection> trajTrackCollectionHandle;
  e.getByLabel(conf_.getParameter<std::string>("trajectoryInput"),trajTrackCollectionHandle);

  TrajectoryStateCombiner tsoscomb;
 
  int NbrTracks =  trajTrackCollectionHandle->size();
  std::cout << " track measurements " << trajTrackCollectionHandle->size()  << std::endl;

  int trackNumber = 0;
  int numberOfClusters = 0;

  for(TrajTrackAssociationCollection::const_iterator it = trajTrackCollectionHandle->begin(), itEnd = trajTrackCollectionHandle->end(); it!=itEnd;++it){

    int pixelHits = 0;
    int stripHits = 0;
    const Track&      track = *it->val;
    const Trajectory& traj  = *it->key;
   
    std::vector<TrajectoryMeasurement> checkColl = traj.measurements();
    for(std::vector<TrajectoryMeasurement>::const_iterator checkTraj = checkColl.begin();
	checkTraj != checkColl.end(); ++checkTraj) {

      if(! checkTraj->updatedState().isValid()) continue;
      TransientTrackingRecHit::ConstRecHitPointer testhit = checkTraj->recHit();
      if(! testhit->isValid() || testhit->geographicalId().det() != DetId::Tracker ) continue;
      uint testSubDetID = (testhit->geographicalId().subdetId());
      if(testSubDetID == PixelSubdetector::PixelBarrel || testSubDetID == PixelSubdetector::PixelEndcap) pixelHits++;
      else if (testSubDetID == StripSubdetector::TIB || testSubDetID == StripSubdetector::TOB ||
	       testSubDetID == StripSubdetector::TID || testSubDetID == StripSubdetector::TEC) stripHits++;

    }


    if (pixelHits == 0) continue;

    trackNumber++;
    std::cout << " track " << trackNumber <<" has pixelhits "<<pixelHits << std::endl;
    pixelHits = 0;

    //std::vector<TrajectoryMeasurement> tmColl = traj.measurements();
    for(std::vector<TrajectoryMeasurement>::const_iterator itTraj = checkColl.begin();
	itTraj != checkColl.end(); ++itTraj) {
      if(! itTraj->updatedState().isValid()) continue;

      TrajectoryStateOnSurface tsos = tsoscomb( itTraj->forwardPredictedState(), itTraj->backwardPredictedState() );
      TransientTrackingRecHit::ConstRecHitPointer hit = itTraj->recHit();
      if(! hit->isValid() || hit->geographicalId().det() != DetId::Tracker ) continue; 

      const DetId & hit_detId = hit->geographicalId();
      uint IntSubDetID = (hit_detId.subdetId());
	
      if(IntSubDetID == 0 ) continue;  // Select ??
      if(IntSubDetID != PixelSubdetector::PixelBarrel) continue; // look only at bpix || IntSubDetID == PixelSubdetector::PixelEndcap) {


//       const GeomDetUnit* detUnit = hit->detUnit();
//       if(detUnit) {
// 	const Surface& surface = hit->detUnit()->surface();
// 	const TrackerGeometry& theTracker(*tkGeom_);
// 	const PixelGeomDetUnit* theGeomDet = dynamic_cast<const PixelGeomDetUnit*> (theTracker.idToDet(hit_detId) );
// 	const RectangularPixelTopology * topol = dynamic_cast<const RectangularPixelTopology*>(&(theGeomDet->specificTopology()));
//       }


      // get the enclosed persistent hit
      const TrackingRecHit *persistentHit = hit->hit();
      // check if it's not null, and if it's a valid pixel hit
      if ((persistentHit != 0) && (typeid(*persistentHit) == typeid(SiPixelRecHit))) {
	// tell the C++ compiler that the hit is a pixel hit
	const SiPixelRecHit* pixhit = dynamic_cast<const SiPixelRecHit*>( hit->hit() );
	// get the edm::Ref to the cluster
	edm::Ref<edmNew::DetSetVector<SiPixelCluster>, SiPixelCluster> const& clust = (*pixhit).cluster();
	//  check if the ref is not null
	if (clust.isNonnull()) {

	  numberOfClusters++;
	  pixelHits++;
	  float charge = (clust->charge())/1000.0; // convert electrons to kilo-electrons
	  int size = clust->size();
	  int size_x = clust->sizeX();
	  int size_y = clust->sizeY();
	  float row = clust->x();
	  float col = clust->y();
  
	  //LocalPoint lp = topol->localPosition(MeasurementPoint(clust_.row,clust_.col));
	  //float x = lp.x();
	  //float y = lp.y();
  
	  int maxPixelCol = clust->maxPixelCol();
	  int maxPixelRow = clust->maxPixelRow();
	  int minPixelCol = clust->minPixelCol();
	  int minPixelRow = clust->minPixelRow();
  
	  //int geoId = PixGeom->geographicalId().rawId();

	  // Replace with the topology methods
	  // edge method moved to topologi class
	  //int edgeHitX = (int) ( topol->isItEdgePixelInX( minPixelRow ) || topol->isItEdgePixelInX( maxPixelRow ) );
	  //int edgeHitY = (int) ( topol->isItEdgePixelInY( minPixelCol ) || topol->isItEdgePixelInY( maxPixelCol ) );

	  // calculate alpha and beta from cluster position
	  //LocalTrajectoryParameters ltp = tsos.localParameters();
	  //LocalVector localDir = ltp.momentum()/ltp.momentum().mag();

	  //float locx = localDir.x();
	  //float locy = localDir.y();
	  //float locz = localDir.z();
	  //float loctheta = localDir.theta(); // currently unused

	  //float alpha = atan2( locz, locx );
	  //float beta = atan2( locz, locy );

	  //clust_.normalized_charge = clust_.charge*sqrt(1.0/(1.0/pow(tan(clust_.clust_alpha),2)+1.0/pow(tan(clust_.clust_beta),2)+1.0));
	}  // valid cluster
      } // valid peristant hit

    } // loop over trajectory meas.

    if(PRINT) cout<<" Cluster for track "<<trackNumber<<" cluaters "<<numberOfClusters<<" "<<pixelHits<<endl;

  } // loop over tracks

#endif // USE_TRAJ


  cout<<" event with tracks = "<<trackNumber<<" "<<countGoodTracks<<endl;

} // end 


//define this as a plug-in
DEFINE_FWK_MODULE(TestPixTracks);
