// File: ReadPixClusters.cc
// Description: T0 test the pixel clusters. 
// Author: Danek Kotlinski 
// Creation Date:  Initial version. 3/06
// Modify to work with CMSSW354, 11/03/10 d.k.
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
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/DetId/interface/DetId.h"

#include "DataFormats/SiPixelDetId/interface/PXBDetId.h" 
#include "DataFormats/SiPixelDetId/interface/PXFDetId.h" 
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "DataFormats/SiPixelDetId/interface/PixelBarrelName.h"

#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetUnit.h"
#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetType.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/CommonDetUnit/interface/GeomDetType.h"
#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"
#include "Geometry/TrackerTopology/interface/RectangularPixelTopology.h"

// For L1
#include "L1Trigger/GlobalTriggerAnalyzer/interface/L1GtUtils.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutSetupFwd.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutSetup.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutRecord.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerObjectMapRecord.h"

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


#define HISTOS
#define L1

using namespace std;

class ReadPixClusters : public edm::EDAnalyzer {
 public:
  
  explicit ReadPixClusters(const edm::ParameterSet& conf);  
  virtual ~ReadPixClusters();
  virtual void analyze(const edm::Event& e, const edm::EventSetup& c);
  virtual void beginRun(const edm::EventSetup& iSetup);
  virtual void beginJob();
  virtual void endJob();
  
 private:
  edm::ParameterSet conf_;
  edm::InputTag src_;
  //const static bool PRINT = false;
  bool PRINT;
  int countEvents, countAllEvents;
  float sumClusters;

  //TFile* hFile;
  TH1F *hdetunit;
  TH1F *hpixid,*hpixsubid,
    *hlayerid,
    *hladder1id,*hladder2id,*hladder3id,
    *hz1id,*hz2id,*hz3id;

  TH1F *hcharge1,*hcharge2, *hcharge3;
  TH1F *hpixcharge1,*hpixcharge2, *hpixcharge3;
  TH1F *hcols1,*hcols2,*hcols3,*hrows1,*hrows2,*hrows3;
  TH1F *hsize1,*hsize2,*hsize3,
    *hsizex1,*hsizex2,*hsizex3,
    *hsizey1,*hsizey2,*hsizey3;

  TH1F *hclusPerDet1,*hclusPerDet2,*hclusPerDet3;
  TH1F *hpixPerDet1,*hpixPerDet2,*hpixPerDet3;
  TH1F *hclusPerLay1,*hclusPerLay2,*hclusPerLay3;
  TH1F *hpixPerLay1,*hpixPerLay2,*hpixPerLay3;
  TH1F *hdetsPerLay1,*hdetsPerLay2,*hdetsPerLay3;
  TH1F *hclus, *hclusClean, *hdigis;
  TH1F *hclus1,*hclus2,*hclus3,*hclus4,*hclus5,*hclus6,*hclus7,*hclus8,*hclus9,
    *hclus10,*hclus11,*hclus12;

  TH1F *hdetr, *hdetz;
//   TH1F *hcolsB,  *hrowsB,  *hcolsF,  *hrowsF;
//   TH2F *htest1, *htest2;
   TH2F *hDetMap1, *hDetMap2, *hDetMap3;
   TH2F *hpDetMap1, *hpDetMap2, *hpDetMap3;
   TH2F *hpixDetMap1, *hpixDetMap2, *hpixDetMap3, *hpixDetMapNoise;
   TH2F *hcluDetMap1, *hcluDetMap2, *hcluDetMap3;

  TH1F *hncharge1,*hncharge2, *hncharge3;
  TH1F *hnpixcharge1,*hnpixcharge2,*hnpixcharge3;
  TH2F *hpixDetMap10, *hpixDetMap20, *hpixDetMap30;
  TH2F *hpixDetMap11, *hpixDetMap12, *hpixDetMap13, *hpixDetMap14;

  TH1F *hevent, *hlumi, *horbit, *hbx, *hl1a, *hl1t, *hevent0, *hlumi0;
  TH1F *htest;
};
/////////////////////////////////////////////////////////////////
// Contructor, empty.
ReadPixClusters::ReadPixClusters(edm::ParameterSet const& conf) 
  : conf_(conf), src_(conf.getParameter<edm::InputTag>( "src" )) { 
  PRINT = conf.getUntrackedParameter<bool>("Verbosity",false);
  //src_ =  conf.getParameter<edm::InputTag>( "src" );
  if(PRINT) cout<<" Construct "<<endl;

}
// Virtual destructor needed.
ReadPixClusters::~ReadPixClusters() { }  

// ------------ method called at the begining   ------------
void ReadPixClusters::beginRun(const edm::EventSetup& iSetup) {
  cout << "beginRun -  PixelClusterTest " <<endl;
}

// ------------ method called at the begining   ------------
void ReadPixClusters::beginJob() {
  cout << "Initialize PixelClusterTest " <<endl;
 
#ifdef HISTOS


 // NEW way to use root (from 2.0.0?)
  edm::Service<TFileService> fs;

  //hdetunit = new TH1F( "hdetunit", "Det unit", 1000,
  //                            302000000.,302300000.);
  //hpixid = new TH1F( "hpixid", "Pix det id", 10, 0., 10.);
  //hpixsubid = new TH1F( "hpixsubid", "Pix Barrel id", 10, 0., 10.);
  //hlayerid = new TH1F( "hlayerid", "Pix layer id", 10, 0., 10.);

  hladder1id = fs->make<TH1F>( "hladder1id", "Ladder L1 id", 23, -11.5, 11.5);
  hladder2id = fs->make<TH1F>( "hladder2id", "Ladder L2 id", 35, -17.5, 17.5);
  hladder3id = fs->make<TH1F>( "hladder3id", "Ladder L3 id", 47, -23.5, 23.5);
  hz1id = fs->make<TH1F>( "hz1id", "Z-index id L1", 11, -5.5, 5.5);
  hz2id = fs->make<TH1F>( "hz2id", "Z-index id L2", 11, -5.5, 5.5);
  hz3id = fs->make<TH1F>( "hz3id", "Z-index id L3", 11, -5.5, 5.5);

  int sizeH=200;
  float lowH = -0.5;
  float highH = 199.5;

  hclusPerDet1 = fs->make<TH1F>( "hclusPerDet1", "Clus per det l1",
			    sizeH, lowH, highH);
  hclusPerDet2 = fs->make<TH1F>( "hclusPerDet2", "Clus per det l2",
			    sizeH, lowH, highH);
  hclusPerDet3 = fs->make<TH1F>( "hclusPerDet3", "Clus per det l3",
			    sizeH, lowH, highH);

  sizeH=500;
  highH = 999.5;
  hpixPerDet1 = fs->make<TH1F>( "hpixPerDet1", "Pix per det l1",
			    sizeH, lowH, highH);
  hpixPerDet2 = fs->make<TH1F>( "hpixPerDet2", "Pix per det l2",
			    sizeH, lowH, highH);
  hpixPerDet3 = fs->make<TH1F>( "hpixPerDet3", "Pix per det l3",
			    sizeH, lowH, highH);

  sizeH=2000;
  highH = 1999.5;
  hclusPerLay1 = fs->make<TH1F>( "hclusPerLay1", "Clus per layer l1",
				 sizeH, lowH, highH);
  hclusPerLay2 = fs->make<TH1F>( "hclusPerLay2", "Clus per layer l2",
			    sizeH, lowH, highH);
  hclusPerLay3 = fs->make<TH1F>( "hclusPerLay3", "Clus per layer l3",
			    sizeH, lowH, highH);

  sizeH=2000;
  highH = 9999.5;
  hpixPerLay1 = fs->make<TH1F>( "hpixPerLay1", "Pix per layer l1",
				 sizeH, lowH, highH);
  hpixPerLay2 = fs->make<TH1F>( "hpixPerLay2", "Pix per layer l2",
				 sizeH, lowH, highH);
  hpixPerLay3 = fs->make<TH1F>( "hpixPerLay3", "Pix per layer l3",
				 sizeH, lowH, highH);

  hclus = fs->make<TH1F>( "hclus", "Clus per event",
			    sizeH, lowH, highH);
  hdigis = fs->make<TH1F>( "hdigis", "Digis in clus per event",
			    2000, lowH, 19999.5);
  hclusClean = fs->make<TH1F>( "hclusClean", "Clus per event",
			       1000, 0., 1000.);
  htest = fs->make<TH1F>( "htest", "Digis in clus per event",
			  1000, lowH, 999.5);

  sizeH=2000;
  highH = 1999.5;
  hclus1 = fs->make<TH1F>( "hclus1", "Clus per event",
			    sizeH, lowH, highH);
  hclus2 = fs->make<TH1F>( "hclus2", "Clus per event",
			    sizeH, lowH, highH);
  hclus3 = fs->make<TH1F>( "hclus3", "Clus per event",
			    sizeH, lowH, highH);
  hclus4 = fs->make<TH1F>( "hclus4", "Clus per event",
			    sizeH, lowH, highH);
  hclus5 = fs->make<TH1F>( "hclus5", "Clus per event",
			    sizeH, lowH, highH);
  hclus6 = fs->make<TH1F>( "hclus6", "Clus per event",
			    sizeH, lowH, highH);
  hclus7 = fs->make<TH1F>( "hclus7", "Clus per event",
        		    sizeH, lowH, highH);
  hclus8 = fs->make<TH1F>( "hclus8", "Clus per event",
			    sizeH, lowH, highH);
  hclus9 = fs->make<TH1F>( "hclus9", "Clus per event",
			    sizeH, lowH, highH);
  hclus10 = fs->make<TH1F>( "hclus10", "Clus per event",
			    sizeH, lowH, highH);
  hclus11 = fs->make<TH1F>( "hclus11", "Clus per event",
			    sizeH, lowH, highH);
  hclus12 = fs->make<TH1F>( "hclus12", "Clus per event",
			    sizeH, lowH, highH);

  hdetsPerLay1 = fs->make<TH1F>( "hdetsPerLay1", "Full dets per layer l1",
				 161, -0.5, 160.5);
  hdetsPerLay3 = fs->make<TH1F>( "hdetsPerLay3", "Full dets per layer l3",
				 353, -0.5, 352.5);
  hdetsPerLay2 = fs->make<TH1F>( "hdetsPerLay2", "Full dets per layer l2",
				 257, -0.5, 256.5);
 
  sizeH=1000;
  lowH = 0.;
  highH = 100.0; // charge limit in kelec
  hcharge1 = fs->make<TH1F>( "hcharge1", "Clu charge l1", sizeH, 0.,highH); //in ke
  hcharge2 = fs->make<TH1F>( "hcharge2", "Clu charge l2", sizeH, 0.,highH);
  hcharge3 = fs->make<TH1F>( "hcharge3", "Clu charge l3", sizeH, 0.,highH);
 
  hncharge1 = fs->make<TH1F>( "hncharge1", "Noise charge l1", sizeH, 0.,highH);//in ke
  hncharge2 = fs->make<TH1F>( "hncharge2", "Noise charge l2", sizeH, 0.,highH);
  hncharge3 = fs->make<TH1F>( "hncharge3", "Noise charge l3", sizeH, 0.,highH);
  sizeH=600;
  highH = 60.0; // charge limit in kelec
  hpixcharge1 = fs->make<TH1F>( "hpixcharge1", "Pix charge l1",sizeH, 0.,highH);//in ke
  hpixcharge2 = fs->make<TH1F>( "hpixcharge2", "Pix charge l2",sizeH, 0.,highH);
  hpixcharge3 = fs->make<TH1F>( "hpixcharge3", "Pix charge l3",sizeH, 0.,highH);
 
  hnpixcharge1 = fs->make<TH1F>( "hnpixcharge1", "Noise pix charge l1",sizeH, 0.,highH); 
  hnpixcharge2 = fs->make<TH1F>( "hnpixcharge2", "Noise pix charge l2",sizeH, 0.,highH);
  hnpixcharge3 = fs->make<TH1F>( "hnpixcharge3", "Noise pix charge l3",sizeH, 0.,highH);
 
  hcols1 = fs->make<TH1F>( "hcols1", "Layer 1 cols", 500,-0.5,499.5);
  hcols2 = fs->make<TH1F>( "hcols2", "Layer 2 cols", 500,-0.5,499.5);
  hcols3 = fs->make<TH1F>( "hcols3", "Layer 3 cols", 500,-0.5,499.5);
  
  hrows1 = fs->make<TH1F>( "hrows1", "Layer 1 rows", 200,-0.5,199.5);
  hrows2 = fs->make<TH1F>( "hrows2", "Layer 2 rows", 200,-0.5,199.5);
  hrows3 = fs->make<TH1F>( "hrows3", "layer 3 rows", 200,-0.5,199.5);


  sizeH=1000;
  highH = 999.5; // charge limit in kelec
  hsize1 = fs->make<TH1F>( "hsize1", "layer 1 clu size",sizeH,-0.5,highH);
  hsize2 = fs->make<TH1F>( "hsize2", "layer 2 clu size",sizeH,-0.5,highH);
  hsize3 = fs->make<TH1F>( "hsize3", "layer 3 clu size",sizeH,-0.5,highH);

  hsizex1 = fs->make<TH1F>( "hsizex1", "lay1 clu size in x",
		      10,-0.5,9.5);
  hsizex2 = fs->make<TH1F>( "hsizex2", "lay2 clu size in x",
		      10,-0.5,9.5);
  hsizex3 = fs->make<TH1F>( "hsizex3", "lay3 clu size in x",
		      10,-0.5,9.5);
  hsizey1 = fs->make<TH1F>( "hsizey1", "lay1 clu size in y",
		      20,-0.5,19.5);
  hsizey2 = fs->make<TH1F>( "hsizey2", "lay2 clu size in y",
		      20,-0.5,19.5);
  hsizey3 = fs->make<TH1F>( "hsizey3", "lay3 clu size in y",
		      20,-0.5,19.5);

  hevent0 = fs->make<TH1F>("hevent0","event",1000,0,10000000.);
  hevent = fs->make<TH1F>("hevent","event",1000,0,10000000.);
  horbit = fs->make<TH1F>("horbit","orbit",100, 0,100000000.);
  hlumi0  = fs->make<TH1F>("hlumi0", "lumi", 1000,0,1000.);
  hlumi  = fs->make<TH1F>("hlumi", "lumi", 1000,0,1000.);
  hbx    = fs->make<TH1F>("hbx",   "bx",   4000,0,4000.);  
  hl1a    = fs->make<TH1F>("hl1a",   "l1a",   128,-0.5,127.5);
  hl1t    = fs->make<TH1F>("hl1t",   "l1t",   128,-0.5,127.5);

  hDetMap1 = fs->make<TH2F>("hDetMap1"," ",9,-4.5,4.5,21,-10.5,10.5);
  hDetMap1->SetOption("colz");
  hDetMap2 = fs->make<TH2F>("hDetMap2"," ",9,-4.5,4.5,33,-16.5,16.5);
  hDetMap2->SetOption("colz");
  hDetMap3 = fs->make<TH2F>("hDetMap3"," ",9,-4.5,4.5,45,-22.5,22.5);
  hDetMap3->SetOption("colz");

  hpDetMap1 = fs->make<TH2F>("hpDetMap1"," ",9,-4.5,4.5,21,-10.5,10.5);
  hpDetMap1->SetOption("colz");
  hpDetMap2 = fs->make<TH2F>("hpDetMap2"," ",9,-4.5,4.5,33,-16.5,16.5);
  hpDetMap2->SetOption("colz");
  hpDetMap3 = fs->make<TH2F>("hpDetMap3"," ",9,-4.5,4.5,45,-22.5,22.5);
  hpDetMap3->SetOption("colz");
  
  hpixDetMap1 = fs->make<TH2F>( "hpixDetMap1", "pix det layer 1",
		      416,0.,416.,160,0.,160.);
  hpixDetMap2 = fs->make<TH2F>( "hpixDetMap2", "pix det layer 2",
		      416,0.,416.,160,0.,160.);
  hpixDetMap3 = fs->make<TH2F>( "hpixDetMap3", "pix det layer 3",
		      416,0.,416.,160,0.,160.);

  hcluDetMap1 = fs->make<TH2F>( "hcluDetMap1", "clu det layer 1",
				416,0.,416.,160,0.,160.);
  hcluDetMap2 = fs->make<TH2F>( "hcluDetMap2", "clu det layer 1",
				416,0.,416.,160,0.,160.);
  hcluDetMap3 = fs->make<TH2F>( "hcluDetMap3", "clu det layer 1",
				416,0.,416.,160,0.,160.);

  hpixDetMap10 = fs->make<TH2F>( "hpixDetMap10", "pix det layer 1",
		      416,0.,416.,160,0.,160.);
  hpixDetMap20 = fs->make<TH2F>( "hpixDetMap20", "pix det layer 2",
		      416,0.,416.,160,0.,160.);
  hpixDetMap30 = fs->make<TH2F>( "hpixDetMap30", "pix det layer 3",
		      416,0.,416.,160,0.,160.);


  hpixDetMapNoise = fs->make<TH2F>( "hpixDetMapNoise", "pix noise",
				    416,0.,416.,160,0.,160.);


  hpixDetMap11 = fs->make<TH2F>( "hpixDetMap11", "pix det layer 1",
		      416,0.,416.,160,0.,160.);
  hpixDetMap12 = fs->make<TH2F>( "hpixDetMap12", "pix det layer 1",
		      416,0.,416.,160,0.,160.);
  hpixDetMap13 = fs->make<TH2F>( "hpixDetMap13", "pix det layer 1",
		      416,0.,416.,160,0.,160.);
  hpixDetMap14 = fs->make<TH2F>( "hpixDetMap14", "pix det layer 1",
		      416,0.,416.,160,0.,160.);


#endif

  countEvents=0;
  countAllEvents=0;
  sumClusters=0.;
}
// ------------ method called to at the end of the job  ------------
void ReadPixClusters::endJob(){
  sumClusters = sumClusters/float(countEvents);
  cout << " End PixelClusTest, events all/with hits=  " << countAllEvents<<"/"<<countEvents<<" "
       <<sumClusters<<endl;

#ifdef HISTOS
  //hFile->Write();
  //hFile->Close();
#endif // HISTOS
}
//////////////////////////////////////////////////////////////////
// Functions that gets called by framework every event
void ReadPixClusters::analyze(const edm::Event& e, 
			      const edm::EventSetup& es) {
  using namespace edm;
  const int MAX_CUT = 1000000;
  const int selectEvent = -1; // 262;
  const int l1ldr = -4, l1mod=2,l2ldr = 8, l2mod=-4,l3ldr = -16, l3mod=1; 

  // Get event setup 
  edm::ESHandle<TrackerGeometry> geom;
  es.get<TrackerDigiGeometryRecord>().get( geom );
  const TrackerGeometry& theTracker(*geom);

  countAllEvents++;
  int run       = e.id().run();
  int event     = e.id().event();

  int lumiBlock = e.luminosityBlock();
  int bx        = e.bunchCrossing();
  int orbit     = e.orbitNumber();
  //if(lumiBlock!=46) return;


#ifdef L1
  // Get L1
  Handle<L1GlobalTriggerReadoutRecord> L1GTRR;
  e.getByLabel("gtDigis",L1GTRR);

  bool bit0=false, bit40=false, bit41=false, halo=false, splash1=false, splash2=false;
  if (L1GTRR.isValid()) {
    bool l1a = L1GTRR->decision();
    //cout<<" L1 status :"<<l1a<<" "<<hex;
    for (unsigned int i = 0; i < L1GTRR->decisionWord().size(); ++i) {
      int l1flag = L1GTRR->decisionWord()[i]; 
      int t1flag = L1GTRR->technicalTriggerWord()[i]; 
    
      if( l1flag>0 ) hl1a->Fill(float(i));

      if( t1flag>0 && i<64) {
	hl1t->Fill(float(i));
	if(i==0) bit0=true;
	else if(i==36) halo=true;
	else if(i==37) halo=true;
	else if(i==38) halo=true;
	else if(i==39) halo=true;
	else if(i==40) bit40=true;
	else if(i==41) bit41=true;
	else if(i==42) {splash1=true;}
	else if(i==43) {splash2=true;}
      }
      //cout<<l1flag<<" "<<t1flag<<" ";

    } // for loop
  } // if l1a

  bool splashAnd = (splash1 && splash2);
  bool splashOr = (splash1 || splash2);
  bool splashXOR = (splashOr && !splashAnd);
  bool mbBits = bit40 || bit41;


#endif


  // Get Cluster Collection from InputTag
  edm::Handle< edmNew::DetSetVector<SiPixelCluster> > clusters;
  e.getByLabel( src_ , clusters);

  const edmNew::DetSetVector<SiPixelCluster>& input = *clusters;     
  int numOf = input.size();
  
  //cout<<numOf<<endl;

  hevent0->Fill(float(event));

  hlumi0->Fill(float(lumiBlock));

  bool lookAtMe = false;  // event selection
  if(numOf<1) return; // skip events with no pixels
  
  hevent->Fill(float(event));

  hlumi->Fill(float(lumiBlock));
  hbx->Fill(float(bx));
  horbit->Fill(float(orbit));
  if(PRINT) cout<<"run "<<run<<" event "<<event<<" bx "<<bx<<" lumi "<<lumiBlock<<" orbit "<<orbit<<" "<<numOf<<endl;  

#ifdef L1
  if(PRINT) cout<<" l1a "<<bit0<<" "<<halo<<" "<<bit40<<" "<<bit41<<" "<<splash1<<" "<<splash2<<endl;
#endif


  lookAtMe=true; // set to always true

  countEvents++;
  int numberOfDetUnits = 0;
  int numberOfClusters = 0;
  int numberOfPixels = 0;
  int numberOfDetUnits1 = 0;
  int numOfClustersPerDet1=0;        
  int numOfClustersPerLay1=0;        
  int numberOfDetUnits2 = 0;
  int numOfClustersPerDet2=0;        
  int numOfClustersPerLay2=0;        
  int numberOfDetUnits3 = 0;
  int numOfClustersPerDet3=0;        
  int numOfClustersPerLay3=0;        

  int numOfPixPerLay1=0;     
  int numOfPixPerLay2=0;     
  int numOfPixPerLay3=0;     

  int numOfPixPerDet1=0;  
  int numOfPixPerDet2=0;  
  int numOfPixPerDet3=0;  
      

  int maxClusPerDet=0;
  int maxPixPerDet=0;

  
  static int module1[416][160] = {{0}};
  static int module2[416][160] = {{0}};
  static int module3[416][160] = {{0}};

  
  // get vector of detunit ids
  //--- Loop over detunits.
  edmNew::DetSetVector<SiPixelCluster>::const_iterator DSViter=input.begin();
  for ( ; DSViter != input.end() ; DSViter++) {
    bool valid = false;
    unsigned int detid = DSViter->detId();
    // Det id
    DetId detId = DetId(detid);       // Get the Detid object
    unsigned int detType=detId.det(); // det type, pixel=1
    unsigned int subid=detId.subdetId(); //subdetector type, barrel=1
 
    if(PRINT)
      cout<<"Det: "<<detId.rawId()<<" "<<detId.null()<<" "<<detType<<" "<<subid<<endl;    


#ifdef HISTOS
    //hdetunit->Fill(float(detid));
    //hpixid->Fill(float(detType));
    //hpixsubid->Fill(float(subid));
#endif // HISTOS

    if(detType!=1) continue; // look only at pixels
    ++numberOfDetUnits;
  
    //const GeomDetUnit * genericDet = geom->idToDet(detId);
    //const PixelGeomDetUnit * pixDet = 
    //dynamic_cast<const PixelGeomDetUnit*>(genericDet);

    // Get the geom-detector
    const PixelGeomDetUnit * theGeomDet =
      dynamic_cast<const PixelGeomDetUnit*> (theTracker.idToDet(detId) );
    double detZ = theGeomDet->surface().position().z();
    double detR = theGeomDet->surface().position().perp();

    //const BoundPlane& plane = theGeomDet->surface(); //for transf.
    
    //double detThick = theGeomDet->specificSurface().bounds().thickness();
    //int cols = theGeomDet->specificTopology().ncolumns();
    //int rows = theGeomDet->specificTopology().nrows();
    

    const RectangularPixelTopology * topol =
       dynamic_cast<const RectangularPixelTopology*>(&(theGeomDet->specificTopology()));

    // barrel ids
    unsigned int layerC=0;
    unsigned int ladderC=0;
    unsigned int zindex=0;

    // Shell { mO = 1, mI = 2 , pO =3 , pI =4 };
    int shell  = 0; // shell id 
    int sector = 0; // 1-8
    int ladder = 0; // 1-22
    int layer  = 0; // 1-3
    int module = 0; // 1-4
    bool half  = false; // 

    // Subdet id, pix barrel=1, forward=2
    if(subid==2) {  // forward

      PXFDetId pdetId = PXFDetId(detid);       
      unsigned int disk=pdetId.disk(); //1,2,3
      unsigned int blade=pdetId.blade(); //1-24
      unsigned int zindex=pdetId.module(); //
      unsigned int side=pdetId.side(); //size=1 for -z, 2 for +z
      unsigned int panel=pdetId.panel(); //panel=1
      
      if(PRINT) cout<<" forward det, disk "<<disk<<", blade "
 		    <<blade<<", module "<<zindex<<", side "<<side<<", panel "
 		    <<panel<<" pos = "<<detZ<<" "<<detR<<endl;
 
      continue; // skip fpix

    } else if (subid==1) {  // barrel

#ifdef HISTOS
      //hdetr->Fill(detR);
      //hdetz->Fill(detZ);
#endif // HISTOS
 
      //hcolsB->Fill(float(cols));
      //hrowsB->Fill(float(rows));
      
      PXBDetId pdetId = PXBDetId(detid);
      unsigned int detTypeP=pdetId.det();
      unsigned int subidP=pdetId.subdetId();
      // Barell layer = 1,2,3
      layerC=pdetId.layer();
      // Barrel ladder id 1-20,32,44.
      ladderC=pdetId.ladder();
      // Barrel Z-index=1,8
      zindex=pdetId.module();

      // Convert to online 
      PixelBarrelName pbn(pdetId);
      // Shell { mO = 1, mI = 2 , pO =3 , pI =4 };
      PixelBarrelName::Shell sh = pbn.shell(); //enum
      sector = pbn.sectorName();
      ladder = pbn.ladderName();
      layer  = pbn.layerName();
      module = pbn.moduleName();
      half  = pbn.isHalfModule();
      shell = int(sh);
      // change the module sign for z<0
      if(shell==1 || shell==2) module = -module;
      // change ladeer sign for Outer )x<0)
      if(shell==1 || shell==3) ladder = -ladder;
      
      if(PRINT) { 
	cout<<" Barrel layer, ladder, module "
	    <<layerC<<" "<<ladderC<<" "<<zindex<<" "
	    <<sh<<"("<<shell<<") "<<sector<<" "<<layer<<" "<<ladder<<" "
	    <<module<<" "<<half<< endl;
	//cout<<" Barrel det, thick "<<detThick<<" "
	//  <<" layer, ladder, module "
	//  <<layer<<" "<<ladder<<" "<<zindex<<endl;
	//cout<<" col/row, pitch "<<cols<<" "<<rows<<" "
	//  <<pitchX<<" "<<pitchY<<endl;
      }      
      
    } // if subid

    if(PRINT) {
      cout<<"List clusters : "<<endl;
      cout<<"Num Charge Size SizeX SizeY X Y Xmin Xmax Ymin Ymax Edge"
	  <<endl;
    }

    // Loop over clusters
    edmNew::DetSet<SiPixelCluster>::const_iterator clustIt;
    for (clustIt = DSViter->begin(); clustIt != DSViter->end(); clustIt++) {
      sumClusters++;
      numberOfClusters++;
      float ch = float(clustIt->charge())/1000.; // convert ke to electrons
      int size = clustIt->size();
      int sizeX = clustIt->sizeX(); //x=row=rfi, 
      int sizeY = clustIt->sizeY(); //y=col=z_global
      float x = clustIt->x(); // cluster position as float (int+0.5)
      float y = clustIt->y(); // analog average
      // Returns int index of the cluster min/max  
      int minPixelRow = clustIt->minPixelRow(); //x
      int maxPixelRow = clustIt->maxPixelRow();
      int minPixelCol = clustIt->minPixelCol(); //y
      int maxPixelCol = clustIt->maxPixelCol();
      
      //unsigned int geoId = clustIt->geographicalId(); // always 0?!
      // edge method moved to topologu class
      bool edgeHitX = (topol->isItEdgePixelInX(minPixelRow)) || 
	(topol->isItEdgePixelInX(maxPixelRow)); 
      bool edgeHitY = (topol->isItEdgePixelInY(minPixelCol)) || 
	(topol->isItEdgePixelInY(maxPixelCol)); 

      bool edgeHitX2 = false; // edge method moved 
      bool edgeHitY2 = false; // to topologu class
            
      if(PRINT) cout<<numberOfClusters<<" "<<ch<<" "<<size<<" "<<sizeX<<" "<<sizeY<<" "
		    <<x<<" "<<y<<" "<<minPixelRow<<" "<<maxPixelRow<<" "<<minPixelCol<<" "
		    <<maxPixelCol<<" "<<edgeHitX<<" "<<edgeHitY<<endl;

//       if(layer==2 && ladder==16 && module==5 ) 
// 	cout<<numberOfClusters<<" "<<ch<<" "<<size<<" "<<sizeX<<" "<<sizeY<<" "
// 	    <<x<<" "<<y<<" "<<minPixelRow<<" "<<maxPixelRow<<" "<<minPixelCol<<" "
// 	    <<maxPixelCol<<" "<<edgeHitX<<" "<<edgeHitY<<endl;



      // Get the pixels in the Cluster
      const vector<SiPixelCluster::Pixel>& pixelsVec = clustIt->pixels();
      if(PRINT) cout<<" Pixels in this cluster "<<endl;
      bool bigInX=false, bigInY=false;
      // Look at pixels in this cluster. ADC is calibrated, in electrons
      bool edgeInX = false; // edge method moved 
      bool edgeInY = false; // to topologu class
      bool cluBigInX = false; // does this clu include a big pixel
      bool cluBigInY = false; // does this clu include a big pixel
      int noisy = 0;

      if(pixelsVec.size()>maxPixPerDet) maxPixPerDet = pixelsVec.size();
 
      for (unsigned int i = 0;  i < pixelsVec.size(); ++i) { // loop over pixels
	numberOfPixels++;
	float pixx = pixelsVec[i].x; // index as float=iteger
	float pixy = pixelsVec[i].y; // same
	float adc = (float(pixelsVec[i].adc)/1000.);
	//int chan = PixelChannelIdentifier::pixelToChannel(int(pixx),int(pixy));
	//bool binInX = (RectangularPixelTopology::isItBigPixelInX(int(pixx)));
	//bool bigInY = (RectangularPixelTopology::isItBigPixelInY(int(pixy)));
	
#ifdef HISTOS
	// Pixel histos
	if (subid==1 && (selectEvent==-1 || countEvents==selectEvent)) {  // barrel
	  if(layer==1) {
	    numOfPixPerDet1++;
	    numOfPixPerLay1++;     
	    valid = valid || true;
	    hpixcharge1->Fill(adc);
	    hpixDetMap1->Fill(pixy,pixx);
	    if(ladder==l1ldr&&module==l1mod) hpixDetMap10->Fill(pixy,pixx,adc);
	    hpDetMap1->Fill(float(module),float(ladder));
	    module1[int(pixx)][int(pixy)]++;
	    if(module1[int(pixx)][int(pixy)]>MAX_CUT) 
	      cout<<" module "<<layer<<" "<<ladder<<" "<<module<<" "
		  <<pixx<<" "<<pixy<<" "<<module1[int(pixx)][int(pixy)]<<endl;

	  } else if(layer==2) {

	    numOfPixPerDet2++;
	    numOfPixPerLay2++;   
	    bool noise = false; // (ladder==6) && (module==-2) && (pixy==364) && (pixx==1);
	    if(noise) {
	      cout<<" noise pixel "<<layer<<" "<<sector<<" "<<shell<<endl;
	      hpixDetMapNoise->Fill(pixy,pixx);
 	      hnpixcharge2->Fill(adc);
	      noisy++;
	    } else {                    
	      valid = valid || true;
	      hpixcharge2->Fill(adc);
	      hpixDetMap2->Fill(pixy,pixx);
	      if(ladder==l2ldr&&module==l2mod) hpixDetMap20->Fill(pixy,pixx,adc);
	      else if(ladder==7 && module==-4) hpixDetMap11->Fill(pixy,pixx,adc);
	      else if(ladder==7 && module==-3) hpixDetMap12->Fill(pixy,pixx,adc);
	      else if(ladder==7 && module==-2) hpixDetMap13->Fill(pixy,pixx,adc);
	      else if(ladder==7 && module==-1) hpixDetMap14->Fill(pixy,pixx,adc);
	      hpDetMap2->Fill(float(module),float(ladder));
	      module2[int(pixx)][int(pixy)]++;
	      if(module2[int(pixx)][int(pixy)]>MAX_CUT) 
		cout<<" module "<<layer<<" "<<ladder<<" "<<module<<" "
		    <<pixx<<" "<<pixy<<" "<<module2[int(pixx)][int(pixy)]<<endl;
	    } // noise

	  } else if(layer==3) {

	    numOfPixPerDet3++;
	    numOfPixPerLay3++; 
	    valid = valid || true;
	    hpixcharge3->Fill(adc);
	    hpixDetMap3->Fill(pixy,pixx);
	    if(ladder==l3ldr&&module==l3mod) hpixDetMap30->Fill(pixy,pixx,adc);
	    hpDetMap3->Fill(float(module),float(ladder));
	    module3[int(pixx)][int(pixy)]++;
	    if(module3[int(pixx)][int(pixy)]>MAX_CUT) 
	      cout<<" module "<<layer<<" "<<ladder<<" "<<module<<" "
		  <<pixx<<" "<<pixy<<" "<<module3[int(pixx)][int(pixy)]<<endl;
	  }
	} // if barrel
#endif // HISTOS
	
	edgeInX = topol->isItEdgePixelInX(int(pixx));
	edgeInY = topol->isItEdgePixelInY(int(pixy));
	
	if(PRINT) cout<<i<<" "<<pixx<<" "<<pixy<<" "<<adc<<" "<<bigInX<<" "<<bigInY
		      <<" "<<edgeInX<<" "<<edgeInY<<endl;
	
	if(edgeInX) edgeHitX2=true;
	if(edgeInY) edgeHitY2=true; 
	if(bigInX) cluBigInX=true;
	if(bigInY) cluBigInY=true;

      } // pixel loop
      


#ifdef HISTOS
      
      // Cluster histos
      if (subid==1 && (selectEvent==-1 || countEvents==selectEvent) ) {  // barrel
	//if (subid==1) {  // barrel
	if(layer==1) {  // layer
	  
	  hDetMap1->Fill(float(module),float(ladder));
	  hcluDetMap1->Fill(y,x);
	  hcharge1->Fill(ch);
	  hcols1->Fill(y);
	  hrows1->Fill(x);
	  hsize1->Fill(float(size));
	  hsizex1->Fill(float(sizeX));
	  hsizey1->Fill(float(sizeY));
	  numOfClustersPerDet1++;
	  numOfClustersPerLay1++;
	  //htest2->Fill(float(zindex),float(adc));

	} else if(layer==2) {

	  // Skip noise 
	  if(noisy>0) {
 	    hncharge2->Fill(ch);
 	    continue; // skip plotting noise cluster
 	  }

	  hDetMap2->Fill(float(module),float(ladder));
	  hcluDetMap2->Fill(y,x);
	  hcharge2->Fill(ch);
	  hcols2->Fill(y);
	  hrows2->Fill(x);
	  hsize2->Fill(float(size));
	  hsizex2->Fill(float(sizeX));
	  hsizey2->Fill(float(sizeY));
	  numOfClustersPerDet2++;
	  numOfClustersPerLay2++;

	} else if(layer==3) {

	  hDetMap3->Fill(float(module),float(ladder));
	  hcluDetMap3->Fill(y,x);
	  hcharge3->Fill(ch);
	  hcols3->Fill(y);
	  hrows3->Fill(x);
	  hsize3->Fill(float(size));
	  hsizex3->Fill(float(sizeX));
	  hsizey3->Fill(float(sizeY));
	  numOfClustersPerDet3++;
	  numOfClustersPerLay3++;

	} // end if layer
      } // end barrel/forward
#endif // HISTOS


      if(edgeHitX != edgeHitX2) 
	cout<<" wrong egdeX "<<edgeHitX<<" "<<edgeHitX2<<endl;
      if(edgeHitY != edgeHitY2) 
	cout<<" wrong egdeY "<<edgeHitY<<" "<<edgeHitY2<<endl;

    } // clusters 

    
    if(numOfClustersPerDet1>maxClusPerDet) maxClusPerDet = numOfClustersPerDet1;
    if(numOfClustersPerDet2>maxClusPerDet) maxClusPerDet = numOfClustersPerDet2;
    if(numOfClustersPerDet3>maxClusPerDet) maxClusPerDet = numOfClustersPerDet3;

    if(PRINT) {
      if(layer==1) 
	cout<<"Lay1: number of clusters per det = "<<numOfClustersPerDet1<<endl;
      else if(layer==2) 
	cout<<"Lay2: number of clusters per det = "<<numOfClustersPerDet1<<endl;
      else if(layer==3) 
	cout<<"Lay3: number of clusters per det = "<<numOfClustersPerDet1<<endl;
    } // end if PRINT

#ifdef HISTOS
    if (subid==1 && (selectEvent==-1 || countEvents==selectEvent) ) {  // barrel
      //if (subid==1 && valid && countEvents==selectEvent) {  // barrel

      //hlayerid->Fill(float(layer));

      // Det histos
      if(layer==1) {
	
	hladder1id->Fill(float(ladder));
	hz1id->Fill(float(module));
	++numberOfDetUnits1;
	hclusPerDet1->Fill(float(numOfClustersPerDet1));
	hpixPerDet1->Fill(float(numOfPixPerDet1));
	numOfClustersPerDet1=0;        
	numOfPixPerDet1=0;        

      } else if(layer==2) {

	hladder2id->Fill(float(ladder));
	hz2id->Fill(float(module));
	++numberOfDetUnits2;
	hclusPerDet2->Fill(float(numOfClustersPerDet2));
	hpixPerDet2->Fill(float(numOfPixPerDet2));
	numOfClustersPerDet2=0;
	numOfPixPerDet2=0;        

      } else if(layer==3) {

	hladder3id->Fill(float(ladder));
	hz3id->Fill(float(module));
	++numberOfDetUnits3;
	hclusPerDet3->Fill(float(numOfClustersPerDet3));
	hpixPerDet3->Fill(float(numOfPixPerDet3));
	numOfClustersPerDet3=0;
	numOfPixPerDet3=0;        

      } // layer
      
    } // end barrel/forward

#endif // HISTOS
    
  } // detunits loop
    

  if(lookAtMe) {
    if(PRINT) { //  || numberOfPixels==0) {
      cout<<"run "<<run<<" event "<<event<<" bx "<<bx<<" lumi "<<lumiBlock<<" orbit "<<orbit<<endl;   
      cout<<"Num of pix "<<numberOfPixels<<" num of clus "<<numberOfClusters<<" max clus per det "
	  <<maxClusPerDet<<" max pix per det "<<maxPixPerDet<<" "
	  <<countEvents<<endl;
      cout<<"Number of clusters per Lay1,2,3: "<<numOfClustersPerLay1<<" "
	  <<numOfClustersPerLay2<<" "<<numOfClustersPerLay3<<endl;
      cout<<"Number of pixels per Lay1,2,3: "<<numOfPixPerLay1<<" "
	  <<numOfPixPerLay2<<" "<<numOfPixPerLay3<<endl;
      cout<<"Number of dets with clus in Lay1,2,3: "<<numberOfDetUnits1<<" "
	  <<numberOfDetUnits2<<" "<<numberOfDetUnits3<<endl;
      //     if(bx!=101) {
      //     int dummy;
      //     cout<<" enter:";
      //     cin>>dummy;
      //     }
    }
    
#ifdef HISTOS
    //if (countEvents==selectEvent) {  //
    if ( (selectEvent==-1 || countEvents==selectEvent) ) {  // barrel

      hdigis->Fill(float(numberOfPixels));
      htest->Fill(float(numberOfPixels));
      hclus->Fill(float(numberOfClusters));
      int tmp = numOfClustersPerLay1+numOfClustersPerLay2+numOfClustersPerLay3;
      hclusClean->Fill(float(tmp));
      hclusPerLay1->Fill(float(numOfClustersPerLay1));
      hclusPerLay2->Fill(float(numOfClustersPerLay2));
      hclusPerLay3->Fill(float(numOfClustersPerLay3));
      hpixPerLay1->Fill(float(numOfPixPerLay1));
      hpixPerLay2->Fill(float(numOfPixPerLay2));
      hpixPerLay3->Fill(float(numOfPixPerLay3));
      if(numOfClustersPerLay1>0) hdetsPerLay1->Fill(float(numberOfDetUnits1));
      if(numOfClustersPerLay2>0) hdetsPerLay2->Fill(float(numberOfDetUnits2));
      if(numOfClustersPerLay3>0) hdetsPerLay3->Fill(float(numberOfDetUnits3));

      // the bx values are different
      if(bx==51) hclus1->Fill(float(numberOfClusters));
      else if(bx==1830) hclus2->Fill(float(numberOfClusters));
      else if(bx==1833) hclus3->Fill(float(numberOfClusters));
      else if(bx==2276) hclus4->Fill(float(numberOfClusters));
      else if(bx==2724) hclus5->Fill(float(numberOfClusters));
      else if(bx==3170) hclus6->Fill(float(numberOfClusters));

#ifdef L1
      if(splashXOR) hclus7->Fill(float(numberOfClusters));
      if(mbBits) hclus8->Fill(float(numberOfClusters));
      if(halo) hclus9->Fill(float(numberOfClusters));
      if(splashXOR && !mbBits) hclus10->Fill(float(numberOfClusters));
      if(mbBits && !splashXOR && !halo) hclus11->Fill(float(numberOfClusters));
#endif


    } // if select
#endif // HISTOS
  } // if lookAtMe

} // end 

//define this as a plug-in
DEFINE_FWK_MODULE(ReadPixClusters);
