// File: ReadPixClusters.cc
// Description: TO test the pixel clusters. 
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

using namespace std;

class ReadPixClusters : public edm::EDAnalyzer {
 public:
  
  explicit ReadPixClusters(const edm::ParameterSet& conf);  
  virtual ~ReadPixClusters();
  virtual void analyze(const edm::Event& e, const edm::EventSetup& c);
  virtual void beginJob();
  virtual void endJob();
  
 private:
  edm::ParameterSet conf_;
  edm::InputTag src_;
  const static bool PRINT = false;

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
  TH1F *hclusPerLay1,*hclusPerLay2,*hclusPerLay3;
  TH1F *hdetsPerLay1,*hdetsPerLay2,*hdetsPerLay3;
  TH1F *hclus, *hclusClean, *hdigis;

  TH1F *hdetr, *hdetz;
//   TH1F *hcolsB,  *hrowsB,  *hcolsF,  *hrowsF;
//   TH2F *htest, *htest2;
   TH2F *hDetMap1, *hDetMap2, *hDetMap3;
   TH2F *hpixDetMap1, *hpixDetMap2, *hpixDetMap3, *hpixDetMapNoise;
   TH2F *hcluDetMap1, *hcluDetMap2, *hcluDetMap3;

  TH1F *hncharge1,*hncharge2, *hncharge3;
  TH1F *hnpixcharge1,*hnpixcharge2,*hnpixcharge3;

};
/////////////////////////////////////////////////////////////////
// Contructor, empty.
ReadPixClusters::ReadPixClusters(edm::ParameterSet const& conf) 
  : conf_(conf), src_(conf.getParameter<edm::InputTag>( "src" )) { }
// Virtual destructor needed.
ReadPixClusters::~ReadPixClusters() { }  

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
  float hightH = 199.5;

  hclusPerDet1 = fs->make<TH1F>( "hclusPerDet1", "Clus per det l1",
			    200, -0.5, 199.5);
  hclusPerDet2 = fs->make<TH1F>( "hclusPerDet2", "Clus per det l2",
			    200, -0.5, 199.5);
  hclusPerDet3 = fs->make<TH1F>( "hclusPerDet3", "Clus per det l3",
			    200, -0.5, 199.5);

  hclusPerLay1 = fs->make<TH1F>( "hclusPerLay1", "Clus per layer l1",
			    2000, -0.5, 1999.5);
  hclusPerLay2 = fs->make<TH1F>( "hclusPerLay2", "Clus per layer l2",
			    2000, -0.5, 1999.5);
  hclusPerLay3 = fs->make<TH1F>( "hclusPerLay3", "Clus per layer l3",
			    2000, -0.5, 1999.5);

  hclus = fs->make<TH1F>( "hclus", "Clus per event",
			    2000, -0.5, 1999.5);
  hdigis = fs->make<TH1F>( "hdigis", "Digis in clus per event",
			   2000, -0.5, 1999.5);
  hclusClean = fs->make<TH1F>( "hclusClean", "Digis in clus per event",
			   2000, -0.5, 1999.5);

  //hdetsPerLay1 = fs->make<TH1F>( "hdetsPerLay1", "Full dets per layer l1",
  //		   161, -0.5, 160.5);
  //hdetsPerLay3 = fs->make<TH1F>( "hdetsPerLay3", "Full dets per layer l3",
  //		   353, -0.5, 352.5);
  //hdetsPerLay2 = fs->make<TH1F>( "hdetsPerLay2", "Full dets per layer l2",
  //		   257, -0.5, 256.5);
 
  sizeH=1000;
  lowH = 0.;
  hightH = 100.0; // charge limit in kelec
  hcharge1 = fs->make<TH1F>( "hcharge1", "Clu charge l1", sizeH, 0.,hightH); //in ke
  hcharge2 = fs->make<TH1F>( "hcharge2", "Clu charge l2", sizeH, 0.,hightH);
  hcharge3 = fs->make<TH1F>( "hcharge3", "Clu charge l3", sizeH, 0.,hightH);
 
  hncharge1 = fs->make<TH1F>( "hncharge1", "Noise charge l1", sizeH, 0.,hightH);//in ke
  hncharge2 = fs->make<TH1F>( "hncharge2", "Noise charge l2", sizeH, 0.,hightH);
  hncharge3 = fs->make<TH1F>( "hncharge3", "Noise charge l3", sizeH, 0.,hightH);
  sizeH=600;
  hightH = 60.0; // charge limit in kelec
  hpixcharge1 = fs->make<TH1F>( "hpixcharge1", "Pix charge l1",sizeH, 0.,hightH);//in ke
  hpixcharge2 = fs->make<TH1F>( "hpixcharge2", "Pix charge l2",sizeH, 0.,hightH);
  hpixcharge3 = fs->make<TH1F>( "hpixcharge3", "Pix charge l3",sizeH, 0.,hightH);
 
  hnpixcharge1 = fs->make<TH1F>( "hnpixcharge1", "Noise pix charge l1",sizeH, 0.,hightH); 
  hnpixcharge2 = fs->make<TH1F>( "hnpixcharge2", "Noise pix charge l2",sizeH, 0.,hightH);
  hnpixcharge3 = fs->make<TH1F>( "hnpixcharge3", "Noise pix charge l3",sizeH, 0.,hightH);
 
  hcols1 = fs->make<TH1F>( "hcols1", "Layer 1 cols", 500,-0.5,499.5);
  hcols2 = fs->make<TH1F>( "hcols2", "Layer 2 cols", 500,-0.5,499.5);
  hcols3 = fs->make<TH1F>( "hcols3", "Layer 3 cols", 500,-0.5,499.5);
  
  hrows1 = fs->make<TH1F>( "hrows1", "Layer 1 rows", 200,-0.5,199.5);
  hrows2 = fs->make<TH1F>( "hrows2", "Layer 2 rows", 200,-0.5,199.5);
  hrows3 = fs->make<TH1F>( "hrows3", "layer 3 rows", 200,-0.5,199.5);

  hsize1 = fs->make<TH1F>( "hsize1", "layer 1 clu size",100,-0.5,99.5);
  hsize2 = fs->make<TH1F>( "hsize2", "layer 2 clu size",100,-0.5,99.5);
  hsize3 = fs->make<TH1F>( "hsize3", "layer 3 clu size",100,-0.5,99.5);
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
  
  hDetMap1 = fs->make<TH2F>("hDetMap1"," ",9,-4.5,4.5,21,-10.5,10.5);
  hDetMap1->SetOption("colz");
  hDetMap2 = fs->make<TH2F>("hDetMap2"," ",9,-4.5,4.5,33,-16.5,16.5);
  hDetMap2->SetOption("colz");
  hDetMap3 = fs->make<TH2F>("hDetMap3"," ",9,-4.5,4.5,45,-22.5,22.5);
  hDetMap3->SetOption("colz");
  
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

  hpixDetMapNoise = fs->make<TH2F>( "hpixDetMapNoise", "pix noise",
				    416,0.,416.,160,0.,160.);

#endif

}
// ------------ method called to at the end of the job  ------------
void ReadPixClusters::endJob(){
  cout << " End PixelClusTest " << endl;
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

  // Get event setup 
  edm::ESHandle<TrackerGeometry> geom;
  es.get<TrackerDigiGeometryRecord>().get( geom );
  const TrackerGeometry& theTracker(*geom);

  // Get Cluster Collection from InputTag
  edm::Handle< edmNew::DetSetVector<SiPixelCluster> > clusters;
  e.getByLabel( src_ , clusters);

  const edmNew::DetSetVector<SiPixelCluster>& input = *clusters;     

  if(PRINT) cout<<"Clusters : "<<endl;
  
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

  // Gavril : Add another set of braces to make to compiler happy. Otherwise it complains. 
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
      
      // Gavril: These variables are not used. Comment out to avoid warnings.  
      // unsigned int detTypeP=pdetId.det();
      // unsigned int subidP=pdetId.subdetId();
     

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
	if (subid==1) {  // barrel
	  if(layer==1) {

	    valid = valid || true;
	    hpixcharge1->Fill(adc);
	    hpixDetMap1->Fill(pixy,pixx);
	    module1[int(pixx)][int(pixy)]++;
	    if(module1[int(pixx)][int(pixy)]>MAX_CUT) 
	      cout<<" module "<<layer<<" "<<ladder<<" "<<module<<" "
		  <<pixx<<" "<<pixy<<" "<<module1[int(pixx)][int(pixy)]<<endl;

	  } else if(layer==2) {

	    bool noise = (ladder==6) && (module==-2) && (pixy==364) && (pixx==1);
	    if(noise) {
	      cout<<" noise pixel "<<layer<<" "<<sector<<" "<<shell<<endl;
	      hpixDetMapNoise->Fill(pixy,pixx);
 	      hnpixcharge2->Fill(adc);
	      noisy++;
	    } else {                    
	      valid = valid || true;
	      hpixcharge2->Fill(adc);
	      hpixDetMap2->Fill(pixy,pixx);
	      module2[int(pixx)][int(pixy)]++;
	      if(module2[int(pixx)][int(pixy)]>MAX_CUT) 
		cout<<" module "<<layer<<" "<<ladder<<" "<<module<<" "
		    <<pixx<<" "<<pixy<<" "<<module2[int(pixx)][int(pixy)]<<endl;
	    } // noise

	  } else if(layer==3) {

	    valid = valid || true;
	    hpixcharge3->Fill(adc);
	    hpixDetMap3->Fill(pixy,pixx);
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
      if (subid==1) {  // barrel
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

    
#ifdef HISTOS
    if (subid==1 && valid) {  // barrel

      //hlayerid->Fill(float(layer));

      // Det histos
      if(layer==1) {
	
	hladder1id->Fill(float(ladder));
	hz1id->Fill(float(module));
	++numberOfDetUnits1;
	numOfClustersPerDet1=0;        
	hclusPerDet1->Fill(float(numOfClustersPerDet1));

      } else if(layer==2) {

	hladder2id->Fill(float(ladder));
	hz2id->Fill(float(module));
	++numberOfDetUnits2;
	numOfClustersPerDet2=0;
	hclusPerDet2->Fill(float(numOfClustersPerDet2));

      } else if(layer==3) {

	hladder3id->Fill(float(ladder));
	hz3id->Fill(float(module));
	++numberOfDetUnits3;
	numOfClustersPerDet3=0;
	hclusPerDet3->Fill(float(numOfClustersPerDet3));
      } // layer
      
    } // end barrel/forward

#endif // HISTOS

    if(PRINT) {
      if(layer==1) 
	cout<<"Lay1: number of clusters per det = "<<numOfClustersPerDet1<<endl;
      else if(layer==2) 
	cout<<"Lay2: number of clusters per det = "<<numOfClustersPerDet1<<endl;
      else if(layer==3) 
	cout<<"Lay3: number of clusters per det = "<<numOfClustersPerDet1<<endl;
    } // end if PRINT
    
  } // detunits loop
    

  if(PRINT) {
    cout<<"Number of clusters per Lay1,2,3: "<<numOfClustersPerLay1<<" "
	<<numOfClustersPerLay2<<" "<<numOfClustersPerLay3<<endl;
    cout<<"Number of dets with clus in Lay1,2,3: "<<numberOfDetUnits1<<" "
	<<numberOfDetUnits2<<" "<<numberOfDetUnits3<<endl;
  }
  
#ifdef HISTOS
  hdigis->Fill(float(numberOfPixels));
  hclus->Fill(float(numberOfClusters));
  int tmp = numOfClustersPerLay1+numOfClustersPerLay2+numOfClustersPerLay3;
  hclusClean->Fill(float(tmp));
  hclusPerLay1->Fill(float(numOfClustersPerLay1));
  hclusPerLay2->Fill(float(numOfClustersPerLay2));
  hclusPerLay3->Fill(float(numOfClustersPerLay3));
  //hdetsPerLay1->Fill(float(numberOfDetUnits1));
  //hdetsPerLay2->Fill(float(numberOfDetUnits2));
  //hdetsPerLay3->Fill(float(numberOfDetUnits3));
#endif // HISTOS

} // end 

//define this as a plug-in
DEFINE_FWK_MODULE(ReadPixClusters);
