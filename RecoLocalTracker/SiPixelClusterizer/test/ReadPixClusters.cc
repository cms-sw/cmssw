// File: ReadPixClusters.cc
// Description: TO test the pixel clusters. 
// Author: Danek Kotlinski 
// Creation Date:  Initial version. 3/06
//
//--------------------------------------------
#include <memory>
#include <string>
#include <iostream>

using namespace std;

#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
//#include "FWCore/Framework/interface/Handle.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/ParameterSet/interface/InputTag.h"

#include "DataFormats/Common/interface/EDProduct.h"

//#include "DataFormats/SiPixelCluster/interface/SiPixelClusterCollection.h"
#include "DataFormats/SiPixelCluster/interface/SiPixelCluster.h"
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
#include "Geometry/TrackerTopology/interface/RectangularPixelTopology.h"

// For ROOT
#include <TROOT.h>
#include <TChain.h>
#include <TFile.h>
#include <TF1.h>
#include <TH2F.h>
#include <TH1F.h>

class ReadPixClusters : public edm::EDAnalyzer {
 public:
  
  explicit ReadPixClusters(const edm::ParameterSet& conf);  
  virtual ~ReadPixClusters();
  virtual void analyze(const edm::Event& e, const edm::EventSetup& c);
  virtual void beginJob(const edm::EventSetup& iSetup);
  virtual void endJob();
  
 private:
  edm::ParameterSet conf_;
  edm::InputTag src_;
  const static bool PRINT = true;
  
  TFile* hFile;
  TH1F *hdetunit;
  TH1F *hpixid,*hpixsubid,
    *hlayerid,
    *hladder1id,*hladder2id,*hladder3id,
    *hz1id,*hz2id,*hz3id;

  TH1F *hcharge1,*hcharge2, *hcharge3;
  TH1F *hcols1,*hcols2,*hcols3,*hrows1,*hrows2,*hrows3;
  TH1F *hsize1,*hsize2,*hsize3,
    *hsizex1,*hsizex2,*hsizex3,
    *hsizey1,*hsizey2,*hsizey3;

  TH1F *hclusPerDet1,*hclusPerDet2,*hclusPerDet3;
  TH1F *hclusPerLay1,*hclusPerLay2,*hclusPerLay3;
  TH1F *hdetsPerLay1,*hdetsPerLay2,*hdetsPerLay3;

  TH1F *hdetr, *hdetz;
//   TH1F *hcolsB,  *hrowsB,  *hcolsF,  *hrowsF;
//   TH2F *htest, *htest2;
 
};
/////////////////////////////////////////////////////////////////
// Contructor, empty.
ReadPixClusters::ReadPixClusters(edm::ParameterSet const& conf) 
  : conf_(conf), src_(conf.getParameter<edm::InputTag>( "src" )) { }
// Virtual destructor needed.
ReadPixClusters::~ReadPixClusters() { }  

// ------------ method called at the begining   ------------
void ReadPixClusters::beginJob(const edm::EventSetup& iSetup) {
  cout << "Initialize PixelClusterTest " <<endl;
 
  // put here whatever you want to do at the beginning of the job
  hFile = new TFile ( "histo.root", "RECREATE" );

  hdetunit = new TH1F( "hdetunit", "Det unit", 1000,
                              302000000.,302300000.);
  hpixid = new TH1F( "hpixid", "Pix det id", 10, 0., 10.);
  hpixsubid = new TH1F( "hpixsubid", "Pix Barrel id", 10, 0., 10.);
  hlayerid = new TH1F( "hlayerid", "Pix layer id", 10, 0., 10.);
  hladder1id = new TH1F( "hladder1id", "Ladder L1 id", 50, 0., 50.);
  hladder2id = new TH1F( "hladder2id", "Ladder L2 id", 50, 0., 50.);
  hladder3id = new TH1F( "hladder3id", "Ladder L3 id", 50, 0., 50.);
  hz1id = new TH1F( "hz1id", "Z-index id L1", 10, 0., 10.);
  hz2id = new TH1F( "hz2id", "Z-index id L2", 10, 0., 10.);
  hz3id = new TH1F( "hz3id", "Z-index id L3", 10, 0., 10.);
  
  hclusPerDet1 = new TH1F( "hclusPerDet1", "Clus per det l1",
			    200, -0.5, 199.5);
  hclusPerDet2 = new TH1F( "hclusPerDet2", "Clus per det l2",
			    200, -0.5, 199.5);
  hclusPerDet3 = new TH1F( "hclusPerDet3", "Clus per det l3",
			    200, -0.5, 199.5);
  hclusPerLay1 = new TH1F( "hclusPerLay1", "Clus per layer l1",
			    2000, -0.5, 1999.5);
  hclusPerLay2 = new TH1F( "hclusPerLay2", "Clus per layer l2",
			    2000, -0.5, 1999.5);
			    
			    
  hclusPerLay3 = new TH1F( "hclusPerLay3", "Clus per layer l3",
			    2000, -0.5, 1999.5);
  hdetsPerLay1 = new TH1F( "hdetsPerLay1", "Full dets per layer l1",
			   161, -0.5, 160.5);
  hdetsPerLay3 = new TH1F( "hdetsPerLay3", "Full dets per layer l3",
			   353, -0.5, 352.5);
  hdetsPerLay2 = new TH1F( "hdetsPerLay2", "Full dets per layer l2",
			   257, -0.5, 256.5);
 
  hcharge1 = new TH1F( "hcharge1", "Clu charge l1", 200, 0.,200.); //in ke
  hcharge2 = new TH1F( "hcharge2", "Clu charge l2", 200, 0.,200.);
  hcharge3 = new TH1F( "hcharge3", "Clu charge l3", 200, 0.,200.);
 
  hcols1 = new TH1F( "hcols1", "Layer 1 cols", 500,-0.5,499.5);
  hcols2 = new TH1F( "hcols2", "Layer 2 cols", 500,-0.5,499.5);
  hcols3 = new TH1F( "hcols3", "Layer 3 cols", 500,-0.5,499.5);
  
  hrows1 = new TH1F( "hrows1", "Layer 1 rows", 200,-0.5,199.5);
  hrows2 = new TH1F( "hrows2", "Layer 2 rows", 200,-0.5,199.5);
  hrows3 = new TH1F( "hrows3", "layer 3 rows", 200,-0.5,199.5);

  hsize1 = new TH1F( "hsize1", "layer 1 clu size",100,-0.5,99.5);
  hsize2 = new TH1F( "hsize2", "layer 2 clu size",100,-0.5,99.5);
  hsize3 = new TH1F( "hsize3", "layer 3 clu size",100,-0.5,99.5);
  hsizex1 = new TH1F( "hsizex1", "lay1 clu size in x",
		      10,-0.5,9.5);
  hsizex2 = new TH1F( "hsizex2", "lay2 clu size in x",
		      10,-0.5,9.5);
  hsizex3 = new TH1F( "hsizex3", "lay3 clu size in x",
		      10,-0.5,9.5);
  hsizey1 = new TH1F( "hsizey1", "lay1 clu size in y",
		      20,-0.5,19.5);
  hsizey2 = new TH1F( "hsizey2", "lay2 clu size in y",
		      20,-0.5,19.5);
  hsizey3 = new TH1F( "hsizey3", "lay3 clu size in y",
		      20,-0.5,19.5);
  
    hdetr = new TH1F("hdetr","det r",150,0.,15.);
    hdetz = new TH1F("hdetz","det z",520,-26.,26.);

//     hcolsB = new TH1F("hcolsB","cols per bar det",450,0.,450.);
//     hrowsB = new TH1F("hrowsB","rows per bar det",200,0.,200.);
 
//     htest = new TH2F("htest"," ",10,0.,10.,20,0.,20.);
//     htest2 = new TH2F("htest2"," ",10,0.,10.,300,0.,300.);


}
// ------------ method called to at the end of the job  ------------
void ReadPixClusters::endJob(){
  cout << " End PixelClusTest " << endl;
  hFile->Write();
  hFile->Close();
}
//////////////////////////////////////////////////////////////////
// Functions that gets called by framework every event
void ReadPixClusters::analyze(const edm::Event& e, 
			      const edm::EventSetup& es) {
  using namespace edm;

  // Get event setup 
  edm::ESHandle<TrackerGeometry> geom;
  es.get<TrackerDigiGeometryRecord>().get( geom );
  const TrackerGeometry& theTracker(*geom);

  // Get Cluster Collection from InputTag
  edm::Handle< edm::DetSetVector<SiPixelCluster> > clusters;
  e.getByLabel( src_ , clusters);

  const edm::DetSetVector<SiPixelCluster>& input = *clusters;     

  if(PRINT) cout<<"Clusters : "<<endl;
  
  int numberOfDetUnits = 0;
  int numberOfClusters = 0;
  int numberOfDetUnits1 = 0;
  int numOfClustersPerDet1=0;        
  int numOfClustersPerLay1=0;        
  int numberOfDetUnits2 = 0;
  int numOfClustersPerDet2=0;        
  int numOfClustersPerLay2=0;        
  int numberOfDetUnits3 = 0;
  int numOfClustersPerDet3=0;        
  int numOfClustersPerLay3=0;        
  
  // get vector of detunit ids
  //--- Loop over detunits.
  edm::DetSetVector<SiPixelCluster>::const_iterator DSViter;  
  for (DSViter=input.begin(); DSViter != input.end() ; DSViter++) {

    unsigned int detid = DSViter->id;
    // Det id
    DetId detId = DetId(detid);       // Get the Detid object
    unsigned int detType=detId.det(); // det type, pixel=1
    unsigned int subid=detId.subdetId(); //subdetector type, barrel=1
 
    if(PRINT)
      cout<<"Det: "<<detId.rawId()<<" "<<detId.null()<<" "<<detType<<" "<<subid<<endl;    
    hdetunit->Fill(float(detid));
    hpixid->Fill(float(detType));
    hpixsubid->Fill(float(subid));

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

    const BoundPlane& plane = theGeomDet->surface(); //for transf.
    
    double detThick = theGeomDet->specificSurface().bounds().thickness();
    int cols = theGeomDet->specificTopology().ncolumns();
    int rows = theGeomDet->specificTopology().nrows();
    

    const RectangularPixelTopology * topol =
       dynamic_cast<const RectangularPixelTopology*>(&(theGeomDet->specificTopology()));

    // barrel ids
    unsigned int layer=0;
    unsigned int ladder=0;
    unsigned int zindex=0;

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
 
    } else if (subid==1) {  // barrel

      hdetr->Fill(detR);
      hdetz->Fill(detZ);
      //hcolsB->Fill(float(cols));
      //hrowsB->Fill(float(rows));
      
      PXBDetId pdetId = PXBDetId(detid);
      unsigned int detTypeP=pdetId.det();
      unsigned int subidP=pdetId.subdetId();
      // Barell layer = 1,2,3
      layer=pdetId.layer();
      // Barrel ladder id 1-20,32,44.
      ladder=pdetId.ladder();
      // Barrel Z-index=1,8
      zindex=pdetId.module();
      if(PRINT)
	cout<<"  barrel det "<<detTypeP<<" "<<subidP
	    <<" layer, ladder. module "<<layer<<" "<<ladder<<" "<<zindex<<endl;
      
      hlayerid->Fill(float(layer));
      if(layer==1) {
	hladder1id->Fill(float(ladder));
	hz1id->Fill(float(zindex));
	++numberOfDetUnits1;
	numOfClustersPerDet1=0;        
      } else if(layer==2) {
	hladder2id->Fill(float(ladder));
	hz2id->Fill(float(zindex));
	++numberOfDetUnits2;
	numOfClustersPerDet2=0;
      } else if(layer==3) {
	hladder3id->Fill(float(ladder));
	hz3id->Fill(float(zindex));
	++numberOfDetUnits3;
	numOfClustersPerDet3=0;
      }
    } // end barrel/forward

    if(PRINT) {
      cout<<"List clusters : "<<endl;
      cout<<"Num Charge Size SizeX SizeY X Y Xmin Xmax Ymin Ymax Edge"
	  <<endl;
    }
    numberOfClusters = 0;

    edm::DetSet<SiPixelCluster>::const_iterator clustIt;
    for (clustIt = DSViter->data.begin(); clustIt != DSViter->data.end(); 
	 clustIt++) {

      numberOfClusters++;
      float ch = (clustIt->charge())/1000.; // convert ke to electrons
      int size = clustIt->size();
      int sizeX = clustIt->sizeX(); //x=row=rfi, 
      int sizeY = clustIt->sizeY(); //y=col=z_global
      float x = clustIt->x(); // cluster position as float
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
            
      if(PRINT) {
	cout<<numberOfClusters<<" "<<ch<<" "<<size<<" "<<sizeX<<" "<<sizeY<<" "
	    <<x<<" "<<y<<" "<<minPixelRow<<" "<<maxPixelRow<<" "<<minPixelCol<<" "
	    <<maxPixelCol<<" "<<edgeHitX<<" "<<edgeHitY<<endl;

	// Get the pixels in the Cluster
	//const vector<Pixel>  = clustIt->pixels();
	const vector<SiPixelCluster::Pixel>& pixelsVec = clustIt->pixels();
	cout<<" Pixels in this cluster "<<endl;
        bool bigInX=false, bigInY=false;
        // Look at pixels in this cluster. ADC is calibrated, in electrons
	bool edgeInX = false; // edge method moved 
	bool edgeInY = false; // to topologu class
	bool cluBigInX = false; // does this clu include a big pixel
	bool cluBigInY = false; // does this clu include a big pixel

        for (unsigned int i = 0;  i < pixelsVec.size(); ++i) {
          float pixx = pixelsVec[i].x; // index as float=i+0.5
          float pixy = pixelsVec[i].y; // same
          float adc = ((pixelsVec[i].adc)/1000);
          //int chan = PixelChannelIdentifier::pixelToChannel(int(pixx),int(pixy));
          bool binInX = (RectangularPixelTopology::isItBigPixelInX(int(pixx)));
          bool bigInY = (RectangularPixelTopology::isItBigPixelInY(int(pixy)));

	  edgeInX = topol->isItEdgePixelInX(int(pixx));
	  edgeInY = topol->isItEdgePixelInY(int(pixy));

	  cout<<i<<" "<<pixx<<" "<<pixy<<" "<<adc<<" "<<bigInX<<" "<<bigInY
	      <<" "<<edgeInX<<" "<<edgeInY<<endl;

	  if(edgeInX) edgeHitX2=true;
	  if(edgeInY) edgeHitY2=true; 
	  if(bigInX) cluBigInX=true;
	  if(bigInY) cluBigInY=true;
	}

	if(edgeHitX != edgeHitX2) 
	  cout<<" wrong egdeX "<<edgeHitX<<" "<<edgeHitX2<<endl;
	if(edgeHitY != edgeHitY2) 
	  cout<<" wrong egdeY "<<edgeHitY<<" "<<edgeHitY2<<endl;

      } // if PRINT
      



      if (subid==1) {  // barrel
	if(layer==1) {
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
	  hcharge2->Fill(ch);
	  hcols2->Fill(y);
	  hrows2->Fill(x);
	  hsize2->Fill(float(size));
	  hsizex2->Fill(float(sizeX));
	  hsizey2->Fill(float(sizeY));
	  numOfClustersPerDet2++;
	  numOfClustersPerLay2++;
	} else if(layer==3) {
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
    } // clusters 


    
    if (subid==1) {  // barrel
      if(layer==1) {
	hclusPerDet1->Fill(float(numOfClustersPerDet1));
	if(PRINT) cout<<"Lay1: number of clusters per det = "<<numOfClustersPerDet1<<endl;
      } else if(layer==2) {
	hclusPerDet2->Fill(float(numOfClustersPerDet2));
	if(PRINT) cout<<"Lay2: number of clusters per det = "<<numOfClustersPerDet1<<endl;
      } else if(layer==3) { 
	hclusPerDet3->Fill(float(numOfClustersPerDet3));
	if(PRINT) cout<<"Lay3: number of clusters per det = "<<numOfClustersPerDet1<<endl;
      }
    } // end barrel/endcaps

  } // detunits loop
  

  if(PRINT) {
    cout<<"Number of clusters per Lay1,2,3: "<<numOfClustersPerLay1<<" "
	<<numOfClustersPerLay2<<" "<<numOfClustersPerLay3<<endl;
    cout<<"Number of dets with clus in Lay1,2,3: "<<numberOfDetUnits1<<" "
	<<numberOfDetUnits2<<" "<<numberOfDetUnits3<<endl;
  }
  hclusPerLay1->Fill(float(numOfClustersPerLay1));
  hclusPerLay2->Fill(float(numOfClustersPerLay2));
  hclusPerLay3->Fill(float(numOfClustersPerLay3));
  hdetsPerLay1->Fill(float(numberOfDetUnits1));
  hdetsPerLay2->Fill(float(numberOfDetUnits2));
  hdetsPerLay3->Fill(float(numberOfDetUnits3));

} // end 

//define this as a plug-in
DEFINE_FWK_MODULE(ReadPixClusters);
