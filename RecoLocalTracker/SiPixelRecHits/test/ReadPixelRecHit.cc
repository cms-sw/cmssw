//--------------------------------------------
// File: ReadPixelRecHit.cc
// Description:  see ReadPixelRecHit.h
// Author:  J.Sheav (JHU)
//          11/8/06: New loop over rechits and InputTag, V.Chiochia
// Creation Date:  OGU Aug. 1 2005 Initial version.
// Add occupancy histograms. D.Kotlinski. 10/06
//--------------------------------------------
#include <memory>
#include <string>
#include <iostream>

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"

//#include "Geometry/CommonDetUnit/interface/TrackingGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/CommonDetUnit/interface/GeomDetType.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"

#include "Geometry/CommonDetUnit/interface/PixelGeomDetUnit.h"
#include "Geometry/CommonDetUnit/interface/PixelGeomDetType.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/CommonTopologies/interface/PixelTopology.h"

#include "RecoLocalTracker/SiPixelRecHits/test/ReadPixelRecHit.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHitCollection.h"

#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/SiPixelDetId/interface/PXBDetId.h"
#include "DataFormats/SiPixelDetId/interface/PXFDetId.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"

// To use root histos
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

using namespace std;

//#define DO_HISTO Defined in *.h

//----------------------------------------------------------------------
ReadPixelRecHit::ReadPixelRecHit(edm::ParameterSet const& conf)
    : conf_(conf), src_(conf.getParameter<edm::InputTag>("src")) {
  print = conf.getUntrackedParameter<bool>("Verbosity", false);

  cout << " Verbosity " << print << endl;
}
//----------------------------------------------------------------------
// Virtual destructor needed.
ReadPixelRecHit::~ReadPixelRecHit() {}
//---------------------------------------------------------------------
// ------------ method called at the begining   ------------
void ReadPixelRecHit::beginJob() {
  cout << "Initialize PixelRecHitTest " << endl;

#ifdef DO_HISTO
  // put here whatever you want to do at the beginning of the job
  // hFile = new TFile ( "histo.root", "RECREATE" );

  // NEW way to use root (from 2.0.0?)
  edm::Service<TFileService> fs;

  // Histos go to a subdirectory "PixRecHits")
  //TFileDirectory subDir = fs->mkdir( "mySubDirectory" );
  //TFileDirectory subSubDir = subDir.mkdir( "mySubSubDirectory" );
  //h_pt    = fs->make<TH1F>( "pt"  , "p_{t}", 100,  0., ptMax_ );

  hpixid = fs->make<TH1F>("hpixid", "Pix det id", 10, 0., 10.);
  hpixsubid = fs->make<TH1F>("hpixsubid", "Pix Barrel id", 10, 0., 10.);
  hlayerid = fs->make<TH1F>("hlayerid", "Pix layer id", 10, 0., 10.);
  hladder1id = fs->make<TH1F>("hladder1id", "Ladder L1 id", 50, 0., 50.);
  hladder2id = fs->make<TH1F>("hladder2id", "Ladder L2 id", 50, 0., 50.);
  hladder3id = fs->make<TH1F>("hladder3id", "Ladder L3 id", 50, 0., 50.);
  hz1id = fs->make<TH1F>("hz1id", "Z-index id L1", 10, 0., 10.);
  hz2id = fs->make<TH1F>("hz2id", "Z-index id L2", 10, 0., 10.);
  hz3id = fs->make<TH1F>("hz3id", "Z-index id L3", 10, 0., 10.);

  hrecHitsPerDet1 = fs->make<TH1F>("hrecHitsPerDet1", "RecHits per det l1", 200, -0.5, 199.5);
  hrecHitsPerDet2 = fs->make<TH1F>("hrecHitsPerDet2", "RecHits per det l2", 200, -0.5, 199.5);
  hrecHitsPerDet3 = fs->make<TH1F>("hrecHitsPerDet3", "RecHits per det l3", 200, -0.5, 199.5);
  hrecHitsPerLay1 = fs->make<TH1F>("hrecHitsPerLay1", "RecHits per layer l1", 2000, -0.5, 1999.5);
  hrecHitsPerLay2 = fs->make<TH1F>("hrecHitsPerLay2", "RecHits per layer l2", 2000, -0.5, 1999.5);

  hrecHitsPerLay3 = fs->make<TH1F>("hrecHitsPerLay3", "RecHits per layer l3", 2000, -0.5, 1999.5);
  hdetsPerLay1 = fs->make<TH1F>("hdetsPerLay1", "Full dets per layer l1", 161, -0.5, 160.5);
  hdetsPerLay3 = fs->make<TH1F>("hdetsPerLay3", "Full dets per layer l3", 353, -0.5, 352.5);
  hdetsPerLay2 = fs->make<TH1F>("hdetsPerLay2", "Full dets per layer l2", 257, -0.5, 256.5);

  hcharge1 = fs->make<TH1F>("hcharge1", "Clu charge l1", 200, 0., 200.);  //in ke
  hcharge2 = fs->make<TH1F>("hcharge2", "Clu charge l2", 200, 0., 200.);
  hcharge3 = fs->make<TH1F>("hcharge3", "Clu charge l3", 200, 0., 200.);
  hadcCharge1 = fs->make<TH1F>("hadcCharge1", "pix charge l1", 50, 0., 50.);            //in ke
  hadcCharge2 = fs->make<TH1F>("hadcCharge2", "pix charge l2", 50, 0., 50.);            //in ke
  hadcCharge3 = fs->make<TH1F>("hadcCharge3", "pix charge l3", 50, 0., 50.);            //in ke
  hadcCharge1big = fs->make<TH1F>("hadcCharge1big", "big pix charge l1", 50, 0., 50.);  //in ke

  hxpos1 = fs->make<TH1F>("hxpos1", "Layer 1 cols", 700, -3.5, 3.5);
  hxpos2 = fs->make<TH1F>("hxpos2", "Layer 2 cols", 700, -3.5, 3.5);
  hxpos3 = fs->make<TH1F>("hxpos3", "Layer 3 cols", 700, -3.5, 3.5);

  hypos1 = fs->make<TH1F>("hypos1", "Layer 1 rows", 200, -1., 1.);
  hypos2 = fs->make<TH1F>("hypos2", "Layer 2 rows", 200, -1., 1.);
  hypos3 = fs->make<TH1F>("hypos3", "layer 3 rows", 200, -1., 1.);

  hsize1 = fs->make<TH1F>("hsize1", "layer 1 clu size", 100, -0.5, 99.5);
  hsize2 = fs->make<TH1F>("hsize2", "layer 2 clu size", 100, -0.5, 99.5);
  hsize3 = fs->make<TH1F>("hsize3", "layer 3 clu size", 100, -0.5, 99.5);

  hAlignErrorX1 = fs->make<TH1F>("hAlignErrorX1", "Align error Layer 1 X", 100, 0.0, 100.);
  hAlignErrorY1 = fs->make<TH1F>("hAlignErrorY1", "Align error Layer 1 Y", 100, 0.0, 100.);
  hAlignErrorX2 = fs->make<TH1F>("hAlignErrorX2", "Align error Layer 2 X", 100, 0.0, 100.);
  hAlignErrorY2 = fs->make<TH1F>("hAlignErrorY2", "Align error Layer 2 Y", 100, 0.0, 100.);
  hAlignErrorX3 = fs->make<TH1F>("hAlignErrorX3", "Align error Layer 3 X", 100, 0.0, 100.);
  hAlignErrorY3 = fs->make<TH1F>("hAlignErrorY3", "Align error Layer 3 Y", 100, 0.0, 100.);
  hAlignErrorX4 = fs->make<TH1F>("hAlignErrorX4", "Align error Disk -1 X", 100, 0.0, 100.);
  hAlignErrorY4 = fs->make<TH1F>("hAlignErrorY4", "Align error Disk -1 Y", 100, 0.0, 100.);
  hAlignErrorX5 = fs->make<TH1F>("hAlignErrorX5", "Align error Disk -2 X", 100, 0.0, 100.);
  hAlignErrorY5 = fs->make<TH1F>("hAlignErrorY5", "Align error Disk -2 Y", 100, 0.0, 100.);
  hAlignErrorX6 = fs->make<TH1F>("hAlignErrorX6", "Align error Disk +1 X", 100, 0.0, 100.);
  hAlignErrorY6 = fs->make<TH1F>("hAlignErrorY6", "Align error Disk +1 Y", 100, 0.0, 100.);
  hAlignErrorX7 = fs->make<TH1F>("hAlignErrorX7", "Align error Disk +2 X", 100, 0.0, 100.);
  hAlignErrorY7 = fs->make<TH1F>("hAlignErrorY7", "Align error Disk +2 Y", 100, 0.0, 100.);

  hErrorX1 = fs->make<TH1F>("hErrorX1", "Error Layer 1 X", 100, 0.0, 100.);
  hErrorY1 = fs->make<TH1F>("hErrorY1", "Error Layer 1 Y", 100, 0.0, 100.);
  hErrorX2 = fs->make<TH1F>("hErrorX2", "Error Layer 2 X", 100, 0.0, 100.);
  hErrorY2 = fs->make<TH1F>("hErrorY2", "Error Layer 2 Y", 100, 0.0, 100.);
  hErrorX3 = fs->make<TH1F>("hErrorX3", "Error Layer 3 X", 100, 0.0, 100.);
  hErrorY3 = fs->make<TH1F>("hErrorY3", "Error Layer 3 Y", 100, 0.0, 100.);
  hErrorX4 = fs->make<TH1F>("hErrorX4", "Error Disk -1 X", 100, 0.0, 100.);
  hErrorY4 = fs->make<TH1F>("hErrorY4", "Error Disk -1 Y", 100, 0.0, 100.);
  hErrorX5 = fs->make<TH1F>("hErrorX5", "Error Disk -2 X", 100, 0.0, 100.);
  hErrorY5 = fs->make<TH1F>("hErrorY5", "Error Disk -2 Y", 100, 0.0, 100.);
  hErrorX6 = fs->make<TH1F>("hErrorX6", "Error Disk +1 X", 100, 0.0, 100.);
  hErrorY6 = fs->make<TH1F>("hErrorY6", "Error Disk +1 Y", 100, 0.0, 100.);
  hErrorX7 = fs->make<TH1F>("hErrorX7", "Error Disk +2 X", 100, 0.0, 100.);
  hErrorY7 = fs->make<TH1F>("hErrorY7", "Error Disk +2 Y", 100, 0.0, 100.);

  hsizex1 = fs->make<TH1F>("hsizex1", "lay1 clu size in x", 10, -0.5, 9.5);
  hsizex2 = fs->make<TH1F>("hsizex2", "lay2 clu size in x", 10, -0.5, 9.5);
  hsizex3 = fs->make<TH1F>("hsizex3", "lay3 clu size in x", 10, -0.5, 9.5);
  hsizey1 = fs->make<TH1F>("hsizey1", "lay1 clu size in y", 20, -0.5, 19.5);
  hsizey2 = fs->make<TH1F>("hsizey2", "lay2 clu size in y", 20, -0.5, 19.5);
  hsizey3 = fs->make<TH1F>("hsizey3", "lay3 clu size in y", 20, -0.5, 19.5);

  hdetr = fs->make<TH1F>("hdetr", "det r", 150, 0., 15.);
  hdetz = fs->make<TH1F>("hdetz", "det z", 520, -26., 26.);

  // Forward edcaps
  hdetrF = fs->make<TH1F>("hdetrF", "Fdet r", 150, 5., 20.);
  hdetzF = fs->make<TH1F>("hdetzF", "Fdet z", 600, -60., 60.);

  hdisk = fs->make<TH1F>("hdisk", "FPix disk id", 10, 0., 10.);
  hblade = fs->make<TH1F>("hblade", "FPix blade id", 30, 0., 30.);
  hmodule = fs->make<TH1F>("hmodule", "FPix plaq. id", 10, 0., 10.);
  hpanel = fs->make<TH1F>("hpanel", "FPix panel id", 10, 0., 10.);
  hside = fs->make<TH1F>("hside", "FPix size id", 10, 0., 10.);

  hcharge1F = fs->make<TH1F>("hcharge1F", "Clu charge 21", 200, 0., 200.);  //in ke
  hcharge2F = fs->make<TH1F>("hcharge2F", "Clu charge 22", 200, 0., 200.);
  hxpos1F = fs->make<TH1F>("hxpos1F", "Disk 1 cols", 700, -3.5, 3.5);
  hxpos2F = fs->make<TH1F>("hxpos2F", "Disk 2 cols", 700, -3.5, 3.5);
  hypos1F = fs->make<TH1F>("hypos1F", "Disk 1 rows", 200, -1., 1.);
  hypos2F = fs->make<TH1F>("hypos2F", "Disk 2 rows", 200, -1., 1.);
  hsize1F = fs->make<TH1F>("hsize1F", "Disk 1 clu size", 100, -0.5, 99.5);
  hsize2F = fs->make<TH1F>("hsize2F", "Disk 2 clu size", 100, -0.5, 99.5);
  hsizex1F = fs->make<TH1F>("hsizex1F", "d1 clu size in x", 10, -0.5, 9.5);
  hsizex2F = fs->make<TH1F>("hsizex2F", "d2 clu size in x", 10, -0.5, 9.5);
  hsizey1F = fs->make<TH1F>("hsizey1F", "d1 clu size in y", 20, -0.5, 19.5);
  hsizey2F = fs->make<TH1F>("hsizey2F", "d2 clu size in y", 20, -0.5, 19.5);
  hadcCharge1F = fs->make<TH1F>("hadcCharge1F", "pix charge d1", 50, 0., 50.);  //in ke
  hadcCharge2F = fs->make<TH1F>("hadcCharge2F", "pix charge d2", 50, 0., 50.);  //in ke

  hrecHitsPerDet1F = fs->make<TH1F>("hrecHitsPerDet1F", "RecHits per det l1", 200, -0.5, 199.5);
  hrecHitsPerDet2F = fs->make<TH1F>("hrecHitsPerDet2F", "RecHits per det l2", 200, -0.5, 199.5);
  hrecHitsPerLay1F = fs->make<TH1F>("hrecHitsPerLay1F", "RecHits per layer l1", 2000, -0.5, 1999.5);
  hrecHitsPerLay2F = fs->make<TH1F>("hrecHitsPerLay2F", "RecHits per layer l2", 2000, -0.5, 1999.5);
  hdetsPerLay1F = fs->make<TH1F>("hdetsPerLay1F", "Full dets per layer l1", 161, -0.5, 160.5);
  hdetsPerLay2F = fs->make<TH1F>("hdetsPerLay2F", "Full dets per layer l2", 257, -0.5, 256.5);

  hErrorXB = fs->make<TProfile>("hErrorXB", "bpix x errors per ladder", 220, 0., 220., 0.0, 1000.);
  hErrorXF = fs->make<TProfile>("hErrorXF", "fpix x errors per ladder", 100, 0., 100., 0.0, 1000.);
  hErrorYB = fs->make<TProfile>("hErrorYB", "bpix y errors per ladder", 220, 0., 220., 0.0, 1000.);
  hErrorYF = fs->make<TProfile>("hErrorYF", "fpix y errors per ladder", 100, 0., 100., 0.0, 1000.);

  hAErrorXB = fs->make<TProfile>("hAErrorXB", "bpix x errors per ladder", 220, 0., 220., 0.0, 1000.);
  hAErrorXF = fs->make<TProfile>("hAErrorXF", "fpix x errors per ladder", 100, 0., 100., 0.0, 1000.);
  hAErrorYB = fs->make<TProfile>("hAErrorYB", "bpix y errors per ladder", 220, 0., 220., 0.0, 1000.);
  hAErrorYF = fs->make<TProfile>("hAErrorYF", "fpix y errors per ladder", 100, 0., 100., 0.0, 1000.);

  cout << " book histos " << endl;

#endif
}
//-----------------------------------------------------------------------
void ReadPixelRecHit::endJob() { cout << " End PixelRecHitTest " << endl; }
//---------------------------------------------------------------------
// Functions that gets called by framework every event
void ReadPixelRecHit::analyze(const edm::Event& e, const edm::EventSetup& es) {
  using namespace edm;
  const bool localPrint = false;
  //const bool localPrint = true;

  // Get event setup (to get global transformation)
  edm::ESHandle<TrackerGeometry> geom;
  es.get<TrackerDigiGeometryRecord>().get(geom);
  const TrackerGeometry& theTracker(*geom);

  edm::Handle<SiPixelRecHitCollection> recHitColl;
  e.getByLabel(src_, recHitColl);

  if (print)
    cout << " FOUND " << (recHitColl.product())->dataSize() << " Pixel Hits" << endl;

  SiPixelRecHitCollection::const_iterator recHitIdIterator = (recHitColl.product())->begin();
  SiPixelRecHitCollection::const_iterator recHitIdIteratorEnd = (recHitColl.product())->end();

  int numberOfDetUnits = 0;
  int numOfRecHits = 0;

  int numberOfDetUnits1 = 0;
  int numOfRecHitsPerDet1 = 0;
  int numOfRecHitsPerLay1 = 0;
  int numberOfDetUnits2 = 0;
  int numOfRecHitsPerDet2 = 0;
  int numOfRecHitsPerLay2 = 0;
  int numberOfDetUnits3 = 0;
  int numOfRecHitsPerDet3 = 0;
  int numOfRecHitsPerLay3 = 0;
  int numberOfDetUnits1F = 0;
  int numOfRecHitsPerDet1F = 0;
  int numOfRecHitsPerLay1F = 0;
  int numberOfDetUnits2F = 0;
  int numOfRecHitsPerDet2F = 0;
  int numOfRecHitsPerLay2F = 0;

  // Loop over Detector IDs
  for (; recHitIdIterator != recHitIdIteratorEnd; recHitIdIterator++) {
    SiPixelRecHitCollection::DetSet detset = *recHitIdIterator;

    DetId detId = DetId(detset.detId());    // Get the Detid object
    unsigned int detType = detId.det();     // det type, tracker=1
    unsigned int subid = detId.subdetId();  //subdetector type, barrel=1, fpix=2

    if (detType != 1)
      cout << " wrong det id " << detType << endl;
    ;  //look only at tracker
    if (!((subid == 1) || (subid == 2)))
      cout << " wrong sub det id " << subid << endl;  // look only at bpix&fpix

    if (print)
      cout << "     Det ID " << detId.rawId() << endl;

    //  Get rechits
    if (detset.empty())
      continue;

    // Get the geometrical information for this det
    const PixelGeomDetUnit* theGeomDet = dynamic_cast<const PixelGeomDetUnit*>(theTracker.idToDet(detId));
    double detZ = theGeomDet->surface().position().z();
    double detR = theGeomDet->surface().position().perp();

    //const BoundPlane& plane = theGeomDet->surface(); //for transf.  unused
    //double detThick = theGeomDet->specificSurface().bounds().thickness(); unused

    //const RectangularPixelTopology * topol =
    //dynamic_cast<const RectangularPixelTopology*>(&(theGeomDet->specificTopology()));

    const PixelTopology* topol = &(theGeomDet->specificTopology());

    //int cols = theGeomDet->specificTopology().ncolumns(); UNUSED
    //int rows = theGeomDet->specificTopology().nrows();

    float alignErrorX = 0., alignErrorY = 0.;
    LocalError lape = theGeomDet->localAlignmentError();
    //cout<<lape.valid()<<endl;
    if (lape.valid()) {
      if (lape.xx() > 0.)
        alignErrorX = sqrt(lape.xx()) * 1E4;
      //float tmp12= sqrt(lape.xy())*1E4;
      if (lape.yy() > 0.)
        alignErrorY = sqrt(lape.yy()) * 1E4;
      if (print)
        cout << " Alignment errors " << alignErrorX << " " << alignErrorY << endl;
      //cout<<" Alignment errors "<<alignErrorX<<" "<<alignErrorY<<endl;
    }

    unsigned int layer = 0, disk = 0, ladder = 0, zindex = 0, blade = 0, panel = 0, side = 0;
    if (subid == 1) {  // Subdet it, pix barrel=1
      ++numberOfDetUnits;

      PXBDetId pdetId = PXBDetId(detId);
      //unsigned int detTypeP=pdetId.det();   unused
      //unsigned int subidP=pdetId.subdetId(); unused
      // Barell layer = 1,2,3
      layer = pdetId.layer();
      // Barrel ladder id 1-20,32,44.
      ladder = pdetId.ladder();
      // Barrel Z-index=1,8
      zindex = pdetId.module();
      if (zindex < 5)
        side = 1;
      else
        side = 2;

      if (localPrint)
        cout << " Layer " << layer << " ladder " << ladder << " z " << zindex << endl;
        //<<pdetId.rawId()<<" "<<pdetId.null()<<detTypeP<<" "<<subidP<<" "<<endl;

#ifdef DO_HISTO
      hdetr->Fill(detR);
      hdetz->Fill(detZ);
      //hcolsB->Fill(float(cols));
      //hyposB->Fill(float(rows));
      hlayerid->Fill(float(layer));
#endif

    } else {  // FPIX -------------------------------------

      // test cols & rows
      //cout<<" det z/r "<<detZ<<" "<<detR<<" "<<detThick<<" "
      //  <<cols<<" "<<rows<<endl;

      PXFDetId pdetId = PXFDetId(detId.rawId());
      disk = pdetId.disk();                   //1,2,3
      blade = pdetId.blade();                 //1-24
      side = pdetId.side();                   //size=1 for -z, 2 for +z
      panel = pdetId.panel();                 //panel=1,2
      unsigned int module = pdetId.module();  // plaquette

      if (print)
        cout << " forward hit " << side << " " << disk << " " << blade << " " << panel << " " << module << endl;

#ifdef DO_HISTO
      hdetrF->Fill(detR);
      hdetzF->Fill(detZ);
      hdisk->Fill(float(disk));
      hblade->Fill(float(blade));
      hmodule->Fill(float(module));
      hpanel->Fill(float(panel));
      hside->Fill(float(side));
#endif

    }  // end BPix FPix if

#ifdef DO_HISTO
    if (layer == 1) {
      hladder1id->Fill(float(ladder));
      hz1id->Fill(float(zindex));
      hAlignErrorX1->Fill(alignErrorX);
      hAlignErrorY1->Fill(alignErrorY);
      hAErrorXB->Fill(float(ladder + (110 * (side - 1))), alignErrorX);
      hAErrorYB->Fill(float(ladder + (110 * (side - 1))), alignErrorY);
      ++numberOfDetUnits1;
      numOfRecHitsPerDet1 = 0;

    } else if (layer == 2) {
      hladder2id->Fill(float(ladder));
      hz2id->Fill(float(zindex));
      hAlignErrorX2->Fill(alignErrorX);
      hAlignErrorY2->Fill(alignErrorY);
      hAErrorXB->Fill(float(ladder + 25 + (110 * (side - 1))), alignErrorX);
      hAErrorYB->Fill(float(ladder + 25 + (110 * (side - 1))), alignErrorY);
      ++numberOfDetUnits2;
      numOfRecHitsPerDet2 = 0;

    } else if (layer == 3) {
      hladder3id->Fill(float(ladder));
      hz3id->Fill(float(zindex));
      hAlignErrorX3->Fill(alignErrorX);
      hAlignErrorY3->Fill(alignErrorY);
      hAErrorXB->Fill(float(ladder + 60 + (110 * (side - 1))), alignErrorX);
      hAErrorYB->Fill(float(ladder + 60 + (110 * (side - 1))), alignErrorY);
      ++numberOfDetUnits3;
      numOfRecHitsPerDet3 = 0;

    } else if (disk == 1) {
      ++numberOfDetUnits1F;
      numOfRecHitsPerDet1F = 0;
      if (side == 1) {
        hAlignErrorX4->Fill(alignErrorX);
        hAlignErrorY4->Fill(alignErrorY);
        hAErrorXF->Fill(float(blade + 25), alignErrorX);
        hAErrorYF->Fill(float(blade + 25), alignErrorY);
      } else {
        hAlignErrorX6->Fill(alignErrorX);
        hAlignErrorY6->Fill(alignErrorY);
        hAErrorXF->Fill(float(blade + 50), alignErrorX);
        hAErrorYF->Fill(float(blade + 50), alignErrorY);
      }

    } else if (disk == 2) {
      ++numberOfDetUnits2F;
      numOfRecHitsPerDet2F = 0;
      if (side == 1) {
        hAlignErrorX5->Fill(alignErrorX);
        hAlignErrorY5->Fill(alignErrorY);
        hAErrorXF->Fill(float(blade), alignErrorX);
        hAErrorYF->Fill(float(blade), alignErrorY);
      } else {
        hAlignErrorX7->Fill(alignErrorX);
        hAlignErrorY7->Fill(alignErrorY);
        hAErrorXF->Fill(float(blade + 75), alignErrorX);
        hAErrorYF->Fill(float(blade + 75), alignErrorY);
      }
    }
#endif

    //----Loop over rechits for this detId
    SiPixelRecHitCollection::DetSet::const_iterator pixeliter = detset.begin();
    SiPixelRecHitCollection::DetSet::const_iterator rechitRangeIteratorEnd = detset.end();
    for (; pixeliter != rechitRangeIteratorEnd; ++pixeliter) {  //loop on the rechit

      numOfRecHits++;

      // RecHit local position is now transient,
      // one needs to run tracking to get position OR rerun localreco
      LocalPoint lp = pixeliter->localPosition();
      LocalError le = pixeliter->localPositionError();
      float xRecHit = lp.x();
      float yRecHit = lp.y();
      float xerror = sqrt(le.xx()) * 1E4;
      float yerror = sqrt(le.yy()) * 1E4;
      if (print)
        cout << " RecHit: " << numOfRecHits << " x/y " << xRecHit << " " << yRecHit << " errors x/y " << xerror << " "
             << yerror << endl;

      //MeasurementPoint mp = topol->measurementPosition(xRecHit,yRecHit);
      //GlobalPoint GP = PixGeom->surface().toGlobal(Local3DPoint(lp));

      // Get cluster
      edm::Ref<edmNew::DetSetVector<SiPixelCluster>, SiPixelCluster> const& clust = pixeliter->cluster();

      float ch = (clust->charge()) / 1000.;  // convert ke to electrons
      int size = clust->size();
      int sizeX = clust->sizeX();
      int sizeY = clust->sizeY();
      float xClu = clust->x();
      float yClu = clust->y();
      int maxPixelCol = clust->maxPixelCol();
      int maxPixelRow = clust->maxPixelRow();
      int minPixelCol = clust->minPixelCol();
      int minPixelRow = clust->minPixelRow();

      // unsigned int geoId = clust->geographicalId(); // alsways 0

      // edge method moved to topologu class
      bool edgeHitX = (topol->isItEdgePixelInX(minPixelRow)) || (topol->isItEdgePixelInX(maxPixelRow));
      bool edgeHitY = (topol->isItEdgePixelInY(minPixelCol)) || (topol->isItEdgePixelInY(maxPixelCol));

      if (print)
        cout << "Clu: charge " << ch << " size " << size << " size x/y " << sizeX << " " << sizeY << " meas. " << xClu
             << " " << yClu << " edge " << edgeHitX << " " << edgeHitY << endl;

      if (print)
        cout << " pixels:" << endl;

      // Get the pixels in the Cluster
      const vector<SiPixelCluster::Pixel>& pixelsVec = clust->pixels();
      //if(localPrint) cout<<" Pixels in this cluster "<<endl;
      map<unsigned int, float, less<unsigned int> > chanMap;  // Channel map
      // Look at pixels in this cluster. ADC is calibrated, in electrons
      for (unsigned int i = 0; i < pixelsVec.size(); ++i) {
        float pixx = pixelsVec[i].x;  // index as a float so = i+0.5
        float pixy = pixelsVec[i].y;
        float adc = ((pixelsVec[i].adc) / 1000);  // in kelec.

        // OLD way
        //int chan = PixelChannelIdentifier::pixelToChannel(int(pixx),int(pixy));
        bool bigInX = topol->isItBigPixelInX(int(pixx));
        bool bigInY = topol->isItBigPixelInY(int(pixy));

        bool edgeInX = topol->isItEdgePixelInX(int(pixx));
        bool edgeInY = topol->isItEdgePixelInY(int(pixy));

        if (print)
          cout << i << " index " << pixx << " " << pixy << " adc " << adc << " edge " << edgeInX << " " << edgeInY
               << " big " << bigInX << " " << bigInY << endl;

          //if(print && sizeX==1 && bigInX)
          //cout<<" single big x "<<xClu<<" "<<pixx<<" "<<endl;
          //if(print && sizeY==1 && bigInY)
          //cout<<" single big y "<<yClu<<" "<<pixy<<" "<<endl;
#ifdef DO_HISTO
        if (layer == 1) {
          hadcCharge1->Fill(adc);
          //if(bigInX || bigInY) hadcCharge1big->Fill(adc);
        } else if (layer == 2) {
          hadcCharge2->Fill(adc);
        } else if (layer == 3) {
          hadcCharge3->Fill(adc);
        } else if (disk == 1) {
          hadcCharge1F->Fill(adc);
        } else if (disk == 2) {
          hadcCharge2F->Fill(adc);
        }
#endif
      }  // End pixel loop

#ifdef DO_HISTO
      if (layer == 1) {
        hcharge1->Fill(ch);
        hxpos1->Fill(yRecHit);
        hypos1->Fill(xRecHit);
        hsize1->Fill(float(size));
        hsizex1->Fill(float(sizeX));
        hsizey1->Fill(float(sizeY));
        numOfRecHitsPerDet1++;
        numOfRecHitsPerLay1++;
        hErrorX1->Fill(xerror);
        hErrorY1->Fill(yerror);
        hErrorXB->Fill(float(ladder + (110 * (side - 1))), xerror);
        hErrorYB->Fill(float(ladder + (110 * (side - 1))), yerror);

      } else if (layer == 2) {  // layer 2

        hcharge2->Fill(ch);
        hxpos2->Fill(yRecHit);
        hypos2->Fill(xRecHit);
        hsize2->Fill(float(size));
        hsizex2->Fill(float(sizeX));
        hsizey2->Fill(float(sizeY));
        numOfRecHitsPerDet2++;
        numOfRecHitsPerLay2++;
        hErrorX2->Fill(xerror);
        hErrorY2->Fill(yerror);
        hErrorXB->Fill(float(ladder + 25 + (110 * (side - 1))), xerror);
        hErrorYB->Fill(float(ladder + 25 + (110 * (side - 1))), yerror);

      } else if (layer == 3) {  // Layer 3

        hcharge3->Fill(ch);
        hxpos3->Fill(yRecHit);
        hypos3->Fill(xRecHit);
        hsize3->Fill(float(size));
        hsizex3->Fill(float(sizeX));
        hsizey3->Fill(float(sizeY));
        numOfRecHitsPerDet3++;
        numOfRecHitsPerLay3++;
        hErrorX3->Fill(xerror);
        hErrorY3->Fill(yerror);
        hErrorXB->Fill(float(ladder + 60 + (110 * (side - 1))), xerror);
        hErrorYB->Fill(float(ladder + 60 + (110 * (side - 1))), yerror);

      } else if (disk == 1) {
        hcharge1F->Fill(ch);
        hxpos1F->Fill(yRecHit);
        hypos1F->Fill(xRecHit);
        hsize1F->Fill(float(size));
        hsizex1F->Fill(float(sizeX));
        hsizey1F->Fill(float(sizeY));
        if (side == 1) {  // -z
          hErrorX4->Fill(xerror);
          hErrorY4->Fill(yerror);
          hErrorXF->Fill(float(blade + 25), xerror);
          hErrorYF->Fill(float(blade + 25), yerror);
        } else {  // +z
          hErrorX6->Fill(xerror);
          hErrorY6->Fill(yerror);
          hErrorXF->Fill(float(blade + 50), xerror);
          hErrorYF->Fill(float(blade + 50), yerror);
        }
        numOfRecHitsPerDet1F++;
        numOfRecHitsPerLay1F++;
      } else if (disk == 2) {  // disk 2

        hcharge2F->Fill(ch);
        hxpos2F->Fill(yRecHit);
        hypos2F->Fill(xRecHit);
        hsize2F->Fill(float(size));
        hsizex2F->Fill(float(sizeX));
        hsizey2F->Fill(float(sizeY));
        numOfRecHitsPerDet2F++;
        numOfRecHitsPerLay2F++;
        if (side == 1) {  // -z
          hErrorX5->Fill(xerror);
          hErrorY5->Fill(yerror);
          hErrorXF->Fill(float(blade), xerror);
          hErrorYF->Fill(float(blade), yerror);
        } else {  // +z
          hErrorX7->Fill(xerror);
          hErrorY7->Fill(yerror);
          hErrorXF->Fill(float(blade + 75), xerror);
          hErrorYF->Fill(float(blade + 75), yerror);
        }
      }
#endif

    }  // End RecHit loop

#ifdef DO_HISTO
    if (layer == 1)
      hrecHitsPerDet1->Fill(float(numOfRecHitsPerDet1));
    else if (layer == 2)
      hrecHitsPerDet2->Fill(float(numOfRecHitsPerDet2));
    else if (layer == 3)
      hrecHitsPerDet3->Fill(float(numOfRecHitsPerDet3));
    else if (disk == 1)
      hrecHitsPerDet1F->Fill(float(numOfRecHitsPerDet1F));
    else if (disk == 2)
      hrecHitsPerDet2F->Fill(float(numOfRecHitsPerDet2F));
#endif

  }  // End Det loop

#ifdef DO_HISTO
  if (print)
    cout << " Rechits per layer " << numOfRecHitsPerLay1 << " " << numOfRecHitsPerLay2 << " " << numOfRecHitsPerLay3
         << " " << numOfRecHitsPerLay1F << " " << numOfRecHitsPerLay2F << endl;
  hrecHitsPerLay1->Fill(float(numOfRecHitsPerLay1));
  hrecHitsPerLay2->Fill(float(numOfRecHitsPerLay2));
  hrecHitsPerLay3->Fill(float(numOfRecHitsPerLay3));
  hdetsPerLay1->Fill(float(numberOfDetUnits1));
  hdetsPerLay2->Fill(float(numberOfDetUnits2));
  hdetsPerLay3->Fill(float(numberOfDetUnits3));

  hrecHitsPerLay1F->Fill(float(numOfRecHitsPerLay1F));
  hrecHitsPerLay2F->Fill(float(numOfRecHitsPerLay2F));
  hdetsPerLay1F->Fill(float(numberOfDetUnits1F));
  hdetsPerLay2F->Fill(float(numberOfDetUnits2F));
  hdetsPerLay3->Fill(float(numberOfDetUnits3));
#endif
}

//define this as a plug-in
DEFINE_FWK_MODULE(ReadPixelRecHit);
