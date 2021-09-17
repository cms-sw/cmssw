// -*- C++ -*-
//
// Package:    TrackerMapTool
// Class:      TrackerMapTool
//
/**\class TrackerMapTool TrackerMapTool.cc 

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Filippo Ambroglini
//         Created:  Tue Jul 26 08:47:57 CEST 2005
//
//

// system include files
#include <memory>

#include <vector>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerNumberingBuilder/interface/GeometricDet.h"
#include "Geometry/CommonTopologies/interface/PixelTopology.h"
#include "Geometry/CommonTopologies/interface/StripTopology.h"
#include "Geometry/CommonDetUnit/interface/PixelGeomDetType.h"
#include "Geometry/TrackerGeometryBuilder/interface/StripGeomDetType.h"

#include "Geometry/CommonDetUnit/interface/PixelGeomDetUnit.h"
#include "DataFormats/GeometrySurface/interface/BoundSurface.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
//
//
// class decleration
//

class TrackerMapTool : public edm::one::EDAnalyzer<> {
public:
  explicit TrackerMapTool(const edm::ParameterSet&);
  ~TrackerMapTool() override;

  void beginJob() override {}
  void analyze(edm::Event const& iEvent, edm::EventSetup const&) override;
  void endJob() override {}
};

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
TrackerMapTool::TrackerMapTool(const edm::ParameterSet& iConfig) {
  //now do what ever initialization is needed
}

TrackerMapTool::~TrackerMapTool() {
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
}

//
// member functions
//

int layerno(int subdet, int leftright, int layer) {
  if (subdet == 6 && leftright == 1)
    return (10 - layer);
  if (subdet == 6 && leftright == 2)
    return (layer + 21);
  if (subdet == 4 && leftright == 1)
    return (4 - layer + 9);
  if (subdet == 4 && leftright == 2)
    return (layer + 18);
  if (subdet == 2 && leftright == 1)
    return (4 - layer + 12);
  if (subdet == 2 && leftright == 2)
    return (layer + 15);
  if (subdet == 1)
    return (layer + 30);
  if (subdet == 3)
    return (layer + 33);
  if (subdet == 5)
    return (layer + 37);
  // 2009-08-26 Michael Case: to get rid of a compiler warning about control reaching
  // the end of a non-void function I put return -1 here.  This changed the output
  // of the test so I changed it to return 0 to match the "before my changes" run
  // of trackerMap_cfg.py.
  return 0;  // this was added.  No checks have been mad where layerno is used.
}
// ------------ method called to produce the data  ------------
void TrackerMapTool::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  edm::LogInfo("TrackerMapTool") << "Here I am";

  //
  // get the TrackerGeom
  //
  edm::ESHandle<TrackerGeometry> pDD;
  iSetup.get<TrackerDigiGeometryRecord>().get(pDD);
  edm::LogInfo("TrackerMapTool") << " Geometry node for TrackerGeom is  " << &(*pDD);
  edm::LogInfo("TrackerMapTool") << " I have " << pDD->dets().size() << " detectors";
  edm::LogInfo("TrackerMapTool") << " I have " << pDD->detTypes().size() << " types";
  int spicchif[] = {24, 24, 40, 56, 40, 56, 80};
  int spicchib[] = {20, 32, 44, 30, 38, 46, 56, 42, 48, 54, 60, 66, 74};

  int nlay = 0;
  int layer, subdet, leftright = 0, ringno, petalno, moduleno, isStereo, pixel_strip, barrel_forward;
  std::string name0, name1, name2, name3, name4, name5;
  int ring = 0;
  int nmod = 0;
  int ntotmod = 0;
  float r;
  int bar_fow = 1;
  float phi, phi1;
  float rmedioS[] = {0.27665, 0.3671, 0.4474, 0.5617, 0.6768, 0.8189, 0.9907};
  float rmedioP[] = {0.0623081, 0.074111, 0.0870344, 0.103416, 0.115766, 0.132728, 0.140506};
  std::string nameDet;
  int last_layer = 0;
  float width, length, thickness, widthAtHalfLength;
  std::ofstream output("tracker.dat", std::ios::out);

  auto begin = pDD->detUnits().begin();
  auto end = pDD->detUnits().end();

  for (; begin != end; ++begin) {
    ntotmod++;
    subdet = (*begin)->geographicalId().subdetId();
    if (subdet == 1 || subdet == 3 || subdet == 5) {  //barrel
      layer = ((*begin)->geographicalId().rawId() >> 16) & 0xF;
      leftright = 0;
      name0 = " ";
    } else {  //endcap
      leftright = ((*begin)->geographicalId().rawId() >> 23) & 0x3;
      name0 = "+z";
      if (leftright == 1)
        name0 = "-z";
      layer = ((*begin)->geographicalId().rawId() >> 16) & 0xF;
    }
    isStereo = (*begin)->geographicalId().rawId() & 0x3;
    pixel_strip = 2;
    if (subdet <= 2)
      pixel_strip = 1;
    barrel_forward = 2;
    if (subdet == 2 || subdet == 4 || subdet == 6) {
      if (leftright == 1)
        barrel_forward = 1;
      else
        barrel_forward = 3;
    }
    nlay = layerno(subdet, leftright, layer);
    ringno = 0;
    petalno = 0;
    moduleno = 0;
    name1 = "   ";
    if (subdet == 1) {
      nameDet = "BPIX";
      name1 = " ladder ";
      ringno = ((*begin)->geographicalId().rawId() >> 8) & 0xFF;
    }
    if (subdet == 2) {
      nameDet = "FPIX";
      name1 = " ring ";
      ringno = ((*begin)->geographicalId().rawId() >> 8) & 0xFF;
    }
    if (subdet == 3) {
      nameDet = "TIB";
      name1 = " string ";
      ringno = ((*begin)->geographicalId().rawId() >> 8) & 0x3F;
    }
    if (subdet == 5) {
      nameDet = "TOB";
      name1 = " rod ";
      ringno = ((*begin)->geographicalId().rawId() >> 8) & 0x7F;
    }
    if (subdet == 4) {
      nameDet = "TID";
      name1 = " ring ";
      ringno = ((*begin)->geographicalId().rawId() >> 8) & 0xFF;
    }
    if (subdet == 6) {
      nameDet = "TEC";
      name1 = " ring ";
    }
    name2 = " ";
    if (subdet == 6) {
      name2 = " petal ";
      petalno = ((*begin)->geographicalId().rawId() >> 8) & 0x7F;
    }
    if (subdet == 2) {
      name2 = " blade ";
      petalno = ((*begin)->geographicalId().rawId() >> 8) & 0x3F;
    }
    name3 = " ";
    if (subdet == 6) {
      name3 = " forward ";
      if ((((*begin)->geographicalId().rawId() >> 4) & 0x1) == 1)
        name3 = " backward ";
    }
    if (subdet == 2) {
      name3 = " forward ";
      if ((((*begin)->geographicalId().rawId() >> 4) & 0x1) == 1)
        name3 = " backward ";
    }
    name5 = " ";
    if (subdet == 6) {
      name5 = " forward ";
      if ((((*begin)->geographicalId().rawId() >> 15) & 0x1) == 1) {
        name5 = " backward ";
      }
    }
    if (subdet == 2) {
      name5 = " left ";
      if ((((*begin)->geographicalId().rawId() >> 14) & 0x1) == 1) {
        name5 = " right ";
      }
    }
    if (subdet == 3) {
      name2 = " neg ";
      if ((((*begin)->geographicalId().rawId() >> 15) & 0x1) == 1)
        name2 = " pos ";
    }
    if (subdet == 5) {
      name2 = " neg ";
      if ((((*begin)->geographicalId().rawId() >> 15) & 0x1) == 1)
        name2 = " pos ";
    }
    if (subdet == 3) {
      name3 = " internal ";
      if ((((*begin)->geographicalId().rawId() >> 14) & 0x1) == 1)
        name3 = " external ";
    }
    if (subdet == 4) {
      name3 = " forward ";
      if ((((*begin)->geographicalId().rawId() >> 7) & 0x1) == 1)
        name3 = " backward ";
    }
    if (subdet == 1) {
      moduleno = ((*begin)->geographicalId().rawId() >> 2) & 0x3F;
    }
    if (subdet == 3) {
      moduleno = ((*begin)->geographicalId().rawId() >> 2) & 0x3F;
    }
    if (subdet == 5) {
      moduleno = ((*begin)->geographicalId().rawId() >> 2) & 0x3F;
    }
    if (subdet == 4) {
      moduleno = ((*begin)->geographicalId().rawId() >> 2) & 0x1F;
    }
    if (subdet == 6) {
      moduleno = ((*begin)->geographicalId().rawId() >> 2) & 0x3;
    }
    if (subdet == 2) {
      moduleno = ((*begin)->geographicalId().rawId() >> 2) & 0x7;
    }
    length = (*begin)->surface().bounds().length() / 100.0;        // cm -> m
    width = (*begin)->surface().bounds().width() / 100.0;          // cm -> mo
    thickness = (*begin)->surface().bounds().thickness() / 100.0;  // cm -> m
    widthAtHalfLength = (*begin)->surface().bounds().widthAtHalfLength() / 100.0;

    if (nlay != last_layer) {
      ring = -1;
      last_layer = nlay;
    }
    bar_fow = 2;
    if (nlay < 16)
      bar_fow = 1;
    if (nlay > 15 && nlay < 31)
      bar_fow = 3;
    float posx = (*begin)->surface().position().x() / 100.0;  // cm -> m
    float posy = (*begin)->surface().position().y() / 100.0;  // cm -> m
    float posz = (*begin)->surface().position().z() / 100.0;  // cm -> m
    r = pow(((double)(posx * posx) + posy * posy), 0.5);
    phi1 = atan(posy / posx);
    phi = phi1;
    if (posy < 0. && posx > 0)
      phi = phi1 + 2. * M_PI;
    if (posx < 0.)
      phi = phi1 + M_PI;
    if (fabs(posy) < 0.000001 && posx > 0)
      phi = 0;
    if (fabs(posy) < 0.000001 && posx < 0)
      phi = M_PI;
    if (fabs(posx) < 0.000001 && posy > 0)
      phi = M_PI / 2.;
    if (fabs(posx) < 0.000001 && posy < 0)
      phi = M_PI + M_PI / 2.;

    if (bar_fow == 2) {  //barrel
      if (subdet == 1)
        ring = moduleno;
      if (subdet == 5) {
        ring = moduleno;
        if ((((*begin)->geographicalId().rawId() >> 15) & 0x1) == 1)
          ring = ring + 6;
      }
      if (subdet == 3) {
        ring = moduleno;
        if (layer == 2 || layer == 4) {
          if ((((*begin)->geographicalId().rawId() >> 14) & 0x1) == 1) {
            ring = ring * 2;
          } else {
            ring = ring * 2 - 1;
          }
        } else {
          if ((((*begin)->geographicalId().rawId() >> 14) & 0x1) == 1) {
            ring = ring * 2 - 1;
          } else {
            ring = ring * 2;
          }
        }
        if ((((*begin)->geographicalId().rawId() >> 15) & 0x1) == 1)
          ring = ring + 6;
      }

      nmod = (int)((phi / (2. * M_PI)) * spicchib[nlay - 31] + .1) + 1;
      if (nlay == 40)
        nmod = nmod - 1;
      if (subdet == 1)
        nmod = ringno;
    } else {  // endcap
      if (subdet == 4 || subdet == 6) {
        for (int i = 0; i < 7; i++) {
          if (fabs(r - rmedioS[i]) < 0.015) {
            ring = i + 1;
            break;
          }
        }
        nmod = (int)((phi / (2. * M_PI)) * spicchif[ring - 1] + .1) + 1;
      } else {
        for (int i = 0; i < 7; i++) {
          if (fabs(r - rmedioP[i]) < 0.0001) {
            ring = i + 1;
            break;
          }
        }
        nmod = (int)((phi / (2. * M_PI)) * 24 + .1) + 1;
      }
    }  //end of endcap part
    if (isStereo == 1)
      nmod = nmod + 100;
    name4 = " ";
    if (isStereo == 1)
      name4 = " stereo";
    std::ostringstream outs;
    if (subdet == 6)
      outs << nameDet << " " << name0 << " layer " << layer << name1 << ringno << name2 << petalno << name5
           << " module  " << moduleno << name3 << " " << name4;
    if (subdet == 2)
      outs << nameDet << " " << name0 << " disc " << layer << name1 << ring << name2 << petalno << " plaquette  "
           << moduleno << name5;
    if (subdet == 1 || subdet == 3 || subdet == 4 || subdet == 5)
      outs << nameDet << " " << name0 << " layer " << layer << name1 << ringno << name2 << name3 << " module  "
           << moduleno << " " << name4;
    char buffer[20];
    sprintf(buffer, "%X", (*begin)->geographicalId().rawId());
    int id = (*begin)->geographicalId().rawId();
    output << ntotmod << " " << pixel_strip << " " << barrel_forward << " " << nlay << " " << ring << " " << nmod << " "
           << posx << " " << posy << " " << posz << " " << length << " " << width << " " << thickness << " "
           << widthAtHalfLength << " "
           << " " << id << std::endl;
    output << outs.str() << std::endl;
  }
}
//define this as a plug-in
DEFINE_FWK_MODULE(TrackerMapTool);
