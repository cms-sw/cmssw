#include "TVirtualViewer3D.h"
#include "TBuffer3D.h"
#include "TBuffer3DTypes.h"
#include "TVector3.h"

#include "TObject.h"
#include "TCanvas.h"
#include "TAtt3D.h"
#include "TGLViewer.h"

#include <vector>
#include <iostream>

class Chamber: public TObject {
   public:
      Chamber(Int_t rawid, Int_t subdetid, Int_t station, Int_t ring,
	      Double_t x, Double_t y, Double_t z,
	      TVector3 xhat, TVector3 yhat, TVector3 zhat);
      ~Chamber() {};

      Int_t rawid();
      Int_t subdetid();
      double x();
      double y();
      double z();
      void select(Bool_t selected = kTRUE);
      bool selected();

      TBuffer3D& GetBuffer3D(UInt_t reqSections);

   private:
      Int_t m_rawid, m_subdetid, m_station, m_ring;
      Double_t m_x, m_y, m_z;
      TVector3 m_xhat, m_yhat, m_zhat;
      Bool_t m_selected;

      ClassDef(Chamber, 0);
};

ClassImp(Chamber)

Chamber::Chamber(Int_t rawid, Int_t subdetid, Int_t station, Int_t ring,
		 Double_t x, Double_t y, Double_t z,
		 TVector3 xhat, TVector3 yhat, TVector3 zhat)
   : m_rawid(rawid), m_subdetid(subdetid), m_station(station), m_ring(ring)
   , m_x(x), m_y(y), m_z(z)
   , m_xhat(xhat), m_yhat(yhat), m_zhat(zhat)
   , m_selected(kFALSE) { }

Int_t Chamber::rawid() {
   return m_rawid;
}
Int_t Chamber::subdetid() {
   return m_subdetid;
}
double Chamber::x() {
   return m_x;
}
double Chamber::y() {
   return m_y;
}
double Chamber::z() {
   return m_z;
}

void Chamber::select(Bool_t selected) {
   m_selected = selected;
}

bool Chamber::selected() {
   return m_selected;
}

TBuffer3D& Chamber::GetBuffer3D(UInt_t reqSections) {
   static TBuffer3D buffer(TBuffer3DTypes::kGeneric);

      // Complete kCore section - this could be moved to Shape base class
   if (reqSections & TBuffer3D::kCore) {      
      buffer.ClearSectionsValid();
      buffer.fID = this; 
      buffer.fColor = (m_selected? 46: 38);       // Color index - see gROOT->GetColor()
      buffer.fTransparency = (m_selected? 0: 85);     // Transparency 0 (opaque) - 100 (fully transparent)
      buffer.fLocalFrame = kFALSE;
      buffer.SetLocalMasterIdentity();
      buffer.fReflection = kFALSE;
      buffer.SetSectionsValid(TBuffer3D::kCore);
   }
   // Complete kBoundingBox section
   if (reqSections & TBuffer3D::kBoundingBox) {
      Double_t origin[3] = { m_x, m_y, m_z };
      Double_t halfLength[3] =  { 100., 100., 100. };  // TODO: refine this
      buffer.SetAABoundingBox(origin, halfLength);
      buffer.SetSectionsValid(TBuffer3D::kBoundingBox);
   }
   // No kShapeSpecific section

   // Complete kRawSizes section
   if (reqSections & TBuffer3D::kRawSizes) {
      buffer.SetRawSizes(8, 3*8, 12, 3*12, 6, 6*6);
      buffer.SetSectionsValid(TBuffer3D::kRawSizes);
   }
   // Complete kRaw section
   if (reqSections & TBuffer3D::kRaw) {
      
      Double_t xthin, xwide, ylen, zlen;
      xthin = xwide = ylen = zlen = 0.;  // I don't like warnings

      if (m_subdetid == 1) {
	 if (m_station == 1) {  // I don't think the geometry depends on wheel (here called "m_ring"), but it might

	 }
	 else if (m_station == 2) {

	 }
	 else if (m_station == 3) {

	 }
	 else if (m_station == 4) {

	 }

	 xthin = 80.;  // TODO: should be real geometry, from XML
	 xwide = 80.;
	 ylen = 120.;
	 zlen = 10.;
      }
      else if (m_subdetid == 2) {
	 if (m_station == 1  &&  m_ring == 1) {   // endplug, slightly larger radius
	    ;
	 }
	 else if (m_station == 1  &&  m_ring == 4) {   // endplug, slightly smaller radius
	    ;
	 }
	 else if (m_station == 1  &&  m_ring == 2) {   // ME12
	    ;
	 }
	 else if (m_station == 1  &&  m_ring == 3) {   // ME13
	    ;
	 }
	 else if (m_station == 2  &&  m_ring == 1) {   // ME21
	    ;
	 }
	 else if (m_station == 2  &&  m_ring == 2) {   // ME22
	    ;
	 }
	 else if (m_station == 3  &&  m_ring == 1) {   // ME31
	    ;
	 }
	 else if (m_station == 3  &&  m_ring == 2) {   // ME32
	    ;
	 }
	 else if (m_station == 4  &&  m_ring == 1) {   // ME41
	    ;
	 }  // there is no ME42...

	 xthin = 40.;  // TODO: should be real geometry, from XML
	 xwide = 80.;
	 ylen = 100.;
	 zlen = 10.;
      }

      TVector3 corner000 = -m_xhat*xthin - m_yhat*ylen - m_zhat*zlen;
      TVector3 corner100 =  m_xhat*xthin - m_yhat*ylen - m_zhat*zlen;
      TVector3 corner110 =  m_xhat*xwide + m_yhat*ylen - m_zhat*zlen;
      TVector3 corner010 = -m_xhat*xwide + m_yhat*ylen - m_zhat*zlen;
      TVector3 corner001 = -m_xhat*xthin - m_yhat*ylen + m_zhat*zlen;
      TVector3 corner101 =  m_xhat*xthin - m_yhat*ylen + m_zhat*zlen;
      TVector3 corner111 =  m_xhat*xwide + m_yhat*ylen + m_zhat*zlen;
      TVector3 corner011 = -m_xhat*xwide + m_yhat*ylen + m_zhat*zlen;

      // Points (8)
      // 3 components: x,y,z
      buffer.fPnts[ 0] = m_x + corner000.x(); buffer.fPnts[ 1] = m_y + corner000.y(); buffer.fPnts[ 2] = m_z + corner000.z(); // 0
      buffer.fPnts[ 3] = m_x + corner100.x(); buffer.fPnts[ 4] = m_y + corner100.y(); buffer.fPnts[ 5] = m_z + corner100.z(); // 1
      buffer.fPnts[ 6] = m_x + corner110.x(); buffer.fPnts[ 7] = m_y + corner110.y(); buffer.fPnts[ 8] = m_z + corner110.z(); // 2
      buffer.fPnts[ 9] = m_x + corner010.x(); buffer.fPnts[10] = m_y + corner010.y(); buffer.fPnts[11] = m_z + corner010.z(); // 3
      buffer.fPnts[12] = m_x + corner001.x(); buffer.fPnts[13] = m_y + corner001.y(); buffer.fPnts[14] = m_z + corner001.z(); // 4
      buffer.fPnts[15] = m_x + corner101.x(); buffer.fPnts[16] = m_y + corner101.y(); buffer.fPnts[17] = m_z + corner101.z(); // 5
      buffer.fPnts[18] = m_x + corner111.x(); buffer.fPnts[19] = m_y + corner111.y(); buffer.fPnts[20] = m_z + corner111.z(); // 6
      buffer.fPnts[21] = m_x + corner011.x(); buffer.fPnts[22] = m_y + corner011.y(); buffer.fPnts[23] = m_z + corner011.z(); // 7

      // Segments (12)
      // 3 components: segment color(ignored), start point index, end point index
      // Indexes reference the above points
      buffer.fSegs[ 0] = 0   ; buffer.fSegs[ 1] = 0   ; buffer.fSegs[ 2] = 1   ; // 0
      buffer.fSegs[ 3] = 0   ; buffer.fSegs[ 4] = 1   ; buffer.fSegs[ 5] = 2   ; // 1
      buffer.fSegs[ 6] = 0   ; buffer.fSegs[ 7] = 2   ; buffer.fSegs[ 8] = 3   ; // 2
      buffer.fSegs[ 9] = 0   ; buffer.fSegs[10] = 3   ; buffer.fSegs[11] = 0   ; // 3
      buffer.fSegs[12] = 0   ; buffer.fSegs[13] = 4   ; buffer.fSegs[14] = 5   ; // 4
      buffer.fSegs[15] = 0   ; buffer.fSegs[16] = 5   ; buffer.fSegs[17] = 6   ; // 5
      buffer.fSegs[18] = 0   ; buffer.fSegs[19] = 6   ; buffer.fSegs[20] = 7   ; // 6
      buffer.fSegs[21] = 0   ; buffer.fSegs[22] = 7   ; buffer.fSegs[23] = 4   ; // 7
      buffer.fSegs[24] = 0   ; buffer.fSegs[25] = 0   ; buffer.fSegs[26] = 4   ; // 8
      buffer.fSegs[27] = 0   ; buffer.fSegs[28] = 1   ; buffer.fSegs[29] = 5   ; // 9
      buffer.fSegs[30] = 0   ; buffer.fSegs[31] = 2   ; buffer.fSegs[32] = 6   ; // 10
      buffer.fSegs[33] = 0   ; buffer.fSegs[34] = 3   ; buffer.fSegs[35] = 7   ; // 11
      
      // Polygons (6)
      // 5+ (2+n) components: polygon color (ignored), segment count(n=3+),
      // seg1, seg2 .... segn index
      // Segments indexes refer to the above 12 segments
      // Here n=4 - each polygon defines a rectangle - 4 sides.
      buffer.fPols[ 0] = 0   ; buffer.fPols[ 1] = 4   ;  buffer.fPols[ 2] = 8  ; // 0
      buffer.fPols[ 3] = 4   ; buffer.fPols[ 4] = 9   ;  buffer.fPols[ 5] = 0  ;
      buffer.fPols[ 6] = 0   ; buffer.fPols[ 7] = 4   ;  buffer.fPols[ 8] = 9  ; // 1
      buffer.fPols[ 9] = 5   ; buffer.fPols[10] = 10  ;  buffer.fPols[11] = 1  ;
      buffer.fPols[12] = 0   ; buffer.fPols[13] = 4   ;  buffer.fPols[14] = 10  ; // 2
      buffer.fPols[15] = 6   ; buffer.fPols[16] = 11  ;  buffer.fPols[17] = 2  ;
      buffer.fPols[18] = 0   ; buffer.fPols[19] = 4   ;  buffer.fPols[20] = 11 ; // 3
      buffer.fPols[21] = 7   ; buffer.fPols[22] = 8   ;  buffer.fPols[23] = 3 ;
      buffer.fPols[24] = 0   ; buffer.fPols[25] = 4   ;  buffer.fPols[26] = 1  ; // 4
      buffer.fPols[27] = 2   ; buffer.fPols[28] = 3   ;  buffer.fPols[29] = 0  ;
      buffer.fPols[30] = 0   ; buffer.fPols[31] = 4   ;  buffer.fPols[32] = 7  ; // 5
      buffer.fPols[33] = 6   ; buffer.fPols[34] = 5   ;  buffer.fPols[35] = 4  ;
      
      buffer.SetSectionsValid(TBuffer3D::kRaw);
  }

   return buffer;
}

class MyGeom: public TObject, public TAtt3D {
   public:
      MyGeom();
      ~MyGeom();
      void add(Chamber *c);
      void select(Int_t rawid, Bool_t selected = kTRUE);

      void Draw(Option_t *option);
//      void ReDraw();
      void Paint(Option_t *option);

   private:
      std::vector<Chamber*> m_chambers;
      TVirtualViewer3D *m_viewer;

      ClassDef(MyGeom, 0);
};

ClassImp(MyGeom)

MyGeom::MyGeom() {}
MyGeom::~MyGeom() {}
void MyGeom::add(Chamber *c) {
   m_chambers.push_back(c);
}
void MyGeom::select(Int_t rawid, Bool_t selected) {
   for (std::vector<Chamber*>::const_iterator it = m_chambers.begin();
	it != m_chambers.end();
	++it) {
      if ((*it)->rawid() == rawid) {
	 (*it)->select(selected);
      }
   }
}

void MyGeom::Draw(Option_t* option) {
   TObject::Draw(option);
   m_viewer = gPad->GetViewer3D(option);
}

// void MyGeom::ReDraw() {
//    if (m_viewer->InheritsFrom("TGLViewer")) {
//       TGLViewer *v = (TGLViewer*)(m_viewer);
//       v->DoDraw();
//    }
//    else {
//       std::cerr << "ReDraw() is only implemented for TGLViewer." << std::endl;
//    }
// }

void MyGeom::Paint(Option_t*) {
   TVirtualViewer3D * viewer = gPad->GetViewer3D();

   // If MyGeom derives from TAtt3D then pad will recognise
   // that the object it is asking to paint is 3D, and open/close
   // the scene for us. If not Open/Close are required
   //viewer->BeginScene();

   // We are working in the master frame - so we don't bother
   // to ask the viewer if it prefers local. Viewer's must
   // always support master frame as minimum. c.f. with
   // viewer3DLocal.C

   std::vector<Chamber*>::const_iterator ShapeIt = m_chambers.begin();
   Chamber* shape;
   while (ShapeIt != m_chambers.end()) {
      shape = *ShapeIt;
      UInt_t reqSections = TBuffer3D::kCore|TBuffer3D::kBoundingBox|TBuffer3D::kShapeSpecific;
      TBuffer3D & buffer = shape->GetBuffer3D(reqSections);
      reqSections = viewer->AddObject(buffer);

      if (reqSections != TBuffer3D::kNone) {
         shape->GetBuffer3D(reqSections);
         viewer->AddObject(buffer);
      }
      ShapeIt++;
   }
   // Not required as we are TAtt3D subclass
   //viewer->EndScene();
}

void viewer3d() {}
