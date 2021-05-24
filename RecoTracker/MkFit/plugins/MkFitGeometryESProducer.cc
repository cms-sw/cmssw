#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"

#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"
#include "RecoTracker/TkDetLayers/interface/GeometricSearchTracker.h"
#include "RecoTracker/Record/interface/TrackerRecoGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"

#include "RecoTracker/MkFit/interface/MkFitGeometry.h"

#include "createPhase1TrackerGeometry.h"

// mkFit includes
#include "RecoTracker/MkFitCore/interface/ConfigWrapper.h"
#include "RecoTracker/MkFitCore/interface/TrackerInfo.h"
#include "RecoTracker/MkFitCore/interface/IterationConfig.h"
#include "RecoTracker/MkFitCMS/interface/LayerNumberConverter.h"

#include <atomic>

//------------------------------------------------------------------------------

#include "DataFormats/GeometrySurface/interface/RectangularPlaneBounds.h"
#include "DataFormats/GeometrySurface/interface/TrapezoidalPlaneBounds.h"
//#include "Geometry/CommonDetUnit/interface/GlobalTrackingGeometry.h"
//#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
//#include "Geometry/TrackerGeometryBuilder/interface/StripGeomDetUnit.h"
//#include "Geometry/TrackerGeometryBuilder/interface/StripGeomDetType.h"
//#include "Geometry/TrackerGeometryBuilder/interface/RectangularPixelTopology.h"
#include <fstream>

//------------------------------------------------------------------------------


class MkFitGeometryESProducer : public edm::ESProducer {
public:
  MkFitGeometryESProducer(const edm::ParameterSet& iConfig);

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  void considerPoint(const GlobalPoint &gp, mkfit::LayerInfo &lay_info);
  void fillShapeAndPlacement(const GeomDet *det, mkfit::TrackerInfo &trk_info);
  void addPixBGeometry(mkfit::TrackerInfo &trk_info);
  void addPixEGeometry(mkfit::TrackerInfo &trk_info);
  void addTIBGeometry(mkfit::TrackerInfo &trk_info);
  void addTOBGeometry(mkfit::TrackerInfo &trk_info);
  void addTIDGeometry(mkfit::TrackerInfo &trk_info);
  void addTECGeometry(mkfit::TrackerInfo &trk_info);

  std::unique_ptr<MkFitGeometry> produce(const TrackerRecoGeometryRecord& iRecord);

private:
  edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> geomToken_;
  // QQQQ edm::ESGetToken<TrackerGeometry, TrackerRecoGeometryRecord> geomTokenReco_;
  edm::ESGetToken<GeometricSearchTracker, TrackerRecoGeometryRecord> trackerToken_;
  edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> ttopoToken_;

  const TrackerTopology *m_trackerTopo = nullptr;
  const TrackerGeometry *m_trackerGeom = nullptr;
  mkfit::LayerNumberConverter   m_layerNrConv = { mkfit::TkLayout::phase1 };
};

MkFitGeometryESProducer::MkFitGeometryESProducer(const edm::ParameterSet& iConfig) {
  auto cc = setWhatProduced(this);
  geomToken_ = cc.consumes();
  // QQQQ geomTokenReco_ = cc.consumes();
  trackerToken_ = cc.consumes();
  ttopoToken_ = cc.consumes();
}

void MkFitGeometryESProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  descriptions.addWithDefaultLabel(desc);
}

//------------------------------------------------------------------------------

void MkFitGeometryESProducer::considerPoint(const GlobalPoint &gp, mkfit::LayerInfo &li)
{
  float r = gp.perp(), z = gp.z();
  // need this is a function in LayerInfo
  // float  m_rin, m_rout, m_zmin, m_zmax;
  if (z > li.m_zmax) li.m_zmax = z;
  if (z < li.m_zmin) li.m_zmin = z;
  if (r > li.m_rout) li.m_rout = r;
  if (r < li.m_rin)  li.m_rin  = r;
}

void MkFitGeometryESProducer::fillShapeAndPlacement(const GeomDet* det, mkfit::TrackerInfo &trk_info)
{
  DetId detid = det->geographicalId();
  
  float xy[4][2];
  float dz;
  const Bounds* b = &((det->surface()).bounds());

  if (const TrapezoidalPlaneBounds* b2 = dynamic_cast<const TrapezoidalPlaneBounds*>(b)) {
    // Trapezoidal
    std::array<const float, 4> const& par = b2->parameters();

    // These parameters are half-lengths, as in CMSIM/GEANT3
    // https://github.com/trackreco/cmssw/blob/master/Fireworks/Core/src/FWGeometry.cc#L241
    // https://github.com/root-project/root/blob/master/geom/geom/src/TGeoArb8.cxx#L1331
    /*
    i.shape[0] = 1;
    i.shape[1] = par[0];  // hBottomEdge - dx1
    i.shape[2] = par[1];  // hTopEdge    - dx2
    i.shape[3] = par[2];  // thickness   - dz
    i.shape[4] = par[3];  // apothem     - dy1

          geoShape = new TGeoTrap(info.shape[3],  //dz
                            0,              //theta
                            0,              //phi
                            info.shape[4],  //dy1
                            info.shape[1],  //dx1
                            info.shape[2],  //dx2
                            0,              //alpha1
                            info.shape[4],  //dy2
                            info.shape[1],  //dx3
                            info.shape[2],  //dx4
                            0);             //alpha2

      TGeoTrap::TGeoTrap(Double_t dz, Double_t theta, Double_t phi, Double_t h1,
               Double_t bl1, Double_t tl1, Double_t alpha1, Double_t h2, Double_t bl2,
               Double_t tl2, Double_t alpha2)
          :TGeoArb8("", 0, 0)
    {
       fDz = dz;           par[2]
       fTheta = theta;     = 0
       fPhi = phi;         = 0
       fH1 = h1;           par[3]
       fH2 = h2;           par[3]
       fBl1 = bl1;         par[0]
       fBl2 = bl2;         par[0]
       fTl1 = tl1;         par[1]
       fTl2 = tl2;         par[1]
       fAlpha1 = alpha1;   = 0
       fAlpha2 = alpha2;   = 0
       Double_t tx = TMath::Tan(theta*TMath::DegToRad())*TMath::Cos(phi*TMath::DegToRad()); = 0
       Double_t ty = TMath::Tan(theta*TMath::DegToRad())*TMath::Sin(phi*TMath::DegToRad()); = 0
       Double_t ta1 = TMath::Tan(alpha1*TMath::DegToRad());  = 0
       Double_t ta2 = TMath::Tan(alpha2*TMath::DegToRad());  = 0
       fXY[0][0] = -bl1;    fXY[0][1] = -h1;  -dz
       fXY[1][0] = -tl1;    fXY[1][1] =  h1;  -dz
       fXY[2][0] =  tl1;    fXY[2][1] =  h1;  -dz
       fXY[3][0] =  bl1;    fXY[3][1] = -h1;  -dz
       fXY[4][0] = -bl2;    fXY[4][1] = -h2;   dz
       fXY[5][0] = -tl2;    fXY[5][1] =  h2;   dz
       fXY[6][0] =  tl2;    fXY[6][1] =  h2;   dz
       fXY[7][0] =  bl2;    fXY[7][1] = -h2;   dz
     }
     */
    xy[0][0] = -par[0]; xy[0][1] = -par[3];
    xy[1][0] = -par[1]; xy[1][1] =  par[3];
    xy[2][0] =  par[1]; xy[2][1] =  par[3];
    xy[3][0] =  par[0]; xy[3][1] = -par[3];
    dz = par[2];
    printf("TRAP 0x%x %f %f %f %f\n", detid.rawId(), par[0], par[1], par[2], par[3]);
  }
  else if (const RectangularPlaneBounds* b2 = dynamic_cast<const RectangularPlaneBounds*>(b)) {
    // Rectangular
    float dx = b2->width() * 0.5;      // half width
    float dy = b2->length() * 0.5;     // half length
    xy[0][0] = -dx; xy[0][1] = -dy;
    xy[1][0] = -dx; xy[1][1] =  dy;
    xy[2][0] =  dx; xy[2][1] =  dy;
    xy[3][0] =  dx; xy[3][1] = -dy;
    dz = b2->thickness() * 0.5;  // half thickness
    printf("RECT 0x%x %f %f %f\n", detid.rawId(), dx, dy, dz);
  }

  const bool useMatched = false;
  int lay = m_layerNrConv.convertLayerNumber(detid.det(), m_trackerTopo->layer(detid), useMatched,
                                             m_trackerTopo->isStereo(detid),
                                             m_trackerTopo->side(detid) == 1);

  mkfit::LayerInfo &layer_info = trk_info.m_layers[lay];
  for (int i = 0; i < 4; ++i)
  {
    Local3DPoint lp1(xy[i][0], xy[i][1], -dz);
    Local3DPoint lp2(xy[i][0], xy[i][1],  dz);

    considerPoint(det->surface().toGlobal( lp1 ), layer_info);
    considerPoint(det->surface().toGlobal( lp2 ), layer_info);
  }
}

//==============================================================================

void MkFitGeometryESProducer::addPixBGeometry(mkfit::TrackerInfo &trk_info) {
  for (TrackerGeometry::DetContainer::const_iterator it = m_trackerGeom->detsPXB().begin(),
                                                     end = m_trackerGeom->detsPXB().end();
       it != end;
       ++it) {
    const GeomDet* det = *it;

    if (det) {
      //DetId detid = det->geographicalId();
      //unsigned int rawid = detid.rawId();
      fillShapeAndPlacement(det, trk_info);

      //ADD_PIXEL_TOPOLOGY(current, m_trackerGeom->idToDetUnit(detid), fwRecoGeometry);
    }
  }
}

void MkFitGeometryESProducer::addPixEGeometry(mkfit::TrackerInfo &trk_info) {
  for (TrackerGeometry::DetContainer::const_iterator it = m_trackerGeom->detsPXF().begin(),
                                                     end = m_trackerGeom->detsPXF().end();
       it != end;
       ++it) {
    const GeomDet* det = *it;

    if (det) {
      //DetId detid = det->geographicalId();
      //unsigned int rawid = detid.rawId();
      fillShapeAndPlacement(det, trk_info);

      //ADD_PIXEL_TOPOLOGY(current, m_trackerGeom->idToDetUnit(detid), fwRecoGeometry);
    }
  }
}

void MkFitGeometryESProducer::addTIBGeometry(mkfit::TrackerInfo &trk_info) {
  for (TrackerGeometry::DetContainer::const_iterator it = m_trackerGeom->detsTIB().begin(),
                                                     end = m_trackerGeom->detsTIB().end();
       it != end;
       ++it) {
    const GeomDet* det = *it;

    if (det) {
      //DetId detid = det->geographicalId();
      //unsigned int rawid = detid.rawId();
      fillShapeAndPlacement(det, trk_info);

      //ADD_SISTRIP_TOPOLOGY(current, m_trackerGeom->idToDet(detid));
    }
  }
}

void MkFitGeometryESProducer::addTOBGeometry(mkfit::TrackerInfo &trk_info) {
  for (TrackerGeometry::DetContainer::const_iterator it = m_trackerGeom->detsTOB().begin(),
                                                     end = m_trackerGeom->detsTOB().end();
       it != end;
       ++it) {
    const GeomDet* det = *it;

    if (det) {
      //DetId detid = det->geographicalId();
      //unsigned int rawid = detid.rawId();
      fillShapeAndPlacement(det, trk_info);

      //ADD_SISTRIP_TOPOLOGY(current, m_trackerGeom->idToDet(detid));
    }
  }
}

void MkFitGeometryESProducer::addTIDGeometry(mkfit::TrackerInfo &trk_info) {
  for (TrackerGeometry::DetContainer::const_iterator it = m_trackerGeom->detsTID().begin(),
                                                     end = m_trackerGeom->detsTID().end();
       it != end;
       ++it) {
    const GeomDet* det = *it;

    if (det) {
      //DetId detid = det->geographicalId();
      //unsigned int rawid = detid.rawId();
      fillShapeAndPlacement(det, trk_info);

      //ADD_SISTRIP_TOPOLOGY(current, m_trackerGeom->idToDet(detid));
    }
  }
}

void MkFitGeometryESProducer::addTECGeometry(mkfit::TrackerInfo &trk_info) {
  for (TrackerGeometry::DetContainer::const_iterator it = m_trackerGeom->detsTEC().begin(),
                                                     end = m_trackerGeom->detsTEC().end();
       it != end;
       ++it) {
    const GeomDet* det = *it;

    if (det) {
      //DetId detid = det->geographicalId();
      //unsigned int rawid = detid.rawId();
      fillShapeAndPlacement(det, trk_info);

      //ADD_SISTRIP_TOPOLOGY(current, m_trackerGeom->idToDet(detid));
    }
  }
}

//------------------------------------------------------------------------------

std::unique_ptr<MkFitGeometry> MkFitGeometryESProducer::produce(const TrackerRecoGeometryRecord& iRecord) {
  auto trackerInfo = std::make_unique<mkfit::TrackerInfo>();

  // QQQQ m_trackerGeom = &iRecord.get(geomTokenReco_);
  m_trackerGeom = &iRecord.get(geomToken_);

  m_trackerTopo = &iRecord.get(ttopoToken_);

  // std::string path = "Geometry/TrackerCommonData/data/";
  if (m_trackerGeom->isThere(GeomDetEnumerators::P1PXB) || m_trackerGeom->isThere(GeomDetEnumerators::P1PXEC)) {
    // path += "PhaseI/";
    std::cout << "-- PhaseI --\n";
    trackerInfo->create_layers(18, 27, 27);
  } else if (m_trackerGeom->isThere(GeomDetEnumerators::P2PXB) || m_trackerGeom->isThere(GeomDetEnumerators::P2PXEC) ||
             m_trackerGeom->isThere(GeomDetEnumerators::P2OTB) || m_trackerGeom->isThere(GeomDetEnumerators::P2OTEC)) {
    // path += "PhaseII/";
    std::cout << "-- PhaseII --\n";
  } else {
    std::cout << "-- Phase Naught --\n";
  }

  // Prepare layer boundaries for bounding-box search
  for (auto &li : trackerInfo->m_layers)
    li.set_limits(1e9, 0, 1e9, -1e9);

  // This is sort of CMS-2017 specific ... but fireworks code uses it for PhaseII as well
  // split in Fireworks, could really iterate over trackerGeometry->dets()
  addPixBGeometry(*trackerInfo);
  addPixEGeometry(*trackerInfo);
  addTIBGeometry(*trackerInfo);
  addTIDGeometry(*trackerInfo);
  addTOBGeometry(*trackerInfo);
  addTECGeometry(*trackerInfo);

  // missing setup of bins ets (as in standalone Geoms/CMS-2017.cc and mk_trk_info.C)

  return std::make_unique<MkFitGeometry>(
    iRecord.get(geomToken_), iRecord.get(trackerToken_), iRecord.get(ttopoToken_), std::move(trackerInfo));
}

DEFINE_FWK_EVENTSETUP_MODULE(MkFitGeometryESProducer);
