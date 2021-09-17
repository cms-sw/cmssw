#ifndef GEOMETRY_FWTGEORECO_GEOMETRY_ES_PRODUCER_H
#define GEOMETRY_FWTGEORECO_GEOMETRY_ES_PRODUCER_H

#include <memory>

#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"

namespace edm {
  class ParameterSet;
}

class CaloGeometry;
class CaloGeometryRecord;
class GlobalTrackingGeometry;
class GlobalTrackingGeometryRecord;
class TrackerGeometry;
class TrackerTopology;
class TrackerTopologyRcd;
class FWTGeoRecoGeometry;
class FWTGeoRecoGeometryRecord;

class TGeoManager;
class TGeoShape;
class TGeoVolume;
class TGeoMaterial;
class TGeoMedium;
class GeomDet;
class CaloCellGeometry;
class FWTGeoRecoGeometryESProducer : public edm::ESProducer {
  enum ERecoDet {
    kDummy,
    kSiPixel,
    kSiStrip,
    kMuonDT,
    kMuonRPC,
    kMuonCSC,
    kMuonGEM,
    kMuonME0,
    kECal,
    kHCal,
    kCaloTower,
    kHGCE,
    kHGCH
  };

public:
  FWTGeoRecoGeometryESProducer(const edm::ParameterSet&);
  ~FWTGeoRecoGeometryESProducer(void) override;

  std::unique_ptr<FWTGeoRecoGeometry> produce(const FWTGeoRecoGeometryRecord&);

private:
  FWTGeoRecoGeometryESProducer(const FWTGeoRecoGeometryESProducer&);
  const FWTGeoRecoGeometryESProducer& operator=(const FWTGeoRecoGeometryESProducer&);

  TGeoManager* createManager(int level);
  TGeoShape* createShape(const GeomDet* det);
  TGeoVolume* createVolume(const std::string& name, const GeomDet* det, ERecoDet = kDummy);
  // TGeoMaterial*     createMaterial( const std::string& name );

  TGeoVolume* GetDaughter(TGeoVolume* mother, const char* prefix, ERecoDet cidx, int id);
  TGeoVolume* GetDaughter(TGeoVolume* mother, const char* prefix, ERecoDet cidx);
  TGeoVolume* GetTopHolder(const char* prefix, ERecoDet cidx);

  TGeoMedium* GetMedium(ERecoDet);

  void addPixelBarrelGeometry();
  void addPixelForwardGeometry();
  void addTIBGeometry();
  void addTOBGeometry();
  void addTIDGeometry();
  void addTECGeometry();
  void addCSCGeometry();
  void addDTGeometry();
  void addRPCGeometry();
  void addGEMGeometry();
  void addME0Geometry();
  void addEcalCaloGeometry();
  void addHcalCaloGeometryBarrel();
  void addHcalCaloGeometryEndcap();
  void addHcalCaloGeometryOuter();
  void addHcalCaloGeometryForward();
  void addCaloTowerGeometry();

  std::map<std::string, TGeoShape*> m_nameToShape;
  std::map<TGeoShape*, TGeoVolume*> m_shapeToVolume;
  std::map<ERecoDet, TGeoMedium*> m_recoMedium;

  edm::ESGetToken<GlobalTrackingGeometry, GlobalTrackingGeometryRecord> m_trackingGeomToken;
  edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> m_trackerTopologyToken;
  edm::ESGetToken<CaloGeometry, CaloGeometryRecord> m_caloGeomToken;

  const GlobalTrackingGeometry* m_trackingGeom = nullptr;
  const CaloGeometry* m_caloGeom = nullptr;
  const TrackerGeometry* m_trackerGeom = nullptr;
  const TrackerTopology* m_trackerTopology = nullptr;

  TGeoMedium* m_dummyMedium = nullptr;

  bool m_tracker;
  bool m_muon;
  bool m_calo;
};

#endif  // GEOMETRY_FWTGEORECO_GEOMETRY_ES_PRODUCER_H
