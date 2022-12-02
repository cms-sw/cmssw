#ifndef GEOMETRY_FWRECO_GEOMETRY_ES_PRODUCER_H
#define GEOMETRY_FWRECO_GEOMETRY_ES_PRODUCER_H

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
class IdealGeometryRecord;
class MTDGeometry;
class MTDDigiGeometryRecord;
class FWRecoGeometry;
class FWRecoGeometryRecord;
class GeomDet;

class FWRecoGeometryESProducer : public edm::ESProducer {
public:
  FWRecoGeometryESProducer(const edm::ParameterSet&);
  ~FWRecoGeometryESProducer(void) override;

  std::unique_ptr<FWRecoGeometry> produce(const FWRecoGeometryRecord&);

  FWRecoGeometryESProducer(const FWRecoGeometryESProducer&) = delete;
  const FWRecoGeometryESProducer& operator=(const FWRecoGeometryESProducer&) = delete;

private:
  void addCSCGeometry(FWRecoGeometry&);
  void addDTGeometry(FWRecoGeometry&);
  void addRPCGeometry(FWRecoGeometry&);
  void addGEMGeometry(FWRecoGeometry&);
  void addME0Geometry(FWRecoGeometry&);
  void addPixelBarrelGeometry(FWRecoGeometry&);
  void addPixelForwardGeometry(FWRecoGeometry&);
  void addTIBGeometry(FWRecoGeometry&);
  void addTOBGeometry(FWRecoGeometry&);
  void addTIDGeometry(FWRecoGeometry&);
  void addTECGeometry(FWRecoGeometry&);
  void addCaloGeometry(FWRecoGeometry&);

  void addMTDGeometry(FWRecoGeometry&);

  void ADD_PIXEL_TOPOLOGY(unsigned int rawid, const GeomDet* detUnit, FWRecoGeometry&);

  void ADD_MTD_TOPOLOGY(unsigned int rawid, const GeomDet* detUnit, FWRecoGeometry&);

  unsigned int insert_id(unsigned int id, FWRecoGeometry&);
  void fillPoints(unsigned int id,
                  std::vector<GlobalPoint>::const_iterator begin,
                  std::vector<GlobalPoint>::const_iterator end,
                  FWRecoGeometry&);
  void fillShapeAndPlacement(unsigned int id, const GeomDet* det, FWRecoGeometry&);
  void writeTrackerParametersXML(FWRecoGeometry&);

  edm::ESGetToken<GlobalTrackingGeometry, GlobalTrackingGeometryRecord> m_trackingGeomToken;
  edm::ESGetToken<MTDGeometry, MTDDigiGeometryRecord> m_mtdGeomToken;
  edm::ESGetToken<CaloGeometry, CaloGeometryRecord> m_caloGeomToken;
  const GlobalTrackingGeometry* m_trackingGeom = nullptr;
  const MTDGeometry* m_mtdGeom = nullptr;
  const CaloGeometry* m_caloGeom = nullptr;
  const TrackerGeometry* m_trackerGeom = nullptr;

  unsigned int m_current;
  bool m_tracker;
  bool m_muon;
  bool m_gem;
  bool m_calo;
  bool m_timing;
};

#endif  // GEOMETRY_FWRECO_GEOMETRY_ES_PRODUCER_H
