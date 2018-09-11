#ifndef __RecoLocalCalo_HGCalRecAlgos_RecHitTools_h__
#define __RecoLocalCalo_HGCalRecAlgos_RecHitTools_h__

#include <array>
#include <cmath>
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/ForwardDetId/interface/ForwardSubdetector.h"

class CaloGeometry;
class CaloSubdetectorGeometry;
class DetId;

namespace edm {
  class Event;
  class EventSetup;
}

namespace hgcal {
  class RecHitTools {
  public:
    RecHitTools() : geom_(nullptr), fhOffset_(0), bhOffset_(0), fhLastLayer_(0), geometryType_(0) {}
    ~RecHitTools() {}

    void getEvent(const edm::Event&);
    void getEventSetup(const edm::EventSetup&);
    const CaloSubdetectorGeometry* getSubdetectorGeometry( const DetId& id ) const;

    GlobalPoint getPosition(const DetId& id) const;
    GlobalPoint getPositionLayer(int layer) const;
    // zside returns +/- 1
    int zside(const DetId& id) const;

    std::float_t getSiThickness(const DetId&) const;
    std::float_t getRadiusToSide(const DetId&) const;
    int getSiThickIndex(const DetId&) const;

    unsigned int getLayer(DetId::Detector type) const;
    unsigned int getLayer(ForwardSubdetector type) const;
    unsigned int getLayer(const DetId&) const;
    unsigned int getLayerWithOffset(const DetId&) const;
    std::pair<int,int> getWafer(const DetId&) const;
    std::pair<int,int> getCell(const DetId&) const;

    bool isHalfCell(const DetId&) const;

    // 4-vector helper functions using GlobalPoint
    float getEta(const GlobalPoint& position, const float& vertex_z = 0.) const;
    float getPhi(const GlobalPoint& position) const;
    float getPt(const GlobalPoint& position, const float& hitEnergy, const float& vertex_z = 0.) const;

    // 4-vector helper functions using DetId
    float getEta(const DetId& id, const float& vertex_z = 0.) const;
    float getPhi(const DetId& id) const;
    float getPt(const DetId& id, const float& hitEnergy, const float& vertex_z = 0.) const;

    inline const CaloGeometry * getGeometry() const {return geom_;};
    unsigned int lastLayerEE() const {return fhOffset_;}
    unsigned int lastLayerFH() const {return fhLastLayer_;}
    unsigned int maxNumberOfWafersPerLayer() const {return maxNumberOfWafersPerLayer_;}
    inline int getGeometryType() const {return geometryType_;}
  private:
    const CaloGeometry* geom_;
    unsigned int        fhOffset_, bhOffset_, fhLastLayer_, maxNumberOfWafersPerLayer_;
    int                 geometryType_;
  };
}

#endif
