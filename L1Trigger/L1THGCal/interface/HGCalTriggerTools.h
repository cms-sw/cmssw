#ifndef __L1Trigger_L1THGCal_HGCalTriggerTools_h__
#define __L1Trigger_L1THGCal_HGCalTriggerTools_h__

/** \class HGCalTriggerTools
 *  Tools for handling HGCal trigger det-ID: in the current version
 *  of trhe HGCAL simulation only HGCalDetId for the TriggerCells (TC)
 *  are used and not HcalDetId as in the offline!
 *  As a consequence the class assumes that only DetIds of the first kind are used in the getTC* methods
 *  NOTE: this uses the trigger geometry hence would give wrong results
 *  when used for offline reco!!!!
 *
 *  \author G. Cerminara (CERN), heavily "inspired" by HGCalRechHitTools ;)
 */


#include <array>
#include <cmath>
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/ForwardDetId/interface/ForwardSubdetector.h"

class HGCalTriggerGeometryBase;
class DetId;





namespace edm {
  class Event;
  class EventSetup;
}

  class HGCalTriggerTools {
  public:
  HGCalTriggerTools() : geom_(nullptr),
                        fhOffset_(0),
                        bhOffset_(0) {}
    ~HGCalTriggerTools() {}

    void setEventSetup(const edm::EventSetup&);
    GlobalPoint getTCPosition(const DetId& id) const;
    unsigned int getLayerWithOffset(const DetId&) const;
    // unsigned int getLayer(ForwardSubdetector type) const;
    unsigned int getLayer(const DetId&) const;

    // 4-vector helper functions using GlobalPoint
    float getEta(const GlobalPoint& position, const float& vertex_z = 0.) const;
    float getPhi(const GlobalPoint& position) const;
    float getPt(const GlobalPoint& position, const float& hitEnergy, const float& vertex_z = 0.) const;

    // 4-vector helper functions using DetId
    float getTCEta(const DetId& id, const float& vertex_z = 0.) const;
    float getTCPhi(const DetId& id) const;
    float getTCPt(const DetId& id, const float& hitEnergy, const float& vertex_z = 0.) const;

    inline const HGCalTriggerGeometryBase * getTriggerGeometry() const {return geom_;};
    unsigned int lastLayerEE() const {return fhOffset_;}
    unsigned int lastLayerFH() const {return bhOffset_;}

    float getLayerZ(const unsigned& layerWithOffset) const;
    float getLayerZ(const int& subdet, const unsigned& layer) const;



  private:
    const HGCalTriggerGeometryBase* geom_;
    unsigned int        fhOffset_, bhOffset_;
  };


#endif
