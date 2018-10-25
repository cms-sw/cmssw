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
#include <vector>
#include "DataFormats/L1Trigger/interface/BXVector.h"

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
    eeLayers_(0), fhLayers_(0), bhLayers_(0), totalLayers_(0){}
    ~HGCalTriggerTools() {}

    void eventSetup(const edm::EventSetup&);
    GlobalPoint getTCPosition(const DetId& id) const;
    unsigned layers(ForwardSubdetector type) const;
    unsigned layer(const DetId&) const;
    unsigned layerWithOffset(const DetId&) const;

    unsigned lastLayerEE() const {return eeLayers_;}
    unsigned lastLayerFH() const {return eeLayers_+fhLayers_;}
    unsigned lastLayerBH() const {return totalLayers_;}

    // 4-vector helper functions using GlobalPoint
    float getEta(const GlobalPoint& position, const float& vertex_z = 0.) const;
    float getPhi(const GlobalPoint& position) const;
    float getPt(const GlobalPoint& position, const float& hitEnergy, const float& vertex_z = 0.) const;

    // 4-vector helper functions using DetId
    float getTCEta(const DetId& id, const float& vertex_z = 0.) const;
    float getTCPhi(const DetId& id) const;
    float getTCPt(const DetId& id, const float& hitEnergy, const float& vertex_z = 0.) const;

    inline const HGCalTriggerGeometryBase * getTriggerGeometry() const {return geom_;};

    float getLayerZ(const unsigned& layerWithOffset) const;
    float getLayerZ(const int& subdet, const unsigned& layer) const;

    template<typename T> 
    std::vector<T> bxVectorToVector(const BXVector<T>& inputBXVector){    
      std::vector<T> outputVector;     
      //loop over collection for a given bx and put the objects into a std::vector
      for( typename std::vector<T>::const_iterator it = inputBXVector.begin(0) ; it != inputBXVector.end(0) ; ++it )
      { 
        outputVector.push_back(*it); 
      }
      return outputVector;
    }

  private:
    const HGCalTriggerGeometryBase* geom_;
    unsigned eeLayers_;
    unsigned fhLayers_;
    unsigned bhLayers_;
    unsigned totalLayers_;
};


#endif
