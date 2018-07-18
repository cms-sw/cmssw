#ifndef __L1TMuon_TTGeometryTranslator_h__
#define __L1TMuon_TTGeometryTranslator_h__

//
// This class implements the translations from trigger primitive to
// global CMS coordinates for the Tracker trigger primitives analogous to
// the class GeometryTranslator.
//

#include "FWCore/Framework/interface/ESHandle.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include <memory>


// forwards
namespace edm {
  class EventSetup;
}

class TrackerGeometry;
class TrackerTopology;
class MagneticField;

namespace L1TMuon {

  class TTTriggerPrimitive;

  class TTGeometryTranslator {
  public:
    TTGeometryTranslator();
    ~TTGeometryTranslator();

    // Things you have to do to just get simple det id info...
    bool isBarrel  (const TTTriggerPrimitive&) const;
    bool isPSModule(const TTTriggerPrimitive&) const;
    int  region    (const TTTriggerPrimitive&) const;  // 0 for Barrel, +/-1 for +/- Endcap
    int  layer     (const TTTriggerPrimitive&) const;
    int  ring      (const TTTriggerPrimitive&) const;
    int  module    (const TTTriggerPrimitive&) const;

    // The translations
    double calculateGlobalEta(const TTTriggerPrimitive&) const;
    double calculateGlobalPhi(const TTTriggerPrimitive&) const;
    double calculateBendAngle(const TTTriggerPrimitive&) const;

    GlobalPoint getGlobalPoint(const TTTriggerPrimitive&) const;

    // Update geometry if necessary
    void checkAndUpdateGeometry(const edm::EventSetup&);

    // Retrieve the geometry records
    const TrackerGeometry& getTrackerGeometry() const { return *_geom; }
    const TrackerTopology& getTrackerTopology() const { return *_topo; }
    const MagneticField& getMagneticField() const { return *_magfield; }

  private:
    // Pointers to the current geometry records
    unsigned long long _geom_cache_id;
    edm::ESHandle<TrackerGeometry> _geom;

    unsigned long long _topo_cache_id;
    edm::ESHandle<TrackerTopology> _topo;

    unsigned long long _magfield_cache_id;
    edm::ESHandle<MagneticField> _magfield;

    GlobalPoint getTTSpecificPoint(const TTTriggerPrimitive&) const;
    double calcTTSpecificEta(const TTTriggerPrimitive&) const;
    double calcTTSpecificPhi(const TTTriggerPrimitive&) const;
    double calcTTSpecificBend(const TTTriggerPrimitive&) const;
  };
}

#endif
