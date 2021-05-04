#ifndef __L1TMuon_GeometryTranslator_h__
#define __L1TMuon_GeometryTranslator_h__
//
// Class: L1TMuon::GeometryTranslator
//
// Info: This class implements a the translations from packed bits or
//       digi information into local or global CMS coordinates for all
//       types of L1 trigger primitives that we want to consider for
//       use in the integrated muon trigger.
//
// Note: This should be considered as a base class to some sort of global
//       look-up table
//
// Author: L. Gray (FNAL)
// Some pieces of code lifted from: Matt Carver & Bobby Scurlock (UF)
//

#include <memory>

#include "FWCore/Framework/interface/ESHandle.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"

// Forward declarations
namespace edm {
  class EventSetup;
}

class DTGeometry;
class CSCGeometry;
class CSCLayer;
class RPCGeometry;
class GEMGeometry;
class ME0Geometry;
class MagneticField;

namespace L1TMuon {

  // Forward declaration
  class TriggerPrimitive;

  class GeometryTranslator {
  public:
    GeometryTranslator();
    ~GeometryTranslator();

    double calculateGlobalEta(const TriggerPrimitive&) const;
    double calculateGlobalPhi(const TriggerPrimitive&) const;
    double calculateBendAngle(const TriggerPrimitive&) const;

    GlobalPoint getGlobalPoint(const TriggerPrimitive&) const;

    void checkAndUpdateGeometry(const edm::EventSetup&);

    const DTGeometry& getDTGeometry() const { return *_geodt; }
    const CSCGeometry& getCSCGeometry() const { return *_geocsc; }
    const RPCGeometry& getRPCGeometry() const { return *_georpc; }
    const GEMGeometry& getGEMGeometry() const { return *_geogem; }
    const ME0Geometry& getME0Geometry() const { return *_geome0; }

    const MagneticField& getMagneticField() const { return *_magfield; }

  private:
    unsigned long long _geom_cache_id;
    edm::ESHandle<DTGeometry> _geodt;
    edm::ESHandle<CSCGeometry> _geocsc;
    edm::ESHandle<RPCGeometry> _georpc;
    edm::ESHandle<GEMGeometry> _geogem;
    edm::ESHandle<ME0Geometry> _geome0;

    unsigned long long _magfield_cache_id;
    edm::ESHandle<MagneticField> _magfield;

    GlobalPoint getME0SpecificPoint(const TriggerPrimitive&) const;
    double calcME0SpecificEta(const TriggerPrimitive&) const;
    double calcME0SpecificPhi(const TriggerPrimitive&) const;
    double calcME0SpecificBend(const TriggerPrimitive&) const;

    GlobalPoint getGEMSpecificPoint(const TriggerPrimitive&) const;
    double calcGEMSpecificEta(const TriggerPrimitive&) const;
    double calcGEMSpecificPhi(const TriggerPrimitive&) const;
    double calcGEMSpecificBend(const TriggerPrimitive&) const;

    GlobalPoint getRPCSpecificPoint(const TriggerPrimitive&) const;
    double calcRPCSpecificEta(const TriggerPrimitive&) const;
    double calcRPCSpecificPhi(const TriggerPrimitive&) const;
    double calcRPCSpecificBend(const TriggerPrimitive&) const;

    GlobalPoint getCSCSpecificPoint(const TriggerPrimitive&) const;
    double calcCSCSpecificEta(const TriggerPrimitive&) const;
    double calcCSCSpecificPhi(const TriggerPrimitive&) const;
    double calcCSCSpecificBend(const TriggerPrimitive&) const;
    bool isCSCCounterClockwise(const std::unique_ptr<const CSCLayer>&) const;

    GlobalPoint calcDTSpecificPoint(const TriggerPrimitive&) const;
    double calcDTSpecificEta(const TriggerPrimitive&) const;
    double calcDTSpecificPhi(const TriggerPrimitive&) const;
    double calcDTSpecificBend(const TriggerPrimitive&) const;
  };

}  // namespace L1TMuon

#endif
