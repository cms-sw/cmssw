#ifndef __L1TMUON_GEOMETRYTRANSLATOR_H__
#define __L1TMUON_GEOMETRYTRANSLATOR_H__
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

#include "FWCore/Framework/interface/ESHandle.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include <memory>


// forwards
namespace edm {
  class EventSetup;
}

class GEMGeometry;
class RPCGeometry;
class CSCGeometry;
class CSCLayer;
class DTGeometry;
class MagneticField;

namespace L1TMuon {
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

    const GEMGeometry& getGEMGeometry() const { return *_geogem; }
    const RPCGeometry& getRPCGeometry() const { return *_georpc; }
    const CSCGeometry& getCSCGeometry() const { return *_geocsc; }
    const DTGeometry&  getDTGeometry()  const { return *_geodt;  }

    const MagneticField& getMagneticField() const { return *_magfield; }

  private:
    // pointers to the current geometry records
    unsigned long long _geom_cache_id;
    edm::ESHandle<GEMGeometry> _geogem;
    edm::ESHandle<RPCGeometry> _georpc;
    edm::ESHandle<CSCGeometry> _geocsc;
    edm::ESHandle<DTGeometry>  _geodt;

    unsigned long long _magfield_cache_id;
    edm::ESHandle<MagneticField> _magfield;

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
}

#endif
