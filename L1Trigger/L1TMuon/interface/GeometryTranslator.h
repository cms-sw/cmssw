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

class RPCGeometry;
class CSCGeometry;
class CSCLayer;
class DTGeometry;

namespace l1t {

  class MuonTriggerPrimitive;

  class GeometryTranslator {
  public:
    GeometryTranslator();
    ~GeometryTranslator();

    double calculateGlobalEta(const MuonTriggerPrimitive&) const;
    double calculateGlobalPhi(const MuonTriggerPrimitive&) const;
    double calculateBendAngle(const MuonTriggerPrimitive&) const;    

    void checkAndUpdateGeometry(const edm::EventSetup&);

  private:
    // pointers to the current geometry records
    unsigned long long _geom_cache_id;
    edm::ESHandle<RPCGeometry> _georpc;    
    edm::ESHandle<CSCGeometry> _geocsc;    
    edm::ESHandle<DTGeometry>  _geodt;    
    
    GlobalPoint getRPCSpecificPoint(const MuonTriggerPrimitive&) const;
    double calcRPCSpecificEta(const MuonTriggerPrimitive&) const;
    double calcRPCSpecificPhi(const MuonTriggerPrimitive&) const;
    double calcRPCSpecificBend(const MuonTriggerPrimitive&) const;

    GlobalPoint getCSCSpecificPoint(const MuonTriggerPrimitive&) const;
    double calcCSCSpecificEta(const MuonTriggerPrimitive&) const;
    double calcCSCSpecificPhi(const MuonTriggerPrimitive&) const;
    double calcCSCSpecificBend(const MuonTriggerPrimitive&) const;
    bool isCSCCounterClockwise(const std::unique_ptr<const CSCLayer>&) const;

    GlobalPoint calcDTSpecificPoint(const MuonTriggerPrimitive&) const;
    double calcDTSpecificEta(const MuonTriggerPrimitive&) const;
    double calcDTSpecificPhi(const MuonTriggerPrimitive&) const;
    double calcDTSpecificBend(const MuonTriggerPrimitive&) const;
  };
}

#endif
