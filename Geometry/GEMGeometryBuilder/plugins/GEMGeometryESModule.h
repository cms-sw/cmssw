#ifndef Geometry_GEMGeometry_GEMGeometryESModule_h
#define Geometry_GEMGeometry_GEMGeometryESModule_h

/** \class GEMGeometryESModule
 * 
 *  ESProducer for GEMGeometry in MuonGeometryRecord
 *
 *  \author M. Maggi - INFN Bari
 */

#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "Geometry/GEMGeometry/interface/GEMGeometry.h"

#include <memory>

class GEMGeometryESModule : public edm::ESProducer 
{
 public:
  /// Constructor
  GEMGeometryESModule(const edm::ParameterSet & p);
  
  /// Destructor
  ~GEMGeometryESModule() override;
  
  /// Produce GEMGeometry.
  std::unique_ptr<GEMGeometry> produce(const MuonGeometryRecord & record);
  
 private:  
  // use the DDD as Geometry source
  bool useDDD;

};
#endif
