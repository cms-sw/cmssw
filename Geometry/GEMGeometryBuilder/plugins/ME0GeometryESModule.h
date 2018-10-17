#ifndef Geometry_GEMGeometryBuilder_ME0GeometryESModule_h
#define Geometry_GEMGeometryBuilder_ME0GeometryESModule_h

/** \class ME0GeometryESModule
 *
 *  ESProducer for ME0Geometry in MuonGeometryRecord
 *
 *  \author M. Maggi - INFN Bari
 */

#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "Geometry/GEMGeometry/interface/ME0Geometry.h"

#include <memory>

class ME0GeometryESModule : public edm::ESProducer
{
 public:
  /// Constructor
  ME0GeometryESModule(const edm::ParameterSet & p);

  /// Destructor
  ~ME0GeometryESModule() override;

  /// Produce ME0Geometry.
  std::unique_ptr<ME0Geometry>  produce(const MuonGeometryRecord & record);

 private:
  // use the DDD as Geometry source
  bool useDDD;
};
#endif
