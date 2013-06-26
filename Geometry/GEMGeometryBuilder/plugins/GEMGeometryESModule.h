#ifndef GEMGeometry_GEMGeometryESModule_h
#define GEMGeometry_GEMGeometryESModule_h

/** \class GEMGeometryESModule
 * 
 *  ESProducer for GEMGeometry in MuonGeometryRecord
 *
 *  \author M. Maggi - INFN Bari
 */

#include <FWCore/Framework/interface/ESProducer.h>
#include <FWCore/ParameterSet/interface/ParameterSet.h>
#include <Geometry/Records/interface/MuonGeometryRecord.h>
#include "Geometry/GEMGeometry/interface/GEMGeometry.h"
#include <boost/shared_ptr.hpp>

class GEMGeometryESModule : public edm::ESProducer {
public:
  /// Constructor
  GEMGeometryESModule(const edm::ParameterSet & p);

  /// Destructor
  virtual ~GEMGeometryESModule();

  /// Produce GEMGeometry.
  boost::shared_ptr<GEMGeometry>  produce(const MuonGeometryRecord & record);

private:  

  bool comp11,useDDD;

};
#endif
