#ifndef ME0Geometry_ME0GeometryESModule_h
#define ME0Geometry_ME0GeometryESModule_h

/** \class ME0GeometryESModule
 * 
 *  ESProducer for ME0Geometry in MuonGeometryRecord
 *
 *  \author M. Maggi - INFN Bari
 */

#include <FWCore/Framework/interface/ESProducer.h>
#include <FWCore/ParameterSet/interface/ParameterSet.h>
#include <Geometry/Records/interface/MuonGeometryRecord.h>
#include "Geometry/GEMGeometry/interface/ME0Geometry.h"
#include <boost/shared_ptr.hpp>

class ME0GeometryESModule : public edm::ESProducer {
public:
  /// Constructor
  ME0GeometryESModule(const edm::ParameterSet & p);

  /// Destructor
  virtual ~ME0GeometryESModule();

  /// Produce ME0Geometry.
  boost::shared_ptr<ME0Geometry>  produce(const MuonGeometryRecord & record);

private:  

  bool comp11,useDDD;

};
#endif
