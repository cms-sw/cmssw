#ifndef RPCGeometry_RPCGeometryESModule_h
#define RPCGeometry_RPCGeometryESModule_h

/** \class RPCGeometryESModule
 * 
 *  ESProducer for RPCGeometry in MuonGeometryRecord
 *
 *  \author M. Maggi - INFN Bari
 */

#include <FWCore/Framework/interface/ESProducer.h>
#include <FWCore/ParameterSet/interface/ParameterSet.h>
#include <Geometry/Records/interface/MuonGeometryRecord.h>
#include "Geometry/RPCGeometry/interface/RPCGeometry.h"
#include <memory>

class RPCGeometryESModule : public edm::ESProducer {
public:
  /// Constructor
  RPCGeometryESModule(const edm::ParameterSet & p);

  /// Destructor
  ~RPCGeometryESModule() override;

  /// Produce RPCGeometry.
  std::unique_ptr<RPCGeometry>  produce(const MuonGeometryRecord & record);

private:  

  bool comp11,useDDD;

};
#endif
