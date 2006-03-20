#ifndef RPCSimAlgo_RPCGeometryESModule_h
#define RPCSimAlgo_RPCGeometryESModule_h

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
#include <boost/shared_ptr.hpp>

class RPCGeometryESModule : public edm::ESProducer {
public:
  /// Constructor
  RPCGeometryESModule(const edm::ParameterSet & p);

  /// Destructor
  virtual ~RPCGeometryESModule();

  /// Produce RPCGeometry.
  boost::shared_ptr<RPCGeometry>  produce(const MuonGeometryRecord & record);

private:  

};
#endif
