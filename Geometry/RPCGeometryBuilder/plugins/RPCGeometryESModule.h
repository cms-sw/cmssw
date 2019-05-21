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

#include <Geometry/Records/interface/IdealGeometryRecord.h>
#include <Geometry/MuonNumbering/interface/MuonDDDConstants.h>
#include <DetectorDescription/Core/interface/DDCompactView.h>

#include "Geometry/Records/interface/RPCRecoGeometryRcd.h"
#include "CondFormats/GeometryObjects/interface/RecoIdealGeometry.h"

#include <memory>

class RPCGeometryESModule : public edm::ESProducer {
public:
  /// Constructor
  RPCGeometryESModule(const edm::ParameterSet & p);

  /// Destructor
  ~RPCGeometryESModule() override = default;

  /// Produce RPCGeometry.
  std::unique_ptr<RPCGeometry>  produce(const MuonGeometryRecord & record);

private:  

  //Used without DDD
  edm::ESGetToken<DDCompactView, IdealGeometryRecord> idealGeomToken_;
  edm::ESGetToken<MuonDDDConstants, MuonNumberingRecord> dddConstantsToken_;
  
  //Used with DDD
  edm::ESGetToken<RecoIdealGeometry, RPCRecoGeometryRcd> recoIdealToken_;

  const bool comp11;
  const bool useDDD;

};
#endif
