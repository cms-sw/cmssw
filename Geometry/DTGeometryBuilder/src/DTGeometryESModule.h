#ifndef DTGeometryBuilder_DTGeometryESModule_h
#define DTGeometryBuilder_DTGeometryESModule_h

/** \class DTGeometryESModule
 * 
 *  ESProducer for DTGeometry in MuonGeometryRecord
 *
 *  $Date: 2005/10/27 08:55:14 $
 *  $Revision: 1.2 $
 *  \author N. Amapane - CERN
 */

#include <FWCore/Framework/interface/ESProducer.h>
#include <FWCore/ParameterSet/interface/ParameterSet.h>
#include <Geometry/Records/interface/MuonGeometryRecord.h>
#include <Geometry/DTGeometry/interface/DTGeometry.h>
#include <boost/shared_ptr.hpp>

class DTGeometryESModule : public edm::ESProducer {
public:
  /// Constructor
  DTGeometryESModule(const edm::ParameterSet & p);

  /// Destructor
  virtual ~DTGeometryESModule();

  /// Produce DTGeometry.
  boost::shared_ptr<DTGeometry>  produce(const MuonGeometryRecord & record);

private:  

};
#endif






