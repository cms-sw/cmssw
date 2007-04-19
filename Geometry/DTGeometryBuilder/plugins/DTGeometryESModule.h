#ifndef DTGeometryBuilder_DTGeometryESModule_h
#define DTGeometryBuilder_DTGeometryESModule_h

/** \class DTGeometryESModule
 * 
 *  ESProducer for DTGeometry in MuonGeometryRecord
 *
 *  $Date: 2006/08/22 15:58:37 $
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

  boost::shared_ptr<DTGeometry> _dtGeometry;
  bool applyAlignment_; // Switch to apply alignment corrections

};
#endif






