#ifndef CSCGeometryBuilder_CSCGeometryESModule_h
#define CSCGeometryBuilder_CSCGeometryESModule_h

/** \class CSCGeometryESModule
 * 
 *  ESProducer for CSCGeometry in MuonGeometryRecord
 *
 *  \author Tim Cox
 */

#include <FWCore/Framework/interface/ESProducer.h>
#include <FWCore/ParameterSet/interface/ParameterSet.h>
#include <Geometry/Records/interface/MuonGeometryRecord.h>
//#include <Geometry/CommonDetUnit/interface/TrackingGeometry.h>
#include <Geometry/CSCGeometry/interface/CSCGeometry.h>
#include <boost/shared_ptr.hpp>

#include <string>

class CSCGeometryESModule : public edm::ESProducer {
public:
  /// Constructor
  CSCGeometryESModule(const edm::ParameterSet & p);

  /// Destructor
  virtual ~CSCGeometryESModule();

  /// Produce CSCGeometry
  boost::shared_ptr<CSCGeometry>  produce(const MuonGeometryRecord & record);
    //  boost::shared_ptr<TrackingGeometry>  produce(const MuonGeometryRecord & record);

private:  

};
#endif






