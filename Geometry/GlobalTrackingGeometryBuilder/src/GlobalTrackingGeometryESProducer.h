#ifndef GlobalTrackingGeometryBuilder_GlobalTrackingGeometryESProducer_h
#define GlobalTrackingGeometryBuilder_GlobalTrackingGeometryESProducer_h

/** \class CSCGeometryESModule
 * 
 *  ESProducer for GlobalTrackingGeometry in MuonGeometryRecord
 *
 *  \author Matteo Sani
 */

#include <FWCore/Framework/interface/ESProducer.h>
#include <FWCore/ParameterSet/interface/ParameterSet.h>
#include <Geometry/Records/interface/GlobalTrackingGeometryRecord.h>
//#include <Geometry/CommonDetUnit/interface/TrackingGeometry.h>
//#include <Geometry/CSCGeometry/interface/CSCGeometry.h>
#include <boost/shared_ptr.hpp>

#include <string>

class GlobalTrackingGeometry;

class GlobalTrackingGeometryESProducer : public edm::ESProducer {
public:

  /// Constructor
  GlobalTrackingGeometryESProducer(const edm::ParameterSet & p);

  /// Destructor
  virtual ~GlobalTrackingGeometryESProducer();

  /// Produce GlobalTrackingGeometry
  boost::shared_ptr<GlobalTrackingGeometry> produce(const GlobalTrackingGeometryRecord& record);
    //  boost::shared_ptr<TrackingGeometry>  produce(const MuonGeometryRecord & record);

private:  

};
#endif






