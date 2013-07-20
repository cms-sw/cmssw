#ifndef GlobalTrackingGeometryBuilder_GlobalTrackingGeometryESProducer_h
#define GlobalTrackingGeometryBuilder_GlobalTrackingGeometryESProducer_h

/** \class GlobalTrackingGeometry
 * 
 *  ESProducer for GlobalTrackingGeometry in MuonGeometryRecord
 *
 *  $Date: 2011/08/16 14:54:34 $
 *  $Revision: 1.1 $
 *  \author Matteo Sani
 */

#include <FWCore/Framework/interface/ESProducer.h>
#include <FWCore/ParameterSet/interface/ParameterSet.h>
#include <Geometry/Records/interface/GlobalTrackingGeometryRecord.h>
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

private:  

};
#endif






