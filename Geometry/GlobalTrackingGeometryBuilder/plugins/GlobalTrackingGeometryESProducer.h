#ifndef GlobalTrackingGeometryBuilder_GlobalTrackingGeometryESProducer_h
#define GlobalTrackingGeometryBuilder_GlobalTrackingGeometryESProducer_h

/** \class GlobalTrackingGeometry
 * 
 *  ESProducer for GlobalTrackingGeometry in MuonGeometryRecord
 *
 *  $Date: 2006/05/09 14:08:52 $
 *  $Revision: 1.2 $
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






