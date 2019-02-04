#ifndef GlobalTrackingGeometryBuilder_GlobalTrackingGeometryESProducer_h
#define GlobalTrackingGeometryBuilder_GlobalTrackingGeometryESProducer_h

/** \class GlobalTrackingGeometry
 * 
 *  ESProducer for GlobalTrackingGeometry in MuonGeometryRecord
 *
 *  \author Matteo Sani
 */

#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "Geometry/Records/interface/GlobalTrackingGeometryRecord.h"

#include <memory>
#include <string>

class GlobalTrackingGeometry;

class GlobalTrackingGeometryESProducer : public edm::ESProducer {
public:

  /// Constructor
  GlobalTrackingGeometryESProducer(const edm::ParameterSet & p);

  /// Destructor
  ~GlobalTrackingGeometryESProducer() override;

  /// Produce GlobalTrackingGeometry
  std::unique_ptr<GlobalTrackingGeometry> produce(const GlobalTrackingGeometryRecord& record);

private:  

};
#endif






