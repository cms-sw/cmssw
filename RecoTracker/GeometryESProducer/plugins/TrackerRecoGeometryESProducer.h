#ifndef RecoTracker_GeometryESProducer_TrackerRecoGeometryESProducer_H
#define RecoTracker_GeometryESProducer_TrackerRecoGeometryESProducer_H

#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "RecoTracker/Record/interface/TrackerRecoGeometryRecord.h"
#include "RecoTracker/TkDetLayers/interface/GeometricSearchTracker.h"
#include <boost/shared_ptr.hpp>

class  TrackerRecoGeometryESProducer: public edm::ESProducer{
 public:
  TrackerRecoGeometryESProducer(const edm::ParameterSet & p);
  virtual ~TrackerRecoGeometryESProducer(); 
  boost::shared_ptr<GeometricSearchTracker> produce(const TrackerRecoGeometryRecord &);
 private:
 boost::shared_ptr<GeometricSearchTracker> _tracker;
 std::string geoLabel;
};


#endif




