#ifndef Geometry_TrackerGeometryBuilder_TrackerDigiGeometryESModule_H
#define Geometry_TrackerGeometryBuilder_TrackerDigiGeometryESModule_H

#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include <boost/shared_ptr.hpp>

#include <string>

class  TrackerDigiGeometryESModule: public edm::ESProducer{
 public:
  TrackerDigiGeometryESModule(const edm::ParameterSet & p);
  virtual ~TrackerDigiGeometryESModule(); 
  boost::shared_ptr<TrackerGeometry> produce(const TrackerDigiGeometryRecord &);
 private:
  /// Called when geometry description changes
  void ddGeometryCallback_( const IdealGeometryRecord& );
  void gdGeometryCallback_( const IdealGeometryRecord& );
  boost::shared_ptr<TrackerGeometry> _tracker;
  bool applyAlignment_; // Switch to apply alignment corrections
  bool fromDDD_;
  const std::string alignmentsLabel_;
  const std::string myLabel_;
};


#endif




