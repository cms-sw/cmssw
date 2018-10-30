#ifndef Geometry_TrackerGeometryBuilder_TrackerDigiGeometryESModule_H
#define Geometry_TrackerGeometryBuilder_TrackerDigiGeometryESModule_H

#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include <memory>

#include <string>

namespace edm {
  class ConfigurationDescriptions;
}

class  TrackerDigiGeometryESModule: public edm::ESProducer{
 public:
  TrackerDigiGeometryESModule(const edm::ParameterSet & p);
  ~TrackerDigiGeometryESModule() override; 
  std::unique_ptr<TrackerGeometry> produce(const TrackerDigiGeometryRecord &);

  static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);
  
 private:
  /// Called when geometry description changes
  const std::string alignmentsLabel_;
  const std::string myLabel_;
  bool applyAlignment_; // Switch to apply alignment corrections
  bool fromDDD_;
};


#endif




