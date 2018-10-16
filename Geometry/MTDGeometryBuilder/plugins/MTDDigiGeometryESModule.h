#ifndef Geometry_MTDGeometryBuilder_MTDDigiGeometryESModule_H
#define Geometry_MTDGeometryBuilder_MTDDigiGeometryESModule_H

#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "Geometry/Records/interface/MTDDigiGeometryRecord.h"
#include "Geometry/MTDGeometryBuilder/interface/MTDGeometry.h"
#include <memory>

#include <string>

namespace edm {
  class ConfigurationDescriptions;
}

class  MTDDigiGeometryESModule: public edm::ESProducer{
 public:
  MTDDigiGeometryESModule(const edm::ParameterSet & p);
  ~MTDDigiGeometryESModule() override; 
  std::shared_ptr<MTDGeometry> produce(const MTDDigiGeometryRecord &);

  static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);
  
 private:
  /// Called when geometry description changes
  std::shared_ptr<MTDGeometry> mtd_;
  const std::string alignmentsLabel_;
  const std::string myLabel_;
  bool applyAlignment_; // Switch to apply alignment corrections
  bool fromDDD_;
};


#endif




