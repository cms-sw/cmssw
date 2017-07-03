#ifndef FastSimulation_TrackerSetup_TrackerInteractionGeometryESProducer_H
#define FastSimulation_TrackerSetup_TrackerInteractionGeometryESProducer_H

#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FastSimulation/TrackerSetup/interface/TrackerInteractionGeometryRecord.h"
#include "FastSimulation/TrackerSetup/interface/TrackerInteractionGeometry.h"
#include <memory>
#include <string>

class  TrackerInteractionGeometryESProducer: public edm::ESProducer{
 public:
  TrackerInteractionGeometryESProducer(const edm::ParameterSet & p);
  ~TrackerInteractionGeometryESProducer() override; 
  std::shared_ptr<TrackerInteractionGeometry> produce(const TrackerInteractionGeometryRecord &);
 private:
  std::shared_ptr<TrackerInteractionGeometry> _tracker;
  std::string _label;
  edm::ParameterSet theTrackerMaterial;
};


#endif




