#ifndef FastSimulation_TrackerSetup_TrackerInteractionGeometryESProducer_H
#define FastSimulation_TrackerSetup_TrackerInteractionGeometryESProducer_H

#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FastSimulation/TrackerSetup/interface/TrackerInteractionGeometryRecord.h"
#include "FastSimulation/TrackerSetup/interface/TrackerInteractionGeometry.h"
#include <boost/shared_ptr.hpp>
#include <string>

class  TrackerInteractionGeometryESProducer: public edm::ESProducer{
 public:
  TrackerInteractionGeometryESProducer(const edm::ParameterSet & p);
  virtual ~TrackerInteractionGeometryESProducer(); 
  boost::shared_ptr<TrackerInteractionGeometry> produce(const TrackerInteractionGeometryRecord &);
 private:
  boost::shared_ptr<TrackerInteractionGeometry> _tracker;
  std::string _label;
  edm::ParameterSet theTrackerMaterial;
};


#endif




