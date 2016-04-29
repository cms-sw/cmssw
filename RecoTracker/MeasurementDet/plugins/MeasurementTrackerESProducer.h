#ifndef RecoTracker_MeasurementDet_MeasurementTrackerESProducer_h
#define RecoTracker_MeasurementDet_MeasurementTrackerESProducer_h

#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "RecoTracker/Record/interface/CkfComponentsRecord.h"
#include "RecoTracker/MeasurementDet/interface/MeasurementTracker.h"
#include <memory>

class  dso_hidden MeasurementTrackerESProducer: public edm::ESProducer{
 public:
  MeasurementTrackerESProducer(const edm::ParameterSet & p);
  virtual ~MeasurementTrackerESProducer(); 
  std::shared_ptr<MeasurementTracker> produce(const CkfComponentsRecord &);
 private:
  std::shared_ptr<MeasurementTracker> _measurementTracker;
  edm::ParameterSet pset_;
};


#endif




