#ifndef RecoTracker_MeasurementDet_MeasurementTrackerESProducer_h
#define RecoTracker_MeasurementDet_MeasurementTrackerESProducer_h

#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "RecoTracker/Record/interface/CkfComponentsRecord.h"
#include "MeasurementTracker.h"
#include <boost/shared_ptr.hpp>

class  MeasurementTrackerESProducer: public edm::ESProducer{
 public:
  MeasurementTrackerESProducer(const edm::ParameterSet & p);
  virtual ~MeasurementTrackerESProducer(); 
  boost::shared_ptr<MeasurementTracker> produce(const CkfComponentsRecord &);
 private:
  boost::shared_ptr<MeasurementTracker> _measurementTracker;
  edm::ParameterSet pset_;
};


#endif




