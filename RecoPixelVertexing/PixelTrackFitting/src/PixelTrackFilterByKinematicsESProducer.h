#ifndef PixelTrackFitting_PixelTrackFilterByKinematicsESProducer_H
#define PixelTrackFitting_PixelTrackFilterByKinematicsESProducer_H

#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include <boost/shared_ptr.hpp>

#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"
#include "RecoPixelVertexing/PixelTrackFitting/interface/PixelTrackFilter.h"
#include "RecoPixelVertexing/PixelTrackFitting/interface/PixelTrackFilterByKinematics.h"


class PixelTrackFilterByKinematicsESProducer : public edm::ESProducer {
public:
  PixelTrackFilterByKinematicsESProducer(const edm::ParameterSet & p);
  virtual ~PixelTrackFilterByKinematicsESProducer();
  boost::shared_ptr<PixelTrackFilter> produce(const TrackingComponentsRecord &);
private:
  boost::shared_ptr<PixelTrackFilter> theFilter;
  edm::ParameterSet theConfig;
};

#endif

