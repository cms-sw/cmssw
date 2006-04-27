#ifndef PixelFitterByConformalMappingAndLineESProducer_H
#define PixelFitterByConformalMappingAndLineESProducer_H

#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include <boost/shared_ptr.hpp>

#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"
#include "RecoPixelVertexing/PixelTrackFitting/interface/PixelFitterByConformalMappingAndLine.h"


class PixelFitterByConformalMappingAndLineESProducer : public edm::ESProducer {
public:
  PixelFitterByConformalMappingAndLineESProducer(const edm::ParameterSet & p);
  virtual ~PixelFitterByConformalMappingAndLineESProducer();
  boost::shared_ptr<PixelFitter> produce(const TrackingComponentsRecord &);
private:
  boost::shared_ptr<PixelFitter> theFitter;
  edm::ParameterSet theConfig;
};

#endif
