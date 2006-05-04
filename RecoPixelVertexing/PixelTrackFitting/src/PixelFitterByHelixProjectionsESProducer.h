#ifndef PixelFitterByHelixProjectionsESProducer_H
#define PixelFitterByHelixProjectionsESProducer_H

#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include <boost/shared_ptr.hpp>

#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"
#include "RecoPixelVertexing/PixelTrackFitting/interface/PixelFitterByHelixProjections.h"


class PixelFitterByHelixProjectionsESProducer : public edm::ESProducer {
public:
  PixelFitterByHelixProjectionsESProducer(const edm::ParameterSet & p);
  virtual ~PixelFitterByHelixProjectionsESProducer();
  boost::shared_ptr<PixelFitter> produce(const TrackingComponentsRecord &);
private:
  boost::shared_ptr<PixelFitter> theFitter;
  edm::ParameterSet theConfig;
};

#endif
