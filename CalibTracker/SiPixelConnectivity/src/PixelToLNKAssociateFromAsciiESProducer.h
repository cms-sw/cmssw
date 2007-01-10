#ifndef PixelToLNKAssociateFromAsciiESProducer_H
#define PixelToLNKAssociateFromAsciiESProducer_H

#include  "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include <boost/shared_ptr.hpp>

#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "CalibTracker/SiPixelConnectivity/interface/PixelToLNKAssociateFromAscii.h"

class PixelToLNKAssociateFromAsciiESProducer : public edm::ESProducer {
public:
  PixelToLNKAssociateFromAsciiESProducer(const edm::ParameterSet & p);
  virtual ~PixelToLNKAssociateFromAsciiESProducer();
  boost::shared_ptr<PixelToFEDAssociate> produce(const TrackerDigiGeometryRecord&);
private:
  boost::shared_ptr<PixelToFEDAssociate> theAssociator;
  edm::ParameterSet theConfig;
};

#endif

