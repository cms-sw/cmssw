#ifndef PixelToFEDAssociateFromAsciiESProducer_H
#define PixelToFEDAssociateFromAsciiESProducer_H

#include  "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include <boost/shared_ptr.hpp>

#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "CalibTracker/SiPixelConnectivity/interface/PixelToFEDAssociateFromAscii.h"

class PixelToFEDAssociateFromAsciiESProducer : public edm::ESProducer {
public:
  PixelToFEDAssociateFromAsciiESProducer(const edm::ParameterSet & p);
  virtual ~PixelToFEDAssociateFromAsciiESProducer();
  boost::shared_ptr<PixelToFEDAssociate> produce(const TrackerDigiGeometryRecord&);
private:
  boost::shared_ptr<PixelToFEDAssociate> theAssociator;
  edm::ParameterSet theConfig;
};

#endif

