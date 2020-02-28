#ifndef PixelToLNKAssociateFromAsciiESProducer_H
#define PixelToLNKAssociateFromAsciiESProducer_H

#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include <memory>

#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "CalibTracker/SiPixelConnectivity/interface/PixelToLNKAssociateFromAscii.h"

class PixelToLNKAssociateFromAsciiESProducer : public edm::ESProducer {
public:
  PixelToLNKAssociateFromAsciiESProducer(const edm::ParameterSet& p);
  ~PixelToLNKAssociateFromAsciiESProducer() override;
  std::unique_ptr<PixelToFEDAssociate> produce(const TrackerDigiGeometryRecord&);

private:
  edm::ParameterSet theConfig;
};

#endif
