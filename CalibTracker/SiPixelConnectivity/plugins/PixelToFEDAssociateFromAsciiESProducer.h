#ifndef PixelToFEDAssociateFromAsciiESProducer_H
#define PixelToFEDAssociateFromAsciiESProducer_H

#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include <memory>

#include "CalibTracker/SiPixelConnectivity/interface/PixelToFEDAssociateFromAscii.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"

class PixelToFEDAssociateFromAsciiESProducer : public edm::ESProducer {
public:
  PixelToFEDAssociateFromAsciiESProducer(const edm::ParameterSet &p);
  ~PixelToFEDAssociateFromAsciiESProducer() override;
  std::unique_ptr<PixelToFEDAssociate>
  produce(const TrackerDigiGeometryRecord &);

private:
  edm::ParameterSet theConfig;
};

#endif
