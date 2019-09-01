#include "PixelToFEDAssociateFromAsciiESProducer.h"

#include <string>

using namespace edm;

PixelToFEDAssociateFromAsciiESProducer::PixelToFEDAssociateFromAsciiESProducer(const edm::ParameterSet& p)
    : theConfig(p) {
  std::string myname = "PixelToFEDAssociateFromAscii";
  setWhatProduced(this, myname);
}

PixelToFEDAssociateFromAsciiESProducer::~PixelToFEDAssociateFromAsciiESProducer() {}

std::unique_ptr<PixelToFEDAssociate> PixelToFEDAssociateFromAsciiESProducer::produce(
    const TrackerDigiGeometryRecord& r) {
  return std::make_unique<PixelToFEDAssociateFromAscii>(theConfig.getParameter<std::string>("fileName"));
}
