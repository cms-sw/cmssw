#include "PixelToLNKAssociateFromAsciiESProducer.h"

#include <string>

using namespace edm;

PixelToLNKAssociateFromAsciiESProducer::PixelToLNKAssociateFromAsciiESProducer(const edm::ParameterSet& p)
    : theConfig(p) {
  std::string myname = "PixelToLNKAssociateFromAscii";
  setWhatProduced(this, myname);
}

PixelToLNKAssociateFromAsciiESProducer::~PixelToLNKAssociateFromAsciiESProducer() {}

std::unique_ptr<PixelToFEDAssociate> PixelToLNKAssociateFromAsciiESProducer::produce(
    const TrackerDigiGeometryRecord& r) {
  return std::make_unique<PixelToLNKAssociateFromAscii>(theConfig.getParameter<std::string>("fileName"));
}
