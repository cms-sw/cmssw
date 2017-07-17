#include "PixelToFEDAssociateFromAsciiESProducer.h"

#include <string>

using namespace edm;

PixelToFEDAssociateFromAsciiESProducer::
    PixelToFEDAssociateFromAsciiESProducer(const edm::ParameterSet & p)
    : theConfig(p)
{
  std::string myname = "PixelToFEDAssociateFromAscii";
  setWhatProduced(this,myname);
}

PixelToFEDAssociateFromAsciiESProducer::
    ~PixelToFEDAssociateFromAsciiESProducer()
{ }

std::shared_ptr<PixelToFEDAssociate> PixelToFEDAssociateFromAsciiESProducer::
    produce(const TrackerDigiGeometryRecord & r)
{
  theAssociator = std::make_shared<PixelToFEDAssociateFromAscii>(theConfig.getParameter<std::string>("fileName"));
  return theAssociator;
}

