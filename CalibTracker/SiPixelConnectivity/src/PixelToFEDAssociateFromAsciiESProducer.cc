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

boost::shared_ptr<PixelToFEDAssociate> PixelToFEDAssociateFromAsciiESProducer::
    produce(const TrackerDigiGeometryRecord & r)
{
  theAssociator = boost::shared_ptr<PixelToFEDAssociate>(
     new PixelToFEDAssociateFromAscii(
             theConfig.getParameter<std::string>("fileName")));
  return theAssociator;
}

