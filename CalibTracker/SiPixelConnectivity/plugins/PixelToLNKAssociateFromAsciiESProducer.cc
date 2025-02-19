#include "PixelToLNKAssociateFromAsciiESProducer.h"

#include <string>

using namespace edm;

PixelToLNKAssociateFromAsciiESProducer::
    PixelToLNKAssociateFromAsciiESProducer(const edm::ParameterSet & p)
    : theConfig(p)
{
  std::string myname = "PixelToLNKAssociateFromAscii";
  setWhatProduced(this,myname);
}

PixelToLNKAssociateFromAsciiESProducer::
    ~PixelToLNKAssociateFromAsciiESProducer()
{ }

boost::shared_ptr<PixelToFEDAssociate> PixelToLNKAssociateFromAsciiESProducer::
    produce(const TrackerDigiGeometryRecord & r)
{
  theAssociator = boost::shared_ptr<PixelToFEDAssociate>(
     new PixelToLNKAssociateFromAscii(
             theConfig.getParameter<std::string>("fileName")));
  return theAssociator;
}

