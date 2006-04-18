
#include "IOPool/Streamer/interface/FragmentInput.h"
#include "IOPool/Streamer/interface/HLTInfo.h"
#include "IOPool/Streamer/interface/ClassFiller.h"
#include "IOPool/Streamer/interface/Utilities.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

namespace stor
{
  inline edm::Service<HLTInfo> getInfo()
  {
    edm::Service<HLTInfo> i;
    return i;
  }

  FragmentInput::FragmentInput(edm::ParameterSet const& pset,
			  edm::InputSourceDescription const& desc):
    edm::InputSource(pset, desc),
    extractor_(getInfo()->getEventQueue())
  {
    edm::Service<HLTInfo> info;

    // jbk - hopefully the next line is not needed
    info->declareStreamers(info->products());
    info->buildClassCache(info->products());

    info->mergeRegistry(productRegistry());
    edm::loadExtraClasses();
  }

  FragmentInput::~FragmentInput()
  {
  }

  std::auto_ptr<edm::EventPrincipal> FragmentInput::read()
  {
    return extractor_.extract();
  }
}
