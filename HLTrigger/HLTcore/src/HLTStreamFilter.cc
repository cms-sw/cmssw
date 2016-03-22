/** \class HLTStreamFilter
 *
 *
 *  This class derives from EDFilter and adds a few HLT specific
 *  items. Any and all HLT filters must derive from the HLTStreamFilter
 *  class!
 *
 *
 *  \author Martin Grunewald
 *
 */

#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"
#include "HLTrigger/HLTcore/interface/HLTStreamFilter.h"

#include "FWCore/ServiceRegistry/interface/PathContext.h"
#include "FWCore/ServiceRegistry/interface/PlaceInPathContext.h"
#include "FWCore/ServiceRegistry/interface/ModuleCallingContext.h"

HLTStreamFilter::HLTStreamFilter(const edm::ParameterSet & config) :
  EDFilter(),
  saveTags_(config.getParameter<bool>("saveTags"))
{
  // register common HLTStreamFilter products
  produces<trigger::TriggerFilterObjectWithRefs>();
}

void
HLTStreamFilter::makeHLTFilterDescription(edm::ParameterSetDescription& desc) {
  desc.add<bool>("saveTags",true);
}

HLTStreamFilter::~HLTStreamFilter()
{ }

bool HLTStreamFilter::filter(edm::Event & event, const edm::EventSetup & setup) {
  std::auto_ptr<trigger::TriggerFilterObjectWithRefs> filterproduct( new trigger::TriggerFilterObjectWithRefs(path(event), module(event)) );

  // compute the result of the HLTStreamFilter implementation
  bool result = hltFilter(event, setup, * filterproduct);

  // put filter object into the Event
  event.put(filterproduct);

  // retunr the result of the HLTStreamFilter
  return result;
}

int HLTStreamFilter::path(edm::Event const& event) const {
  return static_cast<int>(event.moduleCallingContext()->placeInPathContext()->pathContext()->pathID());
}

int HLTStreamFilter::module(edm::Event const& event) const {
  return static_cast<int>(event.moduleCallingContext()->placeInPathContext()->placeInPath());
}

std::pair<int,int> HLTStreamFilter::pmid(edm::Event const& event) const {
  edm::PlaceInPathContext const* placeInPathContext = event.moduleCallingContext()->placeInPathContext();
  return std::make_pair(static_cast<int>(placeInPathContext->pathContext()->pathID()),
                        static_cast<int>(placeInPathContext->placeInPath()));
}

const std::string* HLTStreamFilter::pathName(edm::Event const& event) const {
  return &event.moduleCallingContext()->placeInPathContext()->pathContext()->pathName();
}

const std::string* HLTStreamFilter::moduleLabel() const {
  return &moduleDescription().moduleLabel();
}
