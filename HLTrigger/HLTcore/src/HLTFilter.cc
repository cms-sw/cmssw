/** \class HLTFilter
 *
 *
 *  This class derives from EDFilter and adds a few HLT specific
 *  items. Any and all HLT filters must derive from the HLTFilter
 *  class!
 *
 *
 *  \author Martin Grunewald
 *
 */

#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"
#include "HLTrigger/HLTcore/interface/HLTFilter.h"

#include "FWCore/ServiceRegistry/interface/PathContext.h"
#include "FWCore/ServiceRegistry/interface/PlaceInPathContext.h"
#include "FWCore/ServiceRegistry/interface/ModuleCallingContext.h"

HLTFilter::HLTFilter(const edm::ParameterSet & config) :
  EDFilter(),
  saveTags_(config.getParameter<bool>("saveTags"))
{
  // register common HLTFilter products
  produces<trigger::TriggerFilterObjectWithRefs>();
}

void
HLTFilter::makeHLTFilterDescription(edm::ParameterSetDescription& desc) {
  desc.add<bool>("saveTags",true);
}

HLTFilter::~HLTFilter()
{ }

bool HLTFilter::filter(edm::StreamID, edm::Event & event, const edm::EventSetup & setup) const {
  std::auto_ptr<trigger::TriggerFilterObjectWithRefs> filterproduct( new trigger::TriggerFilterObjectWithRefs(path(event), module(event)) );

  // compute the result of the HLTFilter implementation
  bool result = hltFilter(event, setup, * filterproduct);

  // put filter object into the Event
  event.put(filterproduct);

  // retunr the result of the HLTFilter
  return result;
}

int HLTFilter::path(edm::Event const& event) const {
  return static_cast<int>(event.moduleCallingContext()->placeInPathContext()->pathContext()->pathID());
}

int HLTFilter::module(edm::Event const& event) const {
  return static_cast<int>(event.moduleCallingContext()->placeInPathContext()->placeInPath());
}

std::pair<int,int> HLTFilter::pmid(edm::Event const& event) const {
  edm::PlaceInPathContext const* placeInPathContext = event.moduleCallingContext()->placeInPathContext();
  return std::make_pair(static_cast<int>(placeInPathContext->pathContext()->pathID()),
                        static_cast<int>(placeInPathContext->placeInPath()));
}

const std::string* HLTFilter::pathName(edm::Event const& event) const {
  return &event.moduleCallingContext()->placeInPathContext()->pathContext()->pathName();
}

const std::string* HLTFilter::moduleLabel() const {
  return &moduleDescription().moduleLabel();
}
