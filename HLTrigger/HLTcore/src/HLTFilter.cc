/** \class HLTFilter
 *
 *  
 *  This class derives from EDFilter and adds a few HLT specific
 *  items. Any and all HLT filters must derive from the HLTFilter
 *  class!
 *
 *  $Date: 2012/02/01 13:50:55 $
 *  $Revision: 1.9 $
 *
 *  \author Martin Grunewald
 *
 */

#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"
#include "HLTrigger/HLTcore/interface/HLTFilter.h"

HLTFilter::HLTFilter(const edm::ParameterSet & config) :
  EDFilter(),
  saveTags_(config.getParameter<bool>("saveTags"))
{ 
  // register common HLTFilter products
  produces<trigger::TriggerFilterObjectWithRefs>();
}

void
HLTFilter::makeHLTFilterDescription(edm::ParameterSetDescription& desc) {
  desc.add<bool>("saveTags",false);
}

HLTFilter::~HLTFilter() 
{ }

bool HLTFilter::filter(edm::Event & event, const edm::EventSetup & setup) {
  std::auto_ptr<trigger::TriggerFilterObjectWithRefs> filterproduct( new trigger::TriggerFilterObjectWithRefs(path(), module()) );

  // compute the result of the HLTFilter implementation
  bool result = hltFilter(event, setup, * filterproduct);

  // put filter object into the Event
  event.put(filterproduct);

  // retunr the result of the HLTFilter
  return result;
}


int HLTFilter::path() const {
  int p(-2);
  edm::CurrentProcessingContext const * cpc(currentContext());
  if (cpc != 0)
    p = cpc->pathInSchedule();
  return p;
}

int HLTFilter::module() const {
  int m(-2);
  edm::CurrentProcessingContext const * cpc(currentContext());
  if (cpc != 0)
    m = cpc->slotInPath();
  return m;
}

std::pair<int,int> HLTFilter::pmid() const {
  std::pair<int,int> pm(-2, -2);
  edm::CurrentProcessingContext const * cpc(currentContext());
  if (cpc != 0)
    pm = std::make_pair(cpc->pathInSchedule(), cpc->slotInPath());
  return pm;
}

const std::string* HLTFilter::pathName() const {
  edm::CurrentProcessingContext const * cpc(currentContext());
  if (cpc != 0)
    return cpc->pathName();
  else
    return 0;
}

const std::string* HLTFilter::moduleLabel() const {
  edm::CurrentProcessingContext const * cpc(currentContext());
  if (cpc != 0)
    return cpc->moduleLabel();
  else
    return 0;
}
