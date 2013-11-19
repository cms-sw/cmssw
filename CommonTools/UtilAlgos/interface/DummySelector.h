#ifndef CommonTools_UtilAlgos_DummySelector_h
#define CommonTools_UtilAlgos_DummySelector_h
/* \class DummySelector
 *
 * Dummy generic selector following the
 * interface proposed in the document:
 *
 * https://twiki.cern.ch/twiki/bin/view/CMS/SelectorInterface
 *
 * \author Luca Lista, INFN
 */
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Utilities/interface/EDMException.h"

namespace edm {
  class ParameterSet;
  class Event;
  class EventSetup;
}

class DummySelector {
public:
  explicit DummySelector(const edm::ParameterSet&, edm::ConsumesCollector & iC) : updated_(false) { }
  void newEvent(const edm::Event&, const edm::EventSetup&) { updated_ = true; }
  template<typename T>
  bool operator()(const T&) {
    if(!updated_)
      throw edm::Exception(edm::errors::Configuration)
	<< "DummySelector: forgot to call newEvent\n";
    return true;
  }
private:
  bool updated_;
};

namespace dummy {
  template<typename T>
  inline bool select(const T&) { return true; }
}

EVENTSETUP_STD_INIT(DummySelector);

#endif
