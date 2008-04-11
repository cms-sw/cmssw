//
// $Id: CSA07ProcessIdFilter.cc,v 1.2 2008/02/18 13:47:05 lowette Exp $
//

#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/Framework/interface/EDFilter.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "PhysicsTools/HepMCCandAlgos/interface/CSA07ProcessId.h"

#include <vector>


class CSA07ProcessIdFilter : public edm::EDFilter {

  public:

    explicit CSA07ProcessIdFilter(const edm::ParameterSet & iConfig);
    virtual ~CSA07ProcessIdFilter();

  private:

    virtual bool filter(edm::Event & iEvent, const edm::EventSetup & iSetup);

  private:

    std::vector<int> csa07Ids_;

};


CSA07ProcessIdFilter::CSA07ProcessIdFilter(const edm::ParameterSet & iConfig) :
  csa07Ids_(iConfig.getParameter<std::vector<int> >("csa07Ids")) {
}


CSA07ProcessIdFilter::~CSA07ProcessIdFilter() {
}


bool CSA07ProcessIdFilter::filter(edm::Event & iEvent, const edm::EventSetup & iSetup) {
  bool accepted = false;
  int eventId = csa07::csa07ProcessId(iEvent);
  for (std::vector<int>::iterator id = csa07Ids_.begin(); id < csa07Ids_.end(); id++) {
    if (eventId == *id) {
      accepted = true;
      break;
    }
  }
  return accepted;
}


//define this as a plug-in
DEFINE_FWK_MODULE(CSA07ProcessIdFilter);
