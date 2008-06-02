//
// $Id: CSA07ProcessIdFilter.cc,v 1.1.2.1 2008/04/30 14:53:17 lowette Exp $
//

#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/Framework/interface/EDFilter.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "PhysicsTools/HepMCCandAlgos/interface/CSA07ProcessId.h"

#include <vector>
#include <string>


class CSA07ProcessIdFilter : public edm::EDFilter {

  public:

    explicit CSA07ProcessIdFilter(const edm::ParameterSet & iConfig);
    virtual ~CSA07ProcessIdFilter();

  private:

    virtual bool filter(edm::Event & iEvent, const edm::EventSetup & iSetup);

  private:

    std::vector<int> csa07Ids_;
    double overallLumi_;
    std::string csa07EventWeightProducerLabel_;

};


CSA07ProcessIdFilter::CSA07ProcessIdFilter(const edm::ParameterSet & iConfig) :
  csa07Ids_(iConfig.getParameter<std::vector<int> >("csa07Ids")),
  overallLumi_(iConfig.getParameter<double>("overallLumi")),
  csa07EventWeightProducerLabel_(iConfig.getParameter<std::string>("csa07EventWeightProducerLabel")) {
}


CSA07ProcessIdFilter::~CSA07ProcessIdFilter() {
}


bool CSA07ProcessIdFilter::filter(edm::Event & iEvent, const edm::EventSetup & iSetup) {
  bool accepted = false;
  int eventId = csa07::csa07ProcessId(iEvent, overallLumi_, csa07EventWeightProducerLabel_);
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
