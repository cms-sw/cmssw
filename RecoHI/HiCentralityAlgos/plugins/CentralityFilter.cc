//
// Original Author:  Yetkin Yilmaz,32 4-A08,+41227673039,
//         Created:  Tue Jun 29 12:19:49 CEST 2010
//
//

// system include files
#include <memory>
#include <vector>
#include <string>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDFilter.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/Exception.h"

//
// class declaration
//

class CentralityFilter : public edm::EDFilter {
public:
  explicit CentralityFilter(const edm::ParameterSet&);
  ~CentralityFilter() override;

private:
  void beginJob() override;
  bool filter(edm::Event&, const edm::EventSetup&) override;
  void endJob() override;

  // ----------member data ---------------------------

  std::vector<int> selectedBins_;
  edm::Handle<int> cbin_;
  edm::EDGetTokenT<int> tag_;
};

CentralityFilter::CentralityFilter(const edm::ParameterSet& iConfig)
    : selectedBins_(iConfig.getParameter<std::vector<int> >("selectedBins")) {
  using namespace edm;
  tag_ = consumes<int>(iConfig.getParameter<edm::InputTag>("BinLabel"));
}

CentralityFilter::~CentralityFilter() {}

// ------------ method called on each new Event  ------------
bool CentralityFilter::filter(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  bool result = false;

  using namespace edm;
  iEvent.getByToken(tag_, cbin_);

  int bin = *cbin_;

  for (int selectedBin : selectedBins_) {
    if (bin == selectedBin)
      result = true;
  }

  return result;
}

// ------------ method called once each job just before starting event loop  ------------
void CentralityFilter::beginJob() {}

// ------------ method called once each job just after ending the event loop  ------------
void CentralityFilter::endJob() {}

//define this as a plug-in
DEFINE_FWK_MODULE(CentralityFilter);
