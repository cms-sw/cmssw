// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDFilter.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

//
// class declaration
//

class NumberPerLSFilter : public edm::one::EDFilter<edm::one::WatchLuminosityBlocks> {
public:
  explicit NumberPerLSFilter(const edm::ParameterSet&);
  ~NumberPerLSFilter() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  bool filter(edm::Event&, const edm::EventSetup&) override;
  void beginLuminosityBlock(const edm::LuminosityBlock&, const edm::EventSetup&) override;
  void endLuminosityBlock(edm::LuminosityBlock const& iEvent, edm::EventSetup const&) override;

  // ----------member data ---------------------------
private:
  int counter_;
  int maxN_;
};

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
NumberPerLSFilter::NumberPerLSFilter(const edm::ParameterSet& iConfig) : maxN_(iConfig.getParameter<int>("maxN")) {
  //now do what ever initialization is needed
  counter_ = 0;
}

//
// member functions
//

// ------------ method called on each new LumiBlock ------------
void NumberPerLSFilter::beginLuminosityBlock(const edm::LuminosityBlock& lumiBlock, const edm::EventSetup& setup) {
  counter_ = 0;
}

void NumberPerLSFilter::endLuminosityBlock(edm::LuminosityBlock const& iEvent, edm::EventSetup const&) {}

// ------------ method called on each new Event  ------------
bool NumberPerLSFilter::filter(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  bool pass = true;
  counter_++;

  if (maxN_ < 0)
    return pass;
  if (counter_ > maxN_)
    pass = false;

  return pass;
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void NumberPerLSFilter::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.add<int>("maxN", 100);

  descriptions.add("numberPerLSFilter", desc);
}
//define this as a plug-in
DEFINE_FWK_MODULE(NumberPerLSFilter);
