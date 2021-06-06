// -*- C++ -*-
//
// Package:    __subsys__/__pkgname__
// Class:      __class__
//
/**\class __class__ __class__.cc __subsys__/__pkgname__/plugins/__class__.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  __author__
//         Created:  __date__
//
//

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDFilter.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/StreamID.h"

//
// class declaration
//

class __class__ : public edm::stream::EDFilter<> {
public:
  explicit __class__(const edm::ParameterSet&);
  ~__class__();

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  virtual void beginStream(edm::StreamID) override;
  virtual bool filter(edm::Event&, const edm::EventSetup&) override;
  virtual void endStream() override;

  //virtual void beginRun(edm::Run const&, edm::EventSetup const&) override;
  //virtual void endRun(edm::Run const&, edm::EventSetup const&) override;
  //virtual void beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override;
  //virtual void endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override;

  // ----------member data ---------------------------
#ifdef THIS_IS_AN_EVENT_EXAMPLE
  edm::EDGetTokenT<ExampleData> exampleToken_;
#endif
#ifdef THIS_IS_AN_EVENTSETUP_EXAMPLE
  edm::ESGetToken<SetupData, SetupRecord> setupToken_;
#endif
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
__class__::__class__(const edm::ParameterSet& iConfig) {
  //now do what ever initialization is needed
#ifdef THIS_IS_AN_EVENT_EXAMPLE
  exampleToken_ = consumes<ExampleData>(iConfig.getParameter<edm::InputTag>("examples"));
#endif
#ifdef THIS_IS_EN_EVENTSETUP_EXAMPLE
  setupToken_ = esConsumes<SetupData, SetupRecord>();
#endif
}

__class__::~__class__() {
  // do anything here that needs to be done at destruction time
  // (e.g. close files, deallocate resources etc.)
  //
  // please remove this method altogether if it would be left empty
}

//
// member functions
//

// ------------ method called on each new Event  ------------
bool __class__::filter(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;
#ifdef THIS_IS_AN_EVENT_EXAMPLE
  ExampleData const& in = iEvent.get(exampleToken_);
#endif

#ifdef THIS_IS_AN_EVENTSETUP_EXAMPLE
  SetupData const& setup = iSetup.getData(setupToken_);
#endif
  return true;
}

// ------------ method called once each stream before processing any runs, lumis or events  ------------
void __class__::beginStream(edm::StreamID) {
  // please remove this method if not needed
}

// ------------ method called once each stream after processing all runs, lumis and events  ------------
void __class__::endStream() {
  // please remove this method if not needed
}

// ------------ method called when starting to processes a run  ------------
/*
void
__class__::beginRun(edm::Run const&, edm::EventSetup const&)
{ 
}
*/

// ------------ method called when ending the processing of a run  ------------
/*
void
__class__::endRun(edm::Run const&, edm::EventSetup const&)
{
}
*/

// ------------ method called when starting to processes a luminosity block  ------------
/*
void
__class__::beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&)
{
}
*/

// ------------ method called when ending the processing of a luminosity block  ------------
/*
void
__class__::endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&)
{
}
*/

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void __class__::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}
//define this as a plug-in
DEFINE_FWK_MODULE(__class__);
