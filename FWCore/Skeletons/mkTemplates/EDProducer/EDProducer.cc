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
#include "FWCore/Framework/interface/stream/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/StreamID.h"

@example_myparticle#include "DataFormats/MuonReco/interface/Muon.h"
@example_myparticle#include "DataFormats/EgammaCandidates/interface/PixelMatchGsfElectron.h"
@example_myparticle#include "DataFormats/Candidate/interface/Particle.h"
@example_myparticle#include "FWCore/MessageLogger/interface/MessageLogger.h"
@example_myparticle#include "FWCore/Utilities/interface/InputTag.h"

//
// class declaration
//

class __class__ : public edm::stream::EDProducer<> {
public:
  explicit __class__(const edm::ParameterSet&);
  ~__class__();

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void beginStream(edm::StreamID) override;
  void produce(edm::Event&, const edm::EventSetup&) override;
  void endStream() override;

  //virtual void beginRun(edm::Run const&, edm::EventSetup const&) override;
  //virtual void endRun(edm::Run const&, edm::EventSetup const&) override;
  //virtual void beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override;
  //virtual void endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override;

  // ----------member data ---------------------------
@example_myparticle  edm::EDGetTokenT<reco::MuonCollection> muonToken_;
@example_myparticle  edm::EDGetTokenT<reco::PixelMatchGsfElectronCollection> electronToken_;
@example_myparticle  edm::EDPutTokenT<MyParticleCollection> putToken_;
};

//
// constants, enums and typedefs
//

@example_myparticle// define container that will be booked into event
@example_myparticletypedef std::vector<reco::Particle> MyParticleCollection;

//
// static data member definitions
//

//
// constructors and destructor
//
__class__::__class__(const edm::ParameterSet& iConfig)
@example_myparticle    : muonToken_(consumes<reco::MuonCollection>(iConfig.getParameter<edm::InputTag>("muons"))),
@example_myparticle      electronToken_(consumes<reco::PixelMatchGsfElectronCollection>(iConfig.getParameter<edm::InputTag>("electrons"))),
@example_myparticle      putToken_(produces<MyParticleCollection>("particles"))
{
  //register your products
/* Examples
  produces<ExampleData2>();

  //if do put with a label
  produces<ExampleData2>("label");
 
  //if you want to put into the Run
  produces<ExampleData2,InRun>();
*/
  //now do what ever other initialization is needed
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

// ------------ method called to produce the data  ------------
void __class__::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;
@example_myparticle  using namespace reco;
@example_myparticle  using namespace std;
/* This is an event example
  //Read 'ExampleData' from the Event
  ExampleData const& in = iEvent.get(inToken_);

  //Use the ExampleData to create an ExampleData2 which 
  // is put into the Event
  iEvent.put(std::make_unique<ExampleData2>(in));
*/

/* this is an EventSetup example
  //Read SetupData from the SetupRecord in the EventSetup
  SetupData& setup = iSetup.getData(setupToken_);
*/
@example_myparticle
@example_myparticle  auto const& muons = iEvent.get(muonToken_);
@example_myparticle  auto const& electrons = iEvent.get(electronToken_);
@example_myparticle
@example_myparticle  // create a new collection of Particle objects
@example_myparticle  auto newParticles = std::make_unique<MyParticleCollection>();
@example_myparticle
@example_myparticle  // if the number of electrons or muons is 4 (or 2 and 2), costruct a new particle
@example_myparticle  if (muons.size() == 4 || electrons.size() == 4 || (muons.size() == 2 && electrons.size() == 2)) {
@example_myparticle    // sums of momenta and charges will be calculated
@example_myparticle    Particle::LorentzVector totalP4(0, 0, 0, 0);
@example_myparticle    Particle::Charge charge(0);
@example_myparticle
@example_myparticle    // loop over muons, sum over p4s and charges. Later same for electrons
@example_myparticle    for (auto const& muon : muons) {
@example_myparticle      totalP4 += muon.p4();
@example_myparticle      charge += muon.charge();
@example_myparticle    }
@example_myparticle
@example_myparticle    for (auto const& electron : electrons) {
@example_myparticle      totalP4 += electron.p4();
@example_myparticle      charge += electron.charge();
@example_myparticle    }
@example_myparticle
@example_myparticle    // create a particle in the vector with momentum and charge from muons and electrons
@example_myparticle    newParticles->emplace_back(charge, totalP4);
@example_myparticle  }
@example_myparticle
@example_myparticle  // save the vector
@example_myparticle  iEvent.put(putToken_, move(newParticles));
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
@example_myparticle
@example_myparticle  //Specify that only 'muons' and 'electrons' are allowed
@example_myparticle  //To use, remove the default given above and uncomment below
@example_myparticle  //ParameterSetDescription desc;
@example_myparticle  //desc.add<edm::InputTag>("muons","muons");
@example_myparticle  //desc.add<edm::InputTag>("electrons","pixelMatchGsfElectrons");
@example_myparticle  //descriptions.addWithDefaultLabel(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(__class__);
