// -*- C++ -*-
//
// Package:    LHE2HepMCConverter
// Class:      LHE2HepMCConverter
//
/**\class LHE2HepMCConverter LHE2HepMCConverter.cc GeneratorInterface/LHE2HepMCConverter/src/LHE2HepMCConverter.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Piergiulio Lenzi,40 1-B01,+41227671638,
//         Created:  Wed Aug 31 19:02:24 CEST 2011
//
//

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/Common/interface/Handle.h"

#include "SimDataFormats/GeneratorProducts/interface/LHEEventProduct.h"
#include "SimDataFormats/GeneratorProducts/interface/LHERunInfoProduct.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"

//
// class declaration
//

class LHE2HepMCConverter : public edm::EDProducer {
public:
  explicit LHE2HepMCConverter(const edm::ParameterSet&);
  ~LHE2HepMCConverter() override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

protected:
  //      lhef::LHERunInfo *lheRunInfo() { return lheRunInfo_.get(); }

private:
  void produce(edm::Event&, const edm::EventSetup&) override;
  void beginRun(edm::Run const&, edm::EventSetup const&) override;

  // ----------member data ---------------------------
  edm::InputTag _lheEventSrcTag;
  edm::InputTag _lheRunSrcTag;
  const LHERunInfoProduct* _lheRunSrc;

  //      std::shared_ptr<lhef::LHERunInfo> lheRunInfo_;
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
LHE2HepMCConverter::LHE2HepMCConverter(const edm::ParameterSet& iConfig) : _lheRunSrc(nullptr) {
  //register your products
  produces<edm::HepMCProduct>("unsmeared");

  _lheEventSrcTag = iConfig.getParameter<edm::InputTag>("LHEEventProduct");
  _lheRunSrcTag = iConfig.getParameter<edm::InputTag>("LHERunInfoProduct");
}

LHE2HepMCConverter::~LHE2HepMCConverter() {
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
}

//
// member functions
//

// ------------ method called to produce the data  ------------
void LHE2HepMCConverter::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;
  Handle<LHEEventProduct> lheEventSrc;
  iEvent.getByLabel(_lheEventSrcTag, lheEventSrc);

  HepMC::GenEvent* evt = new HepMC::GenEvent();
  HepMC::GenVertex* v = new HepMC::GenVertex();
  evt->add_vertex(v);
  if (_lheRunSrc) {
    HepMC::FourVector beam1(0, 0, _lheRunSrc->heprup().EBMUP.first, _lheRunSrc->heprup().EBMUP.first);
    HepMC::GenParticle* gp1 = new HepMC::GenParticle(beam1, _lheRunSrc->heprup().IDBMUP.first, 4);
    v->add_particle_in(gp1);
    HepMC::FourVector beam2(0, 0, _lheRunSrc->heprup().EBMUP.second, _lheRunSrc->heprup().EBMUP.second);
    HepMC::GenParticle* gp2 = new HepMC::GenParticle(beam2, _lheRunSrc->heprup().IDBMUP.second, 4);
    v->add_particle_in(gp2);
    evt->set_beam_particles(gp1, gp2);
  } else {
    LogWarning("LHE2HepMCConverter") << "Could not retrieve the LHERunInfoProduct for this event. You'll miss the beam "
                                        "particles in your HepMC product.";
  }

  for (int i = 0; i < lheEventSrc->hepeup().NUP; ++i) {
    if (lheEventSrc->hepeup().ISTUP[i] != 1) {
      //cout << reader->hepeup.ISTUP[i] << ", " << reader->hepeup.IDUP[i] << endl;
      continue;
    }
    HepMC::FourVector p(lheEventSrc->hepeup().PUP[i][0],
                        lheEventSrc->hepeup().PUP[i][1],
                        lheEventSrc->hepeup().PUP[i][2],
                        lheEventSrc->hepeup().PUP[i][3]);
    HepMC::GenParticle* gp = new HepMC::GenParticle(p, lheEventSrc->hepeup().IDUP[i], 1);
    gp->set_generated_mass(lheEventSrc->hepeup().PUP[i][4]);
    v->add_particle_out(gp);
  }

  std::unique_ptr<HepMCProduct> pOut(new HepMCProduct(evt));
  iEvent.put(std::move(pOut), "unsmeared");
}

// ------------ method called when starting to processes a run  ------------
void LHE2HepMCConverter::beginRun(edm::Run const& iRun, edm::EventSetup const&) {
  edm::Handle<LHERunInfoProduct> lheRunSrcHandle;
  iRun.getByLabel(_lheRunSrcTag, lheRunSrcHandle);
  if (lheRunSrcHandle.isValid()) {
    _lheRunSrc = lheRunSrcHandle.product();
  } else {
    if (_lheRunSrcTag.label() != "source") {
      iRun.getByLabel("source", lheRunSrcHandle);
      if (lheRunSrcHandle.isValid()) {
        _lheRunSrc = lheRunSrcHandle.product();
        edm::LogInfo("LHE2HepMCConverter") << "Taking LHERunInfoproduct from source";
      } else
        edm::LogWarning("LHE2HepMCConverter") << "No LHERunInfoProduct from source";
    }
  }
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void LHE2HepMCConverter::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(LHE2HepMCConverter);
