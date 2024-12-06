/*
*/

#include "IOMC/EventVertexGenerators/interface/BaseEvtVtxGenerator.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMC3Product.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"

#include "FWCore/Utilities/interface/Exception.h"

#include "DataFormats/Provenance/interface/Provenance.h"
#include "FWCore/Utilities/interface/EDMException.h"

using namespace edm;
using namespace CLHEP;

BaseEvtVtxGenerator::BaseEvtVtxGenerator(const ParameterSet& pset) {
  Service<RandomNumberGenerator> rng;
  if (!rng.isAvailable()) {
    throw cms::Exception("Configuration") << "The BaseEvtVtxGenerator requires the RandomNumberGeneratorService\n"
                                             "which is not present in the configuration file. \n"
                                             "You must add the service\n"
                                             "in the configuration file or remove the modules that require it.";
  }

  sourceToken3 = consumes<edm::HepMC3Product>(pset.getParameter<edm::InputTag>("src"));
  sourceToken = consumes<edm::HepMCProduct>(pset.getParameter<edm::InputTag>("src"));
  produces<edm::HepMC3Product>();
  produces<edm::HepMCProduct>();
}

BaseEvtVtxGenerator::~BaseEvtVtxGenerator() {}

void BaseEvtVtxGenerator::produce(Event& evt, const EventSetup&) {
  edm::Service<edm::RandomNumberGenerator> rng;
  CLHEP::HepRandomEngine* engine = &rng->getEngine(evt.streamID());

  Handle<HepMCProduct> HepUnsmearedMCEvt;

  bool found = evt.getByToken(sourceToken, HepUnsmearedMCEvt);

  if (found) {  // HepMC event exists

    // Make a copy
    HepMC::GenEvent* genevt = new HepMC::GenEvent(*HepUnsmearedMCEvt->GetEvent());

    std::unique_ptr<edm::HepMCProduct> HepMCEvt(new edm::HepMCProduct(genevt));
    // generate new vertex & apply the shift
    //
    ROOT::Math::XYZTVector VertexShift = vertexShift(engine);
    HepMCEvt->applyVtxGen(HepMC::FourVector(VertexShift.x(), VertexShift.y(), VertexShift.z(), VertexShift.t()));

    HepMCEvt->boostToLab(GetInvLorentzBoost(), "vertex");
    HepMCEvt->boostToLab(GetInvLorentzBoost(), "momentum");

    evt.put(std::move(HepMCEvt));

  } else {  // no HepMC event, try to get HepMC3 event

    Handle<HepMC3Product> HepUnsmearedMCEvt3;
    found = evt.getByToken(sourceToken3, HepUnsmearedMCEvt3);

    if (!found)
      throw cms::Exception("ProductAbsent") << "No HepMCProduct, tried to get HepMC3Product, but it is also absent.";

    HepMC3::GenEvent* genevt3 = new HepMC3::GenEvent();
    genevt3->read_data(*HepUnsmearedMCEvt3->GetEvent());
    HepMC3Product* productcopy3 = new HepMC3Product(genevt3);
    ROOT::Math::XYZTVector VertexShift = vertexShift(engine);
    productcopy3->applyVtxGen(HepMC3::FourVector(VertexShift.x(), VertexShift.y(), VertexShift.z(), VertexShift.t()));

    if (GetInvLorentzBoost() != nullptr) {
      TMatrixD tmplorentz(*GetInvLorentzBoost());
      TMatrixD p4(4, 1);
      p4(0, 0) = 1.;
      p4(1, 0) = 1.;
      p4(2, 0) = 1.;
      p4(3, 0) = 1.;  // Check if the boost matrix is not trivial
      TMatrixD p4lab(4, 1);
      p4lab = tmplorentz * p4;
      if (p4lab(0, 0) - p4(0, 0) != 0. || p4lab(1, 0) - p4(1, 0) != 0. || p4lab(2, 0) - p4(2, 0) != 0. ||
          p4lab(3, 0) - p4(3, 0) != 0.) {  // not trivial:
        productcopy3->boostToLab(GetInvLorentzBoost(), "vertex");
        productcopy3->boostToLab(GetInvLorentzBoost(), "momentum");
      }
    }

    std::unique_ptr<edm::HepMC3Product> HepMC3Evt(productcopy3);
    evt.put(std::move(HepMC3Evt));
  }

  return;
}
