#ifndef IOMC_EventVertexGenerators_PassThroughEvtVtxGenerator_H
#define IOMC_EventVertexGenerators_PassThroughEvtVtxGenerator_H
/*
*/

#include "IOMC/EventVertexGenerators/interface/BaseEvtVtxGenerator.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Utilities/interface/EDGetToken.h"

#include "TMatrixD.h"

namespace HepMC {
  class FourVector;
}

namespace CLHEP {
  class HepRandomEngine;
}

namespace edm {
  class HepMCProduct;
}

class PassThroughEvtVtxGenerator : public BaseEvtVtxGenerator {
public:
  // ctor & dtor
  explicit PassThroughEvtVtxGenerator(const edm::ParameterSet&);
  ~PassThroughEvtVtxGenerator() override;

  void produce(edm::Event&, const edm::EventSetup&) override;

  HepMC::FourVector newVertex(CLHEP::HepRandomEngine*) const override;

  TMatrixD const* GetInvLorentzBoost() const override { return nullptr; };

private:
  edm::EDGetTokenT<edm::HepMCProduct> sourceToken;
};

#endif
