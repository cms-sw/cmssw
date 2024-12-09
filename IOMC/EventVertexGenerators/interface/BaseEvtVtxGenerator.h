#ifndef IOMC_BaseEvtVtxGenerator_H
#define IOMC_BaseEvtVtxGenerator_H
/*
*/

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Utilities/interface/EDGetToken.h"

#include "Math/Vector4D.h"
#include "TMatrixD.h"

namespace HepMC {
  class FourVector;
}

namespace CLHEP {
  class HepRandomEngine;
}

namespace edm {
  class HepMCProduct;
  class HepMC3Product;
}  // namespace edm

class BaseEvtVtxGenerator : public edm::stream::EDProducer<> {
public:
  // ctor & dtor
  explicit BaseEvtVtxGenerator(const edm::ParameterSet&);
  ~BaseEvtVtxGenerator() override;

  void produce(edm::Event&, const edm::EventSetup&) override;

  virtual ROOT::Math::XYZTVector vertexShift(CLHEP::HepRandomEngine*) const = 0;

  virtual TMatrixD const* GetInvLorentzBoost() const = 0;

private:
  edm::EDGetTokenT<edm::HepMCProduct> sourceToken;
  edm::EDGetTokenT<edm::HepMC3Product> sourceToken3;
};

#endif
