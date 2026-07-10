#ifndef IOMC_BaseEvtVtxGenerator_H
#define IOMC_BaseEvtVtxGenerator_H
/*
*/

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMCProductFwd.h"

#include "Math/Vector4D.h"
#include "TMatrixD.h"

namespace HepMC {
  class FourVector;
}

namespace CLHEP {
  class HepRandomEngine;
}

namespace edm {
  class HepMC3Product;
}  // namespace edm

template <typename... T>
class BaseEvtVtxGeneratorT : public edm::stream::EDProducer<T...> {
public:
  // ctor & dtor
  explicit BaseEvtVtxGeneratorT(const edm::ParameterSet&);
  ~BaseEvtVtxGeneratorT() override;

  void produce(edm::Event&, const edm::EventSetup&) override;

  virtual ROOT::Math::XYZTVector vertexShift(CLHEP::HepRandomEngine*) const = 0;

  virtual TMatrixD const* GetInvLorentzBoost() const = 0;

private:
  edm::EDGetTokenT<edm::HepMCProduct> sourceToken;
  edm::EDGetTokenT<edm::HepMC3Product> sourceToken3;
};

using BaseEvtVtxGenerator = BaseEvtVtxGeneratorT<>;
using BaseEvtVtxGeneratorWithLumi = BaseEvtVtxGeneratorT<edm::stream::WatchLuminosityBlocks>;

#endif
