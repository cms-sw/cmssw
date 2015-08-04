#ifndef IOMC_EventVertexGenerators_MixEvtVtxGenerator_h
#define IOMC_EventVertexGenerators_MixEvtVtxGenerator_h
/*
*/
#include "GeneratorInterface/Core/interface/BaseEvtVtxGenerator.h"

#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "FWCore/Utilities/interface/EDGetToken.h"

#include <memory>
#include <vector>

#include "TMatrixD.h"

namespace edm {
  class HepMCProduct;
}

template<typename T> class CrossingFrame;

class MixEvtVtxGenerator : public BaseEvtVtxGenerator {
public:

  // ctor & dtor
  explicit MixEvtVtxGenerator(edm::ParameterSet const& p, edm::ConsumesCollector& iC);
  virtual ~MixEvtVtxGenerator();

  MixEvtVtxGenerator(MixEvtVtxGenerator const&) = delete;
  MixEvtVtxGenerator& operator=(MixEvtVtxGenerator const&) = delete;

private:
  virtual void generateNewVertex_(edm::HepMCProduct& product, CLHEP::HepRandomEngine& engine) override;

  std::unique_ptr<HepMC::FourVector> fVertex;
  std::unique_ptr<TMatrixD> boost_;

  bool useRecVertex;
  std::vector<double> vtxOffset;
  bool useCF_;
};
#endif
