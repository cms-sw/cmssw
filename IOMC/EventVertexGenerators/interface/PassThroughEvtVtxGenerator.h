#ifndef IOMC_EventVertexGenerators_PassThroughEvtVtxGenerator_H
#define IOMC_EventVertexGenerators_PassThroughEvtVtxGenerator_H
/*
*/

#include "IOMC/EventVertexGenerators/interface/BaseEvtVtxGenerator.h"

#include "TMatrixD.h"

namespace CLHEP {
  class HepRandomEngine;
}

class PassThroughEvtVtxGenerator : public BaseEvtVtxGenerator {
public:
  // ctor & dtor
  explicit PassThroughEvtVtxGenerator(const edm::ParameterSet&);
  ~PassThroughEvtVtxGenerator() override;

  ROOT::Math::XYZTVector vertexShift(CLHEP::HepRandomEngine*) const override;

  TMatrixD const* GetInvLorentzBoost() const override { return nullptr; };

private:

};

#endif
