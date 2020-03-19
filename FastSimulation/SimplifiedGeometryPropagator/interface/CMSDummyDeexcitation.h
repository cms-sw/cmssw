#ifndef FASTSIM_CMSDUMMYDEEXCITATION_H
#define FASTSIM_CMSDUMMYDEEXCITATION_H

///////////////////////////////////////////////
// CMSDummyDeexcitation
//
// Description: Needed as a dummy interface to Geant4 nuclear de-excitation module; no secondary produced
//
// Author: Vladimir Ivanchenko
// Date: 20 Jan 2015
//
// Revision: Class structure modified to match SimplifiedGeometryPropagator
//           S. Kurz, 29 May 2017
//////////////////////////////////////////////////////////

#include "G4VPreCompoundModel.hh"
#include "G4ReactionProductVector.hh"

class G4Fragment;
class G4HadFinalState;
class G4HadProjectile;
class G4Nucleus;

namespace fastsim {
  //! Needed as a dummy interface to Geant4 nuclear de-excitation module.
  /*!
        No secondary produced.
    */
  class CMSDummyDeexcitation : public G4VPreCompoundModel {
  public:
    CMSDummyDeexcitation() : G4VPreCompoundModel(nullptr, "PRECO"){};
    ~CMSDummyDeexcitation() override{};
    G4HadFinalState* ApplyYourself(const G4HadProjectile&, G4Nucleus&) override { return nullptr; };
    G4ReactionProductVector* DeExcite(G4Fragment&) override { return new G4ReactionProductVector(); };
    void DeExciteModelDescription(std::ostream& outFile) const override { return; };
  };
}  // namespace fastsim

#endif
