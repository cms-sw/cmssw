#include "Pythia8/ParticleDecays.h"
#include "Pythia8/Pythia.h"

//==========================================================================

// Specialized decayer for resonance decays to taus to allowing biasing to
// leptonic decays
//
class BiasedTauDecayer : public Pythia8::DecayHandler {
public:
  BiasedTauDecayer(Pythia8::Info* infoPtr,
                   Pythia8::Settings* settingsPtr,
                   Pythia8::ParticleData* particleDataPtr,
                   Pythia8::Rndm* rndmPtr,
                   Pythia8::Couplings* couplingsPtr);

  bool decay(std::vector<int>& idProd,
             std::vector<double>& mProd,
             std::vector<Pythia8::Vec4>& pProd,
             int iDec,
             const Pythia8::Event& event) override;

private:
  Pythia8::TauDecays decayer;
  bool filter_;
  bool eDecays_;
  bool muDecays_;
  std::vector<int> idProdSave;
  std::vector<double> mProdSave;
  std::vector<Pythia8::Vec4> pProdSave;
};
