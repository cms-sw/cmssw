#include "Pythia8/ParticleDecays.h"
#include "Pythia8/Pythia.h"

//==========================================================================

// Specialized decayer for resonance decays to taus to allowing biasing to
// leptonic decays
//
class BiasedTauDecayer : public Pythia8::DecayHandler {
public:
  BiasedTauDecayer(const Pythia8::Info* infoPtr,
                   Pythia8::Settings* settingsPtr,
                   Pythia8::ParticleData* particleDataPtr,
                   Pythia8::Rndm* rndmPtr);

  bool decay(std::vector<int>& idProd,
             std::vector<double>& mProd,
             std::vector<Pythia8::Vec4>& pProd,
             int iDec,
             const Pythia8::Event& event) override;

private:
  Pythia8::TauDecays decayer;
  bool filter_;
  bool eMuDecays_;
  std::vector<int> idProdSave;
  std::vector<double> mProdSave;
  std::vector<Pythia8::Vec4> pProdSave;
};
