#include "Pythia8/Pythia.h"
#include "GeneratorInterface/Pythia8Interface/interface/BiasedTauDecayer.h"
using namespace Pythia8;

//==========================================================================

// Specialized decayer for resonance decays to taus to allowing biasing to
// leptonic decays
//

BiasedTauDecayer::BiasedTauDecayer(const Info* infoPtr,
                                   Settings* settingsPtr,
                                   ParticleData* particleDataPtr,
                                   Rndm* rndmPtr) {
  decayer = TauDecays();
  decayer.init();
  filter_ = settingsPtr->flag("BiasedTauDecayer:filter");
  eMuDecays_ = settingsPtr->flag("BiasedTauDecayer:eMuDecays");
}

bool BiasedTauDecayer::decay(
    std::vector<int>& idProd, std::vector<double>& mProd, std::vector<Vec4>& pProd, int iDec, const Event& event) {
  if (!filter_)
    return false;
  if (idProd[0] != 15 && idProd[0] != -15)
    return false;
  int iStart = event[iDec].iTopCopyId();
  int iMom = event[iStart].mother1();
  int idMom = event[iMom].idAbs();
  if (idMom != 23 && idMom != 24 && idMom != 25)
    return false;
  int iDau1 = event[iMom].daughter1();
  int iDau2 = event[iMom].daughter2();
  int iBot1 = event[iDau1].iBotCopyId();
  int iBot2 = event[iDau2].iBotCopyId();
  int iDecSis = (iDec == iBot1) ? iBot2 : iBot1;
  // Check if sister has been decayed
  // Since taus decays are correlated, use one decay, store the other
  bool notDecayed = event[iDecSis].status() > 0 ? true : false;
  if (notDecayed) {
    // bias for leptonic decays
    bool hasLepton = (eMuDecays_) ? false : true;
    Event decay;
    int i1 = -1;
    int i2 = -1;
    while (!hasLepton) {
      decay = event;
      decayer.decay(iDec, decay);
      // check for lepton in first decay
      i1 = decay[iDec].daughter1();
      i2 = decay[iDec].daughter2();
      for (int i = i1; i < i2 + 1; ++i) {
        if (decay[i].isLepton() && decay[i].isCharged()) {
          hasLepton = true;
          break;
        }
      }
      if (hasLepton)
        break;
      // check for lepton in second decay
      i1 = decay[iDecSis].daughter1();
      i2 = decay[iDecSis].daughter2();
      for (int i = i1; i < i2 + 1; ++i) {
        if (decay[i].isLepton() && decay[i].isCharged()) {
          hasLepton = true;
          break;
        }
      }
    }
    // Return decay products
    i1 = decay[iDec].daughter1();
    i2 = decay[iDec].daughter2();
    for (int i = i1; i < i2 + 1; ++i) {
      idProd.push_back(decay[i].id());
      mProd.push_back(decay[i].m());
      pProd.push_back(decay[i].p());
    }
    // Store correlated decay products
    i1 = decay[iDecSis].daughter1();
    i2 = decay[iDecSis].daughter2();
    idProdSave.clear();
    mProdSave.clear();
    pProdSave.clear();
    for (int i = i1; i < i2 + 1; ++i) {
      idProdSave.push_back(decay[i].id());
      mProdSave.push_back(decay[i].m());
      pProdSave.push_back(decay[i].p());
    }
  } else {
    // Return stored decay products
    for (size_t i = 0; i < idProdSave.size(); ++i) {
      idProd.push_back(idProdSave[i]);
      mProd.push_back(mProdSave[i]);
      pProd.push_back(pProdSave[i]);
    }
  }

  return true;
}
