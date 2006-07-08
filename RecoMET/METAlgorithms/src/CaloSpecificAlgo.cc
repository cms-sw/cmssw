#include "RecoMET/METAlgorithms/interface/CaloSpecificAlgo.h"

using namespace reco;
using namespace std;

reco::CaloMET CaloSpecificAlgo::addInfo(CommonMETData met)
{
  SpecificCaloMETData specific;
  specific.mMaxEInEmTowers = 1.0;         // Maximum energy in EM towers
  specific.mMaxEInHadTowers = 1.0;        // Maximum energy in HCAL towers
  specific.mHadEnergyInHO = 1.0;          // Hadronic energy fraction in HO
  specific.mHadEnergyInHB = 1.0;          // Hadronic energy in HB
  specific.mHadEnergyInHF = 1.0;          // Hadronic energy in HF
  specific.mHadEnergyInHE = 1.0;          // Hadronic energy in HE
  specific.mEmEnergyInEB = 1.0;           // Em energy in EB
  specific.mEmEnergyInEE = 1.0;           // Em energy in EE
  specific.mEmEnergyInHF = 1.0;           // Em energy in HF
  specific.mEnergyFractionHadronic = 1.0; // Hadronic energy fraction
  specific.mEnergyFractionEm = 1.0;       // Em energy fraction
  CaloMET specificmet(specific, met);
  return specificmet;
}
