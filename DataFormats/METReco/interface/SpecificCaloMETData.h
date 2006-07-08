#ifndef METReco_SpecificCaloMETData_h
#define METReco_SpecificCaloMETData_h

/** \class SpecificCaloMETData
 *
 * \short MET made from CaloTowers
 *
 * SpecificCaloMETData represents MET made from CaloTowers
 * Provide energy contributions from different subdetectors
 * in addition to generic MET parameters
 *
 * \author    R. Cavanaugh, UFL
 *
 ************************************************************/

struct SpecificCaloMETData
{
  double mMaxEInEmTowers;         // Maximum energy in EM towers
  double mMaxEInHadTowers;        // Maximum energy in HCAL towers
  double mHadEnergyInHO;          // Hadronic energy fraction in HO
  double mHadEnergyInHB;          // Hadronic energy in HB
  double mHadEnergyInHF;          // Hadronic energy in HF
  double mHadEnergyInHE;          // Hadronic energy in HE
  double mEmEnergyInEB;           // Em energy in EB
  double mEmEnergyInEE;           // Em energy in EE
  double mEmEnergyInHF;           // Em energy in HF
  double mEnergyFractionHadronic; // Hadronic energy fraction
  double mEnergyFractionEm;       // Em energy fraction
}; //public : struct SpecificCaloMETData
#endif
