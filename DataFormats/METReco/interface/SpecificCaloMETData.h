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
  double MaxEtInEmTowers;         // Maximum ET in EM towers
  double MaxEtInHadTowers;        // Maximum ET in HCAL towers
  double HadEtInHO;          // Hadronic ET fraction in HO
  double HadEtInHB;          // Hadronic ET in HB
  double HadEtInHF;          // Hadronic ET in HF
  double HadEtInHE;          // Hadronic ET in HE
  double EmEtInEB;           // Em ET in EB
  double EmEtInEE;           // Em ET in EE
  double EmEtInHF;           // Em ET in HF
  double EtFractionHadronic; // Hadronic ET fraction
  double EtFractionEm;       // Em ET fraction
  double METSignificance;       // Em ET fraction
  double CaloMETInpHF;         // CaloMET in HF+ 
  double CaloMETInmHF;         // CaloMET in HF- 
  double CaloMETInpHE;         // CaloMET in HE+ 
  double CaloMETInmHE;         // CaloMET in HE- 
  double CaloMETInpHB;         // CaloMET in HB+ 
  double CaloMETInmHB;         // CaloMET in HB- 
  double CaloMETPhiInpHF;         // CaloMET-phi in HF+ 
  double CaloMETPhiInmHF;         // CaloMET-phi in HF- 
  double CaloMETPhiInpHE;         // CaloMET-phi in HE+ 
  double CaloMETPhiInmHE;         // CaloMET-phi in HE- 
  double CaloMETPhiInpHB;         // CaloMET-phi in HB+ 
  double CaloMETPhiInmHB;         // CaloMET-phi in HB- 
}; //public : struct SpecificCaloMETData
#endif
