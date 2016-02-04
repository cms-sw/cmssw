#include "DataFormats/METReco/interface/CaloMET.h"

using namespace reco;

CaloMET::CaloMET()
{
  calo_data.MaxEtInEmTowers = 0.0;  // Maximum energy in EM towers
  calo_data.MaxEtInHadTowers = 0.0;        // Maximum energy in HCAL towers
  calo_data.HadEtInHO = 0.0;          // Hadronic energy fraction in HO
  calo_data.HadEtInHB = 0.0;          // Hadronic energy in HB
  calo_data.HadEtInHF = 0.0;          // Hadronic energy in HF
  calo_data.HadEtInHE = 0.0;          // Hadronic energy in HE
  calo_data.EmEtInEB = 0.0;           // Em energy in EB
  calo_data.EmEtInEE = 0.0;           // Em energy in EE
  calo_data.EmEtInHF = 0.0;           // Em energy in HF
  calo_data.EtFractionHadronic = 0.0; // Hadronic energy fraction
  calo_data.EtFractionEm = 0.0;       // Em energy fraction
  calo_data.METSignificance = -1.0;    // MET Significance
  calo_data.CaloSETInpHF = 0.0;        // CaloSET in HF+ 
  calo_data.CaloSETInmHF = 0.0;        // CaloSET in HF- 
  calo_data.CaloMETInpHF = 0.0;        // CaloMET in HF+ 
  calo_data.CaloMETInmHF = 0.0;        // CaloMET in HF- 
  calo_data.CaloMETPhiInpHF = 0.0;     // CaloMET-phi in HF+ 
  calo_data.CaloMETPhiInmHF = 0.0;     // CaloMET-phi in HF- 

}

bool CaloMET::overlap( const Candidate & ) const 
{
  return false;
}

