// -*- C++ -*-

// Package:    METReco
// Class:      SpecificCaloMETData
//
/** \class SpecificCaloMETData

    SpecificCaloMETData represents MET made from CaloTowers Provide
    energy contributions from different subdetectors in addition to
    generic MET parameters

*/
//
// Authors:    R. Cavanaugh, UFL
//

//____________________________________________________________________________||
#ifndef METReco_SpecificCaloMETData_h
#define METReco_SpecificCaloMETData_h

//____________________________________________________________________________||
struct SpecificCaloMETData
{
  SpecificCaloMETData()
    : MaxEtInEmTowers(0.0), MaxEtInHadTowers(0.0)
    , HadEtInHO(0.0), HadEtInHB(0.0),  HadEtInHF(0.0), HadEtInHE(0.0)
    , EmEtInEB(0.0), EmEtInEE(0.0), EmEtInHF(0.0), EtFractionHadronic(0.0)
    , EtFractionEm(0.0), METSignificance(0.0), CaloMETInpHF(0.0)
    , CaloMETInmHF(0.0), CaloSETInpHF(0.0), CaloSETInmHF(0.0)
    , CaloMETPhiInpHF(0.0), CaloMETPhiInmHF(0.0) { }

  double MaxEtInEmTowers;    // Maximum ET in EM towers
  double MaxEtInHadTowers;   // Maximum ET in HCAL towers
  double HadEtInHO;          // Hadronic ET fraction in HO
  double HadEtInHB;          // Hadronic ET in HB
  double HadEtInHF;          // Hadronic ET in HF
  double HadEtInHE;          // Hadronic ET in HE
  double EmEtInEB;           // Em ET in EB
  double EmEtInEE;           // Em ET in EE
  double EmEtInHF;           // Em ET in HF
  double EtFractionHadronic; // Hadronic ET fraction
  double EtFractionEm;       // Em ET fraction
  double METSignificance;    // Em ET fraction
  double CaloMETInpHF;       // CaloMET in HF+
  double CaloMETInmHF;       // CaloMET in HF-
  double CaloSETInpHF;       // CaloSET in HF+
  double CaloSETInmHF;       // CaloSET in HF-
  double CaloMETPhiInpHF;    // CaloMET-phi in HF+
  double CaloMETPhiInmHF;    // CaloMET-phi in HF-

};

//____________________________________________________________________________||
#endif // METReco_SpecificCaloMETData_h
