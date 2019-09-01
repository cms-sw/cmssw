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
struct SpecificCaloMETData {
  SpecificCaloMETData()
      : MaxEtInEmTowers(0.0),
        MaxEtInHadTowers(0.0),
        HadEtInHO(0.0),
        HadEtInHB(0.0),
        HadEtInHF(0.0),
        HadEtInHE(0.0),
        EmEtInEB(0.0),
        EmEtInEE(0.0),
        EmEtInHF(0.0),
        EtFractionHadronic(0.0),
        EtFractionEm(0.0),
        METSignificance(0.0),
        CaloMETInpHF(0.0),
        CaloMETInmHF(0.0),
        CaloSETInpHF(0.0),
        CaloSETInmHF(0.0),
        CaloMETPhiInpHF(0.0),
        CaloMETPhiInmHF(0.0) {}

  float MaxEtInEmTowers;     // Maximum ET in EM towers
  float MaxEtInHadTowers;    // Maximum ET in HCAL towers
  float HadEtInHO;           // Hadronic ET fraction in HO
  float HadEtInHB;           // Hadronic ET in HB
  float HadEtInHF;           // Hadronic ET in HF
  float HadEtInHE;           // Hadronic ET in HE
  float EmEtInEB;            // Em ET in EB
  float EmEtInEE;            // Em ET in EE
  float EmEtInHF;            // Em ET in HF
  float EtFractionHadronic;  // Hadronic ET fraction
  float EtFractionEm;        // Em ET fraction
  float METSignificance;     // Em ET fraction
  float CaloMETInpHF;        // CaloMET in HF+
  float CaloMETInmHF;        // CaloMET in HF-
  float CaloSETInpHF;        // CaloSET in HF+
  float CaloSETInmHF;        // CaloSET in HF-
  float CaloMETPhiInpHF;     // CaloMET-phi in HF+
  float CaloMETPhiInmHF;     // CaloMET-phi in HF-
};

//____________________________________________________________________________||
#endif  // METReco_SpecificCaloMETData_h
