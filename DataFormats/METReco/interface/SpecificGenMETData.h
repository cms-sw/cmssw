// -*- C++ -*-

// Package:    METReco
// Class:      SpecificGenMETData
//
/** \class SpecificGenMETData

    SpecificGenMETData represents MET made from HEPMC particles
    Provide energy contributions from different particles in addition
    to generic MET parameters

*/
//  Authors:    Richard Cavanaugh, Ronald Remington
//


//____________________________________________________________________________||
#ifndef METReco_SpecificGenMETData_h
#define METReco_SpecificGenMETData_h

//____________________________________________________________________________||
struct SpecificGenMETData
{

  SpecificGenMETData()
    : NeutralEMEtFraction(0.0), NeutralHadEtFraction(0.0)
    , ChargedEMEtFraction(0.0), ChargedHadEtFraction(0.0)
    , MuonEtFraction(0.0), InvisibleEtFraction(0.0)
    , m_EmEnergy(0.0), m_HadEnergy(0.0)
    , m_InvisibleEnergy(0.0), m_AuxiliaryEnergy(0.0) { }

  float NeutralEMEtFraction;
  float NeutralHadEtFraction;
  float ChargedEMEtFraction;
  float ChargedHadEtFraction;
  float MuonEtFraction;
  float InvisibleEtFraction;

  //Old, obsolete datamembers (to be removed as soon as possible e.g 4_X_Y)
  float m_EmEnergy;         // Event energy from EM particles
  float m_HadEnergy;        // Event energy from Hadronic particles
  float m_InvisibleEnergy;  // Event energy from neutrinos, etc
  float m_AuxiliaryEnergy;  // Event energy from undecayed particles
};

//____________________________________________________________________________||
#endif // METReco_SpecificGenMETData_h
