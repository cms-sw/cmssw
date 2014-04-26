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
    , MuonEtFraction(0.0), InvisibleEtFraction(0.0) { }

  double NeutralEMEtFraction;
  double NeutralHadEtFraction;
  double ChargedEMEtFraction;
  double ChargedHadEtFraction;
  double MuonEtFraction;
  double InvisibleEtFraction;

  //Old, obsolete datamembers (to be removed as soon as possible e.g 4_X_Y)
  double m_EmEnergy;         // Event energy from EM particles
  double m_HadEnergy;        // Event energy from Hadronic particles
  double m_InvisibleEnergy;  // Event energy from neutrinos, etc
  double m_AuxiliaryEnergy;  // Event energy from undecayed particles
};

//____________________________________________________________________________||
#endif // METReco_SpecificGenMETData_h
