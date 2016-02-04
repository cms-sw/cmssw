#ifndef METReco_SpecificGenMETData_h
#define METReco_SpecificGenMETData_h

/** \class SpecificGenMETData
 *
 * \short MET made from CaloTowers
 *
 * SpecificGenMETData represents MET made from HEPMC particles
 * Provide energy contributions from different particles
 * in addition to generic MET parameters
 *
 * \author    R. Cavanaugh, UFL
 *
 * 
 ************************************************************/

/*
Revision: Sept. 29, 2009
Author : Ronald Remington
Notes:  Changed names of data members to align with those in PFMET.  Should be integrated in CMSSW_3_4_X.
*/

struct SpecificGenMETData
{
  double NeutralEMEtFraction ;
  double NeutralHadEtFraction ;
  double ChargedEMEtFraction ;
  double ChargedHadEtFraction ;
  double MuonEtFraction ;
  double InvisibleEtFraction ;

  //Old, obsolete datamembers (to be removed as soon as possible e.g 4_X_Y)
  double m_EmEnergy;         // Event energy from EM particles
  double m_HadEnergy;        // Event energy from Hadronic particles
  double m_InvisibleEnergy;  // Event energy from neutrinos, etc
  double m_AuxiliaryEnergy;  // Event energy from undecayed particles



}; //public : struct SpecificGenMETData
#endif
