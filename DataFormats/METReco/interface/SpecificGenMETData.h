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
 ************************************************************/

struct SpecificGenMETData
{
  double m_EmEnergy;         // Event energy from EM particles
  double m_HadEnergy;        // Event energy from Hadronic particles
  double m_InvisibleEnergy;  // Event energy from neutrinos, etc
  double m_AuxiliaryEnergy;  // Event energy from undecayed particles
}; //public : struct SpecificGenMETData
#endif
