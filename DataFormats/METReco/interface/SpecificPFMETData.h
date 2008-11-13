#ifndef METReco_SpecificPFMETData_h
#define METReco_SpecificPFMETData_h

/** \class SpecificPFMETData
 *
 * \short MET made from Particle Flow Candidates
 *
 * SpecificPFMETData represents MET-related quantities coming from Particle Flow Candidates
 * in addition to generic MET parameters
 *
 * \authors    R. Cavanaugh, UIC & R.Remington, UFL
 *
 ************************************************************/

struct SpecificPFMETData
{
  double NeutralEMFraction ; 
  double NeutralHadFraction ;
  double ChargedEMFraction ; 
  double ChargedHadFraction ;
  double MuonFraction ;
}; //public : struct SpecificPFMETData
#endif
