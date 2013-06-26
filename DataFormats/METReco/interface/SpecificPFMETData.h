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
  // Data Members (should be renamed with "Et" in them to avoid ambiguities, see below)
  double NeutralEMFraction ; 
  double NeutralHadFraction ;
  double ChargedEMFraction ; 
  double ChargedHadFraction ;
  double MuonFraction ;
  double Type6Fraction;
  double Type7Fraction;

  /*
  double NeutralEMEtFraction ; 
  double NeutralHadEtFraction ;
  double ChargedEMEtFraction ; 
  double ChargedHadEtFraction ;
  double MuonEtFraction ;
  double Type6EtFraction;
  double Type7EtFraction;
  */
  

}; //public : struct SpecificPFMETData
#endif
