// -*- C++ -*-
#ifndef METReco_SpecificPFMETData_h
#define METReco_SpecificPFMETData_h

/// \class SpecificPFMETData
///
/// \short MET made from Particle Flow Candidates
///
/// \authors    R. Cavanaugh, UIC & R.Remington, UFL

//____________________________________________________________________________||
struct SpecificPFMETData
{
  SpecificPFMETData() : NeutralEMFraction(0.0), NeutralHadFraction(0.0)
		      , ChargedEMFraction(0.0), ChargedHadFraction(0.0)
		      , MuonFraction(0.0)
		      , Type6Fraction(0.0), Type7Fraction(0.0) { }

  // Data Members (should be renamed with "Et" in them to avoid ambiguities, see below)
  double NeutralEMFraction;
  double NeutralHadFraction;
  double ChargedEMFraction;
  double ChargedHadFraction;
  double MuonFraction;
  double Type6Fraction;
  double Type7Fraction;

  // double NeutralEMEtFraction;
  // double NeutralHadEtFraction;
  // double ChargedEMEtFraction;
  // double ChargedHadEtFraction;
  // double MuonEtFraction;
  // double Type6EtFraction;
  // double Type7EtFraction;

};

//____________________________________________________________________________||
#endif // METReco_SpecificPFMETData_h
