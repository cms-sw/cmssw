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
  float NeutralEMFraction;
  float NeutralHadFraction;
  float ChargedEMFraction;
  float ChargedHadFraction;
  float MuonFraction;
  float Type6Fraction;
  float Type7Fraction;

  // float NeutralEMEtFraction;
  // float NeutralHadEtFraction;
  // float ChargedEMEtFraction;
  // float ChargedHadEtFraction;
  // float MuonEtFraction;
  // float Type6EtFraction;
  // float Type7EtFraction;

};

//____________________________________________________________________________||
#endif // METReco_SpecificPFMETData_h
