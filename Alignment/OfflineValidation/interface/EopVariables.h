#ifndef Alignment_OfflineValidation_EopVariables_h
#define Alignment_OfflineValidation_EopVariables_h

// For ROOT types with '_t':
#include <Rtypes.h>

/// container to hold data to be written into TTree
struct EopVariables
{
  /// constructor initialises to empty values
  EopVariables()  {this->clear();}
  
  /// set to empty values
  void clear() {
    /// doubles
    track_outerRadius = track_chi2 = track_normalizedChi2 = track_p = track_pt = 
      track_ptError = track_theta = track_eta = track_phi = track_emc1 = track_emc3 = 
      track_emc5 = track_hac1 = track_hac3 = track_hac5 = track_maxPNearby = 
      track_EnergyIn = track_EnergyOut = distofmax = 0.;
    /// integers
    track_charge = track_nHits = track_nLostHits = track_innerOk = 0;

  }
  /// fill variables into tree
  void fillVariables(Int_t charge, Int_t innerOk, Double_t outerRadius, Int_t numberOfValidHits, 
		     Int_t numberOfLostHits, Double_t chi2, Double_t normalizedChi2, Double_t p, 
		     Double_t pt, Double_t ptError, Double_t theta, Double_t eta, Double_t phi, 
		     Double_t emc1, Double_t emc3, Double_t emc5, Double_t hac1, Double_t hac3, 
		     Double_t hac5, Double_t maxPNearby, Double_t dist, Double_t EnergyIn, 
		     Double_t EnergyOut) {
    track_charge = charge; track_nHits = numberOfValidHits; track_nLostHits = numberOfLostHits;  
    track_innerOk = innerOk; track_outerRadius = outerRadius; track_chi2 = chi2; 
    track_normalizedChi2 = normalizedChi2; track_p = p; track_pt = pt; track_ptError = ptError; 
    track_theta = theta; track_eta = eta; track_phi = phi; track_emc1 = emc1; track_emc3 = emc3; 
    track_emc5 = emc5; track_hac1 = hac1; track_hac3 = hac3; track_hac5 = hac5;
    track_maxPNearby = maxPNearby; track_EnergyIn = EnergyIn; track_EnergyOut = EnergyOut;
    distofmax = dist;
  }
  
  Double_t track_outerRadius, track_chi2, track_normalizedChi2, track_p, track_pt, track_ptError, 
    track_theta, track_eta, track_phi, track_emc1, track_emc3, track_emc5, track_hac1, track_hac3, 
    track_hac5, track_maxPNearby, track_EnergyIn, track_EnergyOut, distofmax;
  
  Int_t track_charge, track_nHits, track_nLostHits, track_innerOk;

};

#endif
