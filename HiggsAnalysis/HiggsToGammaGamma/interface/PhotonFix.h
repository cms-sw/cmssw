#ifndef PhotonFix_Defined_hh
#define PhotonFix_Defined_hh

//-------------------------------------------------------//
// Project:  PhotonFix
// Author:   Paul Dauncey (p.dauncey@imperial.ac.uk)
// Modified: 11/07/2011
// Admins:   Paul Dauncey (p.dauncey@imperial.ac.uk)
//           Matt Kenzie (matthew.william.kenzie@cern.ch)
//-------------------------------------------------------//

/*
  Does post-reco fixes to ECAL photon energy and estimates resolution.

  Before instantiating any objects of PhotonFix, the constants must be
  initialised in the first event using
    PhotonFix::initialise("3_8");
  
  The string gives the reco version used. Valid strings are 
  "3_8", "3_11", "4_2" and "Nominal", where the latter gives no correction 
  to the energy and a nominal resolution value. There is also "4_2e" which 
  provides corrections for electrons which are reconstructed as photons (to
  aid with testing the performance of these corrections in data).

  Make objects using
    PhotonFixCMS a(p);
  where p is a reco::Photon reference 

  Get the corrected energy using
    a.fixedEnergy();
  and the resolution using
    a.sigmaEnergy();

*/

#include <iostream>
#include <string>

#ifndef LOCAL_FITTING_PROCEDURE
#include "DataFormats/EgammaCandidates/interface/Photon.h"
#endif

class PhotonFix {
 public:
  PhotonFix(double e, double eta, double phi, double r9, double aC, double aS, double aM, double bC, double bS, double bM);

#ifndef LOCAL_FITTING_PROCEDURE
  PhotonFix(const reco::Photon &p);
  
  // Must be called before instantiating any PhotonFix objects
  static bool initialiseParameters(const edm::ParameterSet &iConfig);
  static bool initialiseGeometry(const edm::EventSetup &iSetup);
#endif

  // Find distance of photon from cracks between crystals and module boundaries 
  void setup(bool doGeom);
  
  // Corrected energy and sigma
  double fixedEnergy() const;
  double sigmaEnergy() const;
  
  // Input values
  double rawEnergy() const;
  double eta() const;
  double phi() const;
  double r9() const;

  // Derived EB crystal, submodule and module relative coordinates
  double etaC() const;
  double etaS() const;
  double etaM() const;

  double phiC() const;
  double phiS() const;
  double phiM() const;

  // Derived EE zeta, crystal, subcrystal and D-module relative coordinates
  double xZ() const;
  double xC() const;
  double xS() const;
  double xM() const;

  double yZ() const;
  double yC() const;
  double yS() const;
  double yM() const;

  // Return values for Paul's fit
  double aC() const;
  double aS() const;
  double aM() const;
  double bC() const;
  double bS() const;
  double bM() const;

  // Return arrays containing positions of ecal gaps
  static void barrelCGap(unsigned i, unsigned j, unsigned k, double c);
  static void barrelSGap(unsigned i, unsigned j, unsigned k, double c);
  static void barrelMGap(unsigned i, unsigned j, unsigned k, double c);
  static void endcapCrystal(unsigned i, unsigned j, bool c); 
  static void endcapCGap(unsigned i, unsigned j, unsigned k, double c);
  static void endcapSGap(unsigned i, unsigned j, unsigned k, double c);
  static void endcapMGap(unsigned i, unsigned j, unsigned k, double c);
  
  void print() const;

  // Input and output the fit parameters
  static void setParameters(unsigned be, unsigned hl, const double *p);
  static void getParameters(unsigned be, unsigned hl, double *p);

  // Utility functions
  static void dumpConfigParameters(std::ostream &o);
  static void readConfigParameters(std::istream &i);
  static void dumpParameters(std::ostream &o);
  static void printParameters(std::ostream &o);
  static void dumpGaps(std::ostream &o);
  static double asinh(double s);  
 
 private:

  // Utility functions
  static double dPhi(double f0, double f1);
  static double aPhi(double f0, double f1);

  static double expCorrection(double a, const double *p);
  static double gausCorrection(double a, const double *p);

  // Actual data for each instantiated object
  unsigned _be,_hl;
  double _e,_eta,_phi,_r9;
  double _aC,_aS,_aM,_bC,_bS,_bM;
  
  // Constants
  static const double _onePi;
  static const double _twoPi;
  
  // Initialisation flags
  static bool _initialisedParams;
  static bool _initialisedGeom;
  
  // Parameters for fixes
  static double _meanScale[2][2][4];
  static double _meanAT[2][2][4];
  static double _meanAC[2][2][4];
  static double _meanAS[2][2][4];
  static double _meanAM[2][2][4];
  static double _meanBT[2][2][4];
  static double _meanBC[2][2][4];
  static double _meanBS[2][2][4];
  static double _meanBM[2][2][4];
  static double _meanR9[2][2][4];
  
  // Parameters for resolution
  static double _sigmaScale[2][2][4];
  static double _sigmaAT[2][2][4];
  static double _sigmaAC[2][2][4];
  static double _sigmaAS[2][2][4];
  static double _sigmaAM[2][2][4];
  static double _sigmaBT[2][2][4];
  static double _sigmaBC[2][2][4];
  static double _sigmaBS[2][2][4];
  static double _sigmaBM[2][2][4];
  static double _sigmaR9[2][2][4];
  
  // EB gap positions
  static double _barrelCGap[169][360][2];
  static double _barrelSGap[33][180][2];
  static double _barrelMGap[7][18][2];
  
  // EE crystal existence and gap positions
  static bool   _endcapCrystal[100][100];
  static double _endcapCGap[2][7080][2];
  static double _endcapSGap[2][264][2];
  static double _endcapMGap[2][1][2];

};

#endif
