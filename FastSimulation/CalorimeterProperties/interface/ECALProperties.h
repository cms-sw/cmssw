#ifndef ECALProperties_H
#define ECALProperties_H

#include "FastSimulation/CalorimeterProperties/interface/CalorimeterProperties.h"

/** 
 * Functions to return atomic properties of the material
 * A_eff and Z_eff are computed as the A-weighted sums 
 * of the A's and the Z's of Pb, W and O
 *
 * \author Patrick Janot
 * \date: 25-Jan-2004
 */ 

class ECALProperties : public CalorimeterProperties 
{

 public:

  ECALProperties() : scaleEnergy_(0.0212){ } 

  virtual ~ECALProperties() {
  }

  /// Effective A: 170.87 for Standard ECAL
  inline double theAeff() const { return Aeff_; }

  /// Effective Z: 68.36 for Standard ECAL
  inline double theZeff() const { return Zeff_; }

  /// Density in g/cm3: 8.280 for Standard ECAL
  inline double rho() const { return rho_; }

  /// Radiation length in cm
  //  inline double radLenIncm()  const { return radiationLengthIncm(); }: 0.89 for Standard ECAL
  inline double radLenIncm()  const { return radLenIncm_; }

  /// Radiation length in cm but static 
  // This is needed in Calorimetry/CrystalSegment. Patrick, if you don't like it, give
  // me an other solution to access the ECALProperties efficiently. 
  // static inline double radiationLengthIncm() { return 0.89; }

  /// Radiation length in g/cm^2: 7.37  for Standard ECAL
  inline double radLenIngcm2() const { return radLenIngcm2_; }

  /// Moliere Radius in cm : 2.190 for Standard ECAL
  inline   double moliereRadius() const { return moliereRadius_; }

  /// Critical energy in GeV (2.66E-3*(x0*Z/A)^1.1): 8.74E-3 for Standard ECAL
  inline double criticalEnergy() const { return criticalEnergy_; }

  ///Interaction length in cm: 18.5 for Standard ECAL
  inline double interactionLength() const { return interactionLength_; }

  ///Sampling fraction Fs of the calorimeter. 0 for homogeneous one
  inline double theFs() const { return Fs_; }

  /// ehat = e/mip of the calorimeter. 0 for homogeneous one
  inline double ehat() const { return ehat_; }

  /// a rough estimate of ECAL resolution sigma/E = resE/sqrt(E)
  inline double resE() const { return resE_; }

  /// the width of the active layer in the case of the homogeneous detector
  inline double da() const { return da_; }

  /// the width of the passive layer in the case of the homogeneous detector
  inline double dp() const { return dp_; }


  /// a rough estimate of ECAL resolution sigma/E = resE/sqrt(E)
  inline bool isHom() const { return bHom_; }

 ///Photostatistics (photons/GeV) in the homegeneous material
  virtual double photoStatistics() const=0;

  ///Light Collection efficiency 
  virtual double lightCollectionEfficiency() const=0;

  ///Light Collection uniformity
  virtual double lightCollectionUniformity() const=0;

 protected:

  double lightColl_;
  double lightCollUnif_;
  double photoStatistics_;
  double thickness_;
  double interactionLength_;

  double Aeff_;
  double Zeff_;
  double rho_;
  double radLenIncm_;
  double radLenIngcm2_;
  double moliereRadius_;

  double criticalEnergy_;

  double Fs_;
  double ehat_;
  double resE_;
  
  double da_;
  double dp_;

  bool bHom_;

  //Es=sqrt(4*Pi/alphaEM)*me*c^2=0.0212 GeV used for Moliere Radius calculation
  const double scaleEnergy_;

};

#endif
