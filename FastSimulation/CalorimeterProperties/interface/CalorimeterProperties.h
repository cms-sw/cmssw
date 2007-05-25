#ifndef CalorimeterProperties_H
#define CalorimeterProperties_H

/** 
 * Base class for calorimeter properties
 *
 * \author Patrick Janot
 * \date: 25-Jan-2004
 */ 

class CalorimeterProperties
{
 public:
  CalorimeterProperties() { } 
  
  virtual ~CalorimeterProperties() {
    ;
  }
  
  /// Effective A
  inline virtual double theAeff() const=0;

  /// Effective Z
  inline virtual double theZeff() const=0;

  /// Density in g/cm3
  inline virtual double rho() const=0;

  /// Radiation length in cm
  inline virtual double radLenIncm() const=0;

  /// Radiation length in g/cm^2
  inline virtual double radLenIngcm2() const=0; 

  /// Moliere Radius in cm
  inline virtual double moliereRadius() const=0; 

  /// Critical energy in GeV (2.66E-3*(x0*Z/A)^1.1)
  inline virtual double criticalEnergy() const=0;

  ///Interaction length in cm
  inline virtual double interactionLength() const=0;

  ///Thickness (in cm) of the homegeneous material as a function of rapidity
  virtual double thickness(double eta) const=0;

 private:

};

#endif
