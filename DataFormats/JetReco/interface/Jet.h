#ifndef JetReco_Jet_h
#define JetReco_Jet_h

 /** \class Jet
 *
 * \short Base class for all types of Jets
 *
 * Jet is an pure virtual interface class. Base class for all types of Jets.
 * More to be added...
 *
 * \author Fernando Varela Rodriguez, Boston University
 *
 * \version   1st Version April 22, 2005.
 *
 ************************************************************/

class Jet {
public:

  // The Jet four-vector as a true Lorentz vector
  /** Returns the jet momentum component along the x axis*/
  virtual double getPx() const = 0;
  /** Returns the jet momentum component along the y axis*/
  virtual double getPy() const = 0;
  /** Returns the jet momentum component along the z axis*/
  virtual double getPz() const = 0;
  /** Returns the total energy of the jet*/
  virtual double getE() const = 0;

  // Standard quantities derived from the Jet Lorentz vector
  /** Returns the modulus of the momentum of the jet*/
  virtual double getP() const = 0;
  /** Returns the transverse momentum of the jet*/
  virtual double getPt() const = 0;
  /** Returns the transverse energy of the jet*/
  virtual double getEt() const = 0;
  /** Returns the jet mass of the jet*/
  virtual double getM() const = 0;
  /** Returns the azimuthal angle of the jet, Phi*/
  virtual double getPhi() const = 0;
  /** Returns the pseudorapidity of the jet*/
  virtual double getEta() const = 0;
  /** Returns the rapidity of the jet*/
  virtual double getY() const = 0;
  /** Returns the number of constituents of the jet*/
  virtual int getNConstituents() const = 0;
};

#endif
