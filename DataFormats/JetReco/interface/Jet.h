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
#include "DataFormats/Math/interface/LorentzVector.h"
#include "DataFormats/Math/interface/Vector3D.h"

class Jet {
public:
  /// Lorentz vector
  typedef math::XYZTLorentzVector LorentzVector;
  /// spatial vector
  typedef math::XYZVector Vector;

  /// four-momentum Lorentz vector
  virtual LorentzVector p4() const = 0;
  /// spatial momentum vector
  virtual Vector momentum() const = 0;
  /** Returns the jet momentum component along the x axis */
  virtual double px() const = 0;
  /** Returns the jet momentum component along the y axis */
  virtual double py() const = 0;
  /** Returns the jet momentum component along the z axis */
  virtual double pz() const = 0;
  /** Returns the total energy of the jet*/
  virtual double energy () const = 0;

  /** Standard quantities derived from the Jet Lorentz vector
  /\return the modulus of the momentum of the jet */
  virtual double p() const = 0;
  /** Returns the transverse momentum of the jet*/
  virtual double pt() const = 0;
  /** Returns the transverse energy of the jet*/
  virtual double et() const = 0;
  /** Returns the jet mass of the jet*/
  virtual double mass() const = 0;
  /** Returns the azimuthal angle of the jet, Phi*/
  virtual double phi() const = 0;
  /** Returns the pseudorapidity of the jet*/
  virtual double eta() const = 0;
  /** Returns the number of constituents of the jet*/
  virtual int nConstituents() const = 0;
};

#endif
