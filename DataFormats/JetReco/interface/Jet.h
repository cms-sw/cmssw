#ifndef JetReco_Jet_h
#define JetReco_Jet_h

 /** \class Jet
 *
 * \short Base class for all types of Jets
 *
 * Jet describes properties common for all kinds of jets, 
 * essentially kinematics. Base class for all types of Jets.
 * More to be added...
 *
 * \author Fedor Ratnikov, UMd
 *
 * \version   Original: April 22, 2005 by Fernando Varela Rodriguez.
 * \version   May 23, 2006 by F.R.
 * \version   $Id: Jet.h,v 1.4 2006/05/24 00:40:43 fedor Exp $
 ************************************************************/
#include "DataFormats/Math/interface/LorentzVector.h"

class Jet {
public:
  /// Lorentz vector
  typedef math::XYZTLorentzVector LorentzVector;

  Jet () : mNumberOfConstituents (0) {}
  Jet (const LorentzVector& fP4, int fNumberOfConstituents = 0) 
    : mP4 (fP4),  mNumberOfConstituents (fNumberOfConstituents) {}
  virtual ~Jet () {}

  /// four-momentum Lorentz vector
  const LorentzVector& p4() const {return mP4;}
  /** Returns the jet momentum component along the x axis */
  double px() const {return p4().Px();}
  /** Returns the jet momentum component along the y axis */
  double py() const {return p4().Py();}
  /** Returns the jet momentum component along the z axis */
  double pz() const {return p4().Pz();}
  /** Returns the total energy of the jet*/
  double energy () const {return p4().E();}

  /** Standard quantities derived from the Jet Lorentz vector
  /\return the modulus of the momentum of the jet */
  double p() const {return p4().P();}
  /** Returns the transverse momentum of the jet*/
  double pt() const {return p4().Pt();}
  /** Returns the transverse energy of the jet*/
  double et() const {return p4().Et();}
  /** Returns the jet mass of the jet*/
  double mass() const {return p4().M();}
  /** Returns the azimuthal angle of the jet, Phi*/
  double phi() const {return p4().Phi();}
  /** Returns the pseudorapidity of the jet*/
  double eta() const {return p4().Eta();}
  /** Returns the rapidity of the jet*/
  double y() const {return p4().Rapidity();}
  /** Returns the number of constituents of the jet*/
  virtual int nConstituents() const {return mNumberOfConstituents;}
 protected:
  void setNConstituents (int fNConstituents) {mNumberOfConstituents = fNConstituents;}
  void setP4 (const LorentzVector& fP4) {mP4 = fP4;}
 private:
  /** 4-momentum of the jet*/
  LorentzVector mP4;
  /** Number of constituents of the Jet*/
  int mNumberOfConstituents;
};

#endif
