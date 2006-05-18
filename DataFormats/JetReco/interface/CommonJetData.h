#ifndef JetReco_CommonJetData_h
#define JetReco_CommonJetData_h

 /** \class CommonJetData
 *
 * \short Structure containing data common to all types of Jets
 *
 * CommonJetData is an structure that is inherited by all types of Jets
 * It holds information common to all types of Jets.
 * More to be added...
 *
 * \author Fernando Varela Rodriguez, Boston University
 *
 * \version   1st Version April 22, 2005.
 * \version   2nd Version April 27, 2006. F.Ratnikov. Use ROOT's LorentzVector
 *
 ************************************************************/
#include "DataFormats/Math/interface/LorentzVector.h"

struct CommonJetData {
public:
  /** Lorentz vector */
  typedef math::XYZTLorentzVector LorentzVector;

  /** Default Constructor */
  CommonJetData() : numberOfConstituents(0) {}

  /** Constructor from values*/
  CommonJetData(const LorentzVector& p4, int n);

  CommonJetData(double px, double py, double pz, double e, int n);

  /** Default Destructor*/
  ~CommonJetData() {}

 // The Jet four-vector as a true Lorentz vector
  /** Jet momentum component along the x axis*/
  double px;
  /** Jet momentum component along the y axis*/
  double py;
  /** Jet momentum component along the z axis*/
  double pz;
  /** Total energy of the jet*/
  double e;

  // Standard quantities derived from the Jet Lorentz vector
  /** Modulus of the momentum of the jet*/
  double p;
  /** Transverse momentum of the jet*/
  double pt;
  /** Transverse energy of the jet*/
  double et;
  /** Jet mass*/
  double m;
  /** Azimuthal angle of the jet, Phi*/
  double phi;
  /** Pseudorapidity of the jet*/
  double eta;
  /** 4-momentum of the jet*/
  LorentzVector mP4;
  /** Number of constituents of the Jet*/
  int numberOfConstituents;

private:
  void init ();
};

#endif
