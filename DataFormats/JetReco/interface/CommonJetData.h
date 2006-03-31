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
 *
 ************************************************************/

struct CommonJetData {
public:
  /** Default Constructor */
  CommonJetData() : px(0.), py(0.), pz(0.), e(0.), p(0.), pt(0.),
    et(0.), m(0.), phi(0.), eta(0.), y(0.), numberOfConstituents(0) {
  }

  /** Constructor from values*/
  CommonJetData(double px_, double py_, double pz_, double e_, double p_, double pt_, double et_,
    double m_, double phi_, double eta_, double y_, int n_) :
    px(px_), py(py_), pz(pz_), e(e_), p(p_), pt(pt_), et(et_), m(m_), phi(phi_), eta(eta_), y(y_),
    numberOfConstituents(n_) {
  }

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
  /** Rapidity of the jet*/
  double y;
  /** Number of constituents of the Jet*/
  int numberOfConstituents;
};

#endif
