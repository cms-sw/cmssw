#ifndef CommonTools_BaseParticlePropagator_RawParticle_h
#define CommonTools_BaseParticlePropagator_RawParticle_h

#include "DataFormats/Math/interface/LorentzVector.h"
#include "DataFormats/Math/interface/Vector3D.h"

#include "Math/GenVector/RotationX.h"
#include "Math/GenVector/RotationY.h"
#include "Math/GenVector/RotationZ.h"
#include "Math/GenVector/Rotation3D.h"
#include "Math/GenVector/AxisAngle.h"
#include "Math/GenVector/Boost.h"

#include <string>
#include <iosfwd>

#include <memory>

/**
 * A prototype for a particle class.
 *  This class describes a general particle beeing a fourvector 
 *  and containing a vertex (fourvector). It is defined in RawParticle.h
 * \author Stephan Wynhoff
 */
typedef math::XYZTLorentzVector XYZTLorentzVector;
typedef math::XYZVector XYZVector;

class RawParticle;

namespace rawparticle {
  ///Create a particle with momentum 'p' at space-time point xStart
  /// The particle will be a muon if iParticle==true, else it will
  /// be an anti-muon.
  RawParticle makeMuon(bool isParticle, const XYZTLorentzVector& p, const XYZTLorentzVector& xStart);
}  // namespace rawparticle

class RawParticle {
public:
  friend RawParticle rawparticle::makeMuon(bool, const XYZTLorentzVector&, const XYZTLorentzVector&);
  friend RawParticle unchecked_makeParticle(int id, const math::XYZTLorentzVector& p, double mass, double charge);
  friend RawParticle unchecked_makeParticle(
      int id, const math::XYZTLorentzVector& p, const math::XYZTLorentzVector& xStart, double mass, double charge);

  typedef ROOT::Math::AxisAngle Rotation;
  typedef ROOT::Math::Rotation3D Rotation3D;
  typedef ROOT::Math::RotationX RotationX;
  typedef ROOT::Math::RotationY RotationY;
  typedef ROOT::Math::RotationZ RotationZ;
  typedef ROOT::Math::Boost Boost;

  RawParticle() = default;

  /** Construct from a fourvector.
   *  The fourvector is taken for the particle, the vertex is set to 0. 
   */
  RawParticle(const XYZTLorentzVector& p);

  /** Construct from 2 fourvectors.
   *  The first fourvector is taken for the particle, the second for its vertex.
   */
  RawParticle(const XYZTLorentzVector& p, const XYZTLorentzVector& xStart, double charge = 0.);

  /** Construct from fourmomentum components.
   *  Vertex is set to 0.
   */
  RawParticle(double px, double py, double pz, double e, double charge = 0.);

  /** Copy constructor    */
  RawParticle(const RawParticle& p) = default;
  RawParticle(RawParticle&& p) = default;

  /** Copy assignment operator */
  RawParticle& operator=(const RawParticle& rhs) = default;
  RawParticle& operator=(RawParticle&& rhs) = default;

public:
  /** Set the status of this particle.
   *  The coding follows PYTHIAs convention:
   *  1 = stable
   */
  void setStatus(int istat);

  /// set the RECONSTRUCTED mass
  void setMass(float m);

  /// set the MEASURED charge
  void setCharge(float q);

  /// set the time of creation
  void setT(const double t);

  ///  set the vertex
  void setVertex(const XYZTLorentzVector& vtx);
  void setVertex(double xv, double yv, double zv, double tv);

  ///  set the momentum
  void setMomentum(const XYZTLorentzVector& vtx);
  void setMomentum(double xv, double yv, double zv, double tv);

  void SetPx(double);
  void SetPy(double);
  void SetPz(double);
  void SetE(double);

  /***  methods to be overloaded to include vertex ***/

  /** Boost the particle. 
   *  The arguments are the \f$\beta\f$ values of the boost in x, y 
   * and z direction. \warning What happens to the vertex?
   */
  void boost(double bx, double by, double bz);
  void boost(const Boost& b);

  //  inline void boost(const Hep3Vector<double> &bv );

  /** Rotate the particle around an axis in space.
   *  The arguments give the amount to rotate \a rphi in radian and a vector
   *  \a raxis in 3D space around which the rotation is done. The vertex is
   *  rotated using the same transformation.
   */
  void rotate(double rphi, const XYZVector& raxis);
  void rotate(const Rotation& r);
  void rotate(const Rotation3D& r);

  /** \warning not yet implemented   */
  //   void rotateUz(Hep3Vector &nuz);

  /** Rotate around x axis.
   *  Rotate \a rphi radian around the x axis. The Vertex is rotated as well.
   */
  void rotateX(double rphi);
  void rotate(const RotationX& r);

  /** Rotate around z axis.
   *  Rotate \a rphi radian around the z axis. The Vertex is rotated as well.
   */

  void rotateY(double rphi);
  void rotate(const RotationY& r);
  /** Rotate around z axis.
   *  Rotate \a rphi radian around the z axis. The Vertex is rotated as well.
   */

  void rotateZ(double rphi);
  void rotate(const RotationZ& r);

  /** Translate the vertex by a given space amount */
  void translate(const XYZVector& t);

  //  inline RawParticle & transform(const HepRotation &rot);
  //  inline RawParticle & transform(const HepLorentzRotation &rot);

  /** Convert the particle to its charge conjugate state.
      This operation resets the particle ID to that of the charge conjugated 
      particle (if one exists). Also the measured charge is multiplied by -1.
   */
  void chargeConjugate();

  int pid() const;  //!< get the HEP particle ID number

  int status() const;  //!< get the particle status

  double charge() const;  //!< get the MEASURED charge

  double mass() const;  //!< get the MEASURED mass

  /** Get the pseudo rapidity of the particle.
   * \f$ \eta = -\log ( \tan ( \vartheta/2)) \f$
   */
  double eta() const;

  /// Cos**2(theta) is faster to determine than eta
  double cos2Theta() const;
  double cos2ThetaV() const;

  double et() const;  //!< get the transverse energy

  double x() const;  //!< x of vertex
  double X() const;  //!< x of vertex

  double y() const;  //!< y of vertex
  double Y() const;  //!< y of vertex

  double z() const;  //!< z of vertex
  double Z() const;  //!< z of vertex

  double t() const;  //!< vertex time
  double T() const;  //!< vertex time

  double r() const;  //!< vertex radius
  double R() const;  //!< vertex radius

  double r2() const;  //!< vertex radius**2
  double R2() const;  //!< vertex radius**2

  const XYZTLorentzVector& vertex() const;  //!< the vertex fourvector

  double px() const;  //!< x of the momentum
  double Px() const;  //!< x of the momentum

  double py() const;  //!< y of the momentum
  double Py() const;  //!< y of the momentum

  double pz() const;  //!< z of the momentum
  double Pz() const;  //!< z of the momentum

  double e() const;  //!< energy of the momentum
  double E() const;  //!< energy of the momentum

  double Pt() const;  //!< transverse momentum
  double pt() const;  //!< transverse momentum

  double Perp2() const;  //!< perpendicular momentum squared

  double mag() const;  //!< the magnitude of the momentum

  double theta() const;  //!< theta of momentum vector
  double phi() const;    //!< phi of momentum vector

  double M2() const;  //!< mass squared

  const XYZTLorentzVector& momentum() const;  //!< the momentum fourvector
  XYZTLorentzVector& momentum();              //!< the momentum fourvector

  XYZVector Vect() const;  //!< the momentum threevector

  /** Print the name of the particle.
   *  The name is deduced from the particle ID using a particle data table.
   *  It is printed with a length of 10 characters. If the id number cannot
   *  be found in the table "unknown" is printed as name.
   */
  void printName() const;

  /** Print the formated particle information.
   *  The format is:
   *  NAME______PX______PY______PZ______E_______Mtheo___Mrec____Qrec____X_______Y_______Z_______T_______
   */
  void print() const;

  /** Is the particle marked as used.
   *  The three methods isUsed(), use() and reUse() implement a simple
   *  locking mechanism. 
   */
  int isUsed() const { return myUsed; }

  /** Lock the particle, see isUsed()
   */
  void use() { myUsed = 1; }

  /** Unlock the particle, see isUsed()
   */
  void reUse() { myUsed = 0; }

private:
  /** Construct from a fourvector and a PID.
   *  The fourvector and PID are taken for the particle, the vertex is set to 0.
   */
  RawParticle(const int id, const XYZTLorentzVector& p, double mass, double charge);

  /** Construct from 2 fourvectosr and a PID.
   *  The fourvector and PID are taken for the particle, the vertex is set to 0.
   */
  RawParticle(const int id, const XYZTLorentzVector& p, const XYZTLorentzVector& xStart, double mass, double charge);

private:
  XYZTLorentzVector myMomentum;  //!< the four vector of the momentum
  XYZTLorentzVector myVertex;    //!< the four vector of the vertex
  double myCharge = 0.;          //!< the MEASURED charge
  double myMass = 0.;            //!< the RECONSTRUCTED mass
  int myId = 0;                  //!< the particle id number HEP-PID
  int myStatus = 99;             //!< the status code according to PYTHIA
  int myUsed = 0;                //!< status of the locking
};

std::ostream& operator<<(std::ostream& o, const RawParticle& p);

inline int RawParticle::pid() const { return myId; }
inline int RawParticle::status() const { return myStatus; }
inline double RawParticle::eta() const { return -std::log(std::tan(this->theta() / 2.)); }
inline double RawParticle::cos2Theta() const { return Pz() * Pz() / myMomentum.Vect().Mag2(); }
inline double RawParticle::cos2ThetaV() const { return Z() * Z() / myVertex.Vect().Mag2(); }
inline double RawParticle::x() const { return myVertex.Px(); }
inline double RawParticle::y() const { return myVertex.Py(); }
inline double RawParticle::z() const { return myVertex.Pz(); }
inline double RawParticle::t() const { return myVertex.E(); }
inline double RawParticle::X() const { return myVertex.Px(); }
inline double RawParticle::Y() const { return myVertex.Py(); }
inline double RawParticle::Z() const { return myVertex.Pz(); }
inline double RawParticle::T() const { return myVertex.E(); }
inline double RawParticle::R() const { return std::sqrt(R2()); }
inline double RawParticle::R2() const { return myVertex.Perp2(); }
inline double RawParticle::r() const { return std::sqrt(r2()); }
inline double RawParticle::r2() const { return myVertex.Perp2(); }
inline double RawParticle::charge() const { return myCharge; }
inline double RawParticle::mass() const { return myMass; }
inline double RawParticle::px() const { return myMomentum.px(); }
inline double RawParticle::Px() const { return myMomentum.Px(); }

inline double RawParticle::py() const { return myMomentum.py(); }
inline double RawParticle::Py() const { return myMomentum.Py(); }

inline double RawParticle::pz() const { return myMomentum.pz(); }
inline double RawParticle::Pz() const { return myMomentum.Pz(); }

inline double RawParticle::e() const { return myMomentum.e(); }
inline double RawParticle::E() const { return myMomentum.E(); }

inline double RawParticle::Pt() const { return myMomentum.Pt(); }
inline double RawParticle::pt() const { return myMomentum.pt(); }

inline double RawParticle::Perp2() const { return myMomentum.Perp2(); }

inline double RawParticle::mag() const { return myMomentum.mag(); }

inline double RawParticle::theta() const { return myMomentum.theta(); }
inline double RawParticle::phi() const { return myMomentum.phi(); }

inline double RawParticle::M2() const { return myMomentum.M2(); }

inline const XYZTLorentzVector& RawParticle::vertex() const { return myVertex; }
inline const XYZTLorentzVector& RawParticle::momentum() const { return myMomentum; }
inline XYZTLorentzVector& RawParticle::momentum() { return myMomentum; }
inline XYZVector RawParticle::Vect() const { return myMomentum.Vect(); }

inline void RawParticle::setVertex(const XYZTLorentzVector& vtx) { myVertex = vtx; }
inline void RawParticle::setVertex(double a, double b, double c, double d) { myVertex.SetXYZT(a, b, c, d); }

inline void RawParticle::setMomentum(const XYZTLorentzVector& p4) { myMomentum = p4; }
inline void RawParticle::setMomentum(double a, double b, double c, double d) { myMomentum.SetXYZT(a, b, c, d); }

inline void RawParticle::SetPx(double px) { myMomentum.SetPx(px); }
inline void RawParticle::SetPy(double py) { myMomentum.SetPy(py); }
inline void RawParticle::SetPz(double pz) { myMomentum.SetPz(pz); }
inline void RawParticle::SetE(double e) { myMomentum.SetE(e); }

inline void RawParticle::rotate(const RawParticle::Rotation3D& r) {
  XYZVector v(r(myMomentum.Vect()));
  setMomentum(v.X(), v.Y(), v.Z(), E());
}

inline void RawParticle::rotate(const RawParticle::Rotation& r) {
  XYZVector v(r(myMomentum.Vect()));
  setMomentum(v.X(), v.Y(), v.Z(), E());
}

inline void RawParticle::rotate(const RawParticle::RotationX& r) {
  XYZVector v(r(myMomentum.Vect()));
  setMomentum(v.X(), v.Y(), v.Z(), E());
}

inline void RawParticle::rotate(const RawParticle::RotationY& r) {
  XYZVector v(r(myMomentum.Vect()));
  setMomentum(v.X(), v.Y(), v.Z(), E());
}

inline void RawParticle::rotate(const RawParticle::RotationZ& r) {
  XYZVector v(r(myMomentum.Vect()));
  setMomentum(v.X(), v.Y(), v.Z(), E());
}

inline void RawParticle::boost(const RawParticle::Boost& b) {
  XYZTLorentzVector p(b(momentum()));
  setMomentum(p.Px(), p.Py(), p.Pz(), p.E());
}

inline void RawParticle::translate(const XYZVector& tr) {
  myVertex.SetXYZT(X() + tr.X(), Y() + tr.Y(), Z() + tr.Z(), T());
}

#endif
