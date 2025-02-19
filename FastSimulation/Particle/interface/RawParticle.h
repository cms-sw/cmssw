#ifndef RAWPARTTICLE_H
#define RAWPARTTICLE_H

#include "SimGeneral/HepPDTRecord/interface/ParticleDataTable.h"
#include "DataFormats/Math/interface/LorentzVector.h"
#include "DataFormats/Math/interface/Vector3D.h"

#include "Math/GenVector/RotationX.h"
#include "Math/GenVector/RotationY.h"
#include "Math/GenVector/RotationZ.h"
#include "Math/GenVector/Rotation3D.h"
#include "Math/GenVector/AxisAngle.h"
#include "Math/GenVector/Boost.h"

class ParticleTable;

#include <string>
#include <iosfwd>



/**
 * A prototype for a particle class.
 *  This class describes a general particle beeing a fourvector 
 *  and containing a vertex (fourvector). It is defined in RawParticle.h
 * \author Stephan Wynhoff
 */
typedef math::XYZTLorentzVector XYZTLorentzVector;
typedef math::XYZVector XYZVector;

class RawParticle : public XYZTLorentzVector {
public:

  typedef ROOT::Math::AxisAngle Rotation;
  typedef ROOT::Math::Rotation3D Rotation3D;
  typedef ROOT::Math::RotationX RotationX;
  typedef ROOT::Math::RotationY RotationY;
  typedef ROOT::Math::RotationZ RotationZ;
  typedef ROOT::Math::Boost Boost;

  RawParticle();

  virtual ~RawParticle();

  /** Construct from a fourvector.
   *  The fourvector is taken for the particle, the vertex is set to 0. 
   */
  RawParticle(const XYZTLorentzVector& p);

  /** Construct from a fourvector and a PID.
   *  The fourvector and PID are taken for the particle, the vertex is set to 0.
   */
  RawParticle(const int id, 
	      const XYZTLorentzVector& p);

  /** Construct from a fourvector and a name.
   *  The fourvector and name are taken for the particle, the vertex is set to 0.
   */
  RawParticle(const std::string name, 
	      const XYZTLorentzVector& p);

  /** Construct from 2 fourvectors.
   *  The first fourvector is taken for the particle, the second for its vertex.
   */
  RawParticle(const XYZTLorentzVector& p, 
	      const XYZTLorentzVector& xStart);

  /** Construct from fourmomentum components.
   *  Vertex is set to 0.
   */
  RawParticle(double px, double py, double pz, double e);

  /** Copy constructor    */
  RawParticle(const RawParticle &p);

  /** Copy assignment operator */
  RawParticle&  operator = (const RawParticle & rhs );

public:

  /** Set identifier for this particle.
   *  This should be a standard HEP-PID number. It will be used to deduce the 
   *  name and the properties of the particle from a particle data table.
   */
  void setID(const int id); 

  /** Set identifier for this particle.
   *  This should be a standard HEP-PID name. It will be used to deduce the 
   *  particle properties from a particle data table.
   */
  void setID(const std::string name); 

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
  
  int    pid() const;         //!< get the HEP particle ID number
  
  int    status() const;      //!< get the particle status
  
  double charge() const;      //!< get the MEASURED charge 
  
  double PDGcharge() const;   //!< get the THEORETICAL charge
  
  double mass() const;        //!< get the MEASURED mass
  
  double PDGmass() const;     //!< get the THEORETICAL mass
  
  double PDGcTau() const;     //!< get the THEORETICAL lifetime
  
  /// get the PDG name
  std::string    PDGname() const;

  /** Get the pseudo rapidity of the particle.
   * \f$ \eta = -\log ( \tan ( \vartheta/2)) \f$
   */
  double eta() const; 

  /// Cos**2(theta) is faster to determine than eta 
  double cos2Theta() const;
  double cos2ThetaV() const;
  
  double et() const;   //!< get the transverse energy 
  
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
  
  const XYZTLorentzVector& vertex() const;   //!< the vertex fourvector
  
  const XYZTLorentzVector& momentum() const;   //!< the momentum fourvector
  
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
  int isUsed() const {return myUsed;}
  
  /** Lock the particle, see isUsed()
   */
  void use() { myUsed = 1;}    
  
  /** Unlock the particle, see isUsed()
   */
  void reUse() { myUsed = 0;}  
  
 private:
  
  void init();
  
 protected:
  
  XYZTLorentzVector myVertex;         //!< the four vector of the vertex
  int myId;                           //!< the particle id number HEP-PID 
  int myStatus;                       //!< the status code according to PYTHIA
  int myUsed;                         //!< status of the locking
  double myCharge;                    //!< the MEASURED charge
  double myMass;                      //!< the RECONSTRUCTED mass
  const ParticleData* myInfo;         //!< The pointer to the PDG info
  
 private:
  ParticleTable* tab;
};


std::ostream& operator <<(std::ostream& o , const RawParticle& p); 

inline int RawParticle::pid() const { return myId; }
inline int RawParticle::status() const { return myStatus; }
inline double RawParticle::eta() const { return -std::log(std::tan(this->theta()/2.)); }
inline double RawParticle::cos2Theta() const { return Pz()*Pz()/Vect().Mag2(); }
inline double RawParticle::cos2ThetaV() const { return Z()*Z()/myVertex.Vect().Mag2(); }
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

inline const XYZTLorentzVector&  RawParticle::vertex() const { return myVertex ; }
inline const XYZTLorentzVector&  RawParticle::momentum() const { return *this; }

inline void RawParticle::setVertex(const XYZTLorentzVector& vtx) { myVertex = vtx; }
inline void RawParticle::setVertex(double a, double b, double c, double d) { myVertex.SetXYZT(a,b,c,d); }

inline void RawParticle::rotate(const RawParticle::Rotation3D& r) { 
  XYZVector v ( r(Vect()) ); SetXYZT(v.X(),v.Y(),v.Z(),E());
}

inline void RawParticle::rotate(const RawParticle::Rotation& r) { 
  XYZVector v ( r(Vect()) ); SetXYZT(v.X(),v.Y(),v.Z(),E());
}

inline void RawParticle::rotate(const RawParticle::RotationX& r) { 
  XYZVector v ( r(Vect()) ); SetXYZT(v.X(),v.Y(),v.Z(),E());
}

inline void RawParticle::rotate(const RawParticle::RotationY& r) { 
  XYZVector v ( r(Vect()) ); SetXYZT(v.X(),v.Y(),v.Z(),E());
}

inline void RawParticle::rotate(const RawParticle::RotationZ& r) { 
  XYZVector v ( r(Vect()) ); SetXYZT(v.X(),v.Y(),v.Z(),E());
}

inline void RawParticle::boost(const RawParticle::Boost& b) { 
  XYZTLorentzVector p ( b(momentum()) ); SetXYZT(p.Px(),p.Py(),p.Pz(),p.E());
}

inline void RawParticle::translate(const XYZVector& tr) { 
  myVertex.SetXYZT(X()+tr.X(),Y()+tr.Y(),Z()+tr.Z(),T());
}





#endif
