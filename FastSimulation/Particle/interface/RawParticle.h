#ifndef RAWPARTTICLE_H
#define RAWPARTTICLE_H

#include "CLHEP/config/CLHEP.h"
#include "CLHEP/Vector/LorentzVector.h"

class ParticleTable;

#include <vector>
#include <string>
#include <iosfwd>

/**
 * A prototype for a particle class.
 *  This class describes a general particle beeing a fourvector 
 *  and containing a vertex (fourvector). It is defined in RawParticle.h
 * \author Stephan Wynhoff
 */
class RawParticle : public HepLorentzVector {
public:

  RawParticle();

  virtual ~RawParticle();

  /** Construct from a fourvector.
   *  The fourvector is taken for the particle, the vertex is set to 0. 
   */
  RawParticle(const HepLorentzVector& p);

  /** Construct from a fourvector and a PID.
   *  The fourvector and PID are taken for the particle, the vertex is set to 0.
   */
  RawParticle(const int id, const HepLorentzVector& p);

  /** Construct from a fourvector and a name.
   *  The fourvector and name are taken for the particle, the vertex is set to 0.
   */
  RawParticle(const std::string name, const HepLorentzVector& p);

  /** Construct from 2 fourvectors.
   *  The first fourvector is taken for the particle, the second for its vertex.
   */
  RawParticle(const HepLorentzVector& p, const HepLorentzVector& xStart);

  /** Construct from fourmomentum components.
   *  Vertex is set to 0.
   */
  RawParticle(HepDouble px, HepDouble py, HepDouble pz, HepDouble e);

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
   void setVertex(const HepLorentzVector& vtx); 

  /***  methods to be overloaded to include vertex ***/

  /** Boost the particle. 
   *  The arguments are the \f$\beta\f$ values of the boost in x, y 
   * and z direction. \warning What happens to the vertex?
   */
  // void boost(HepDouble bx, HepDouble by, HepDouble bz);

  //  inline void boost(const Hep3Vector &bv );

  /** Rotate the particle around an axis in space.
   *  The arguments give the amount to rotate \a rphi in radian and a vector
   *  \a raxis in 3D space around which the rotation is done. The vertex is
   *  rotated using the same transformation.
   */
   void rotate(HepDouble rphi, const Hep3Vector &raxis);

  /** \warning not yet implemented   */
  //   void rotateUz(Hep3Vector &nuz);

  /** Rotate around x axis.
   *  Rotate \a rphi radian around the x axis. The Vertex is rotated as well.
   */
   void rotateX(HepDouble rphi); 

  /** Rotate around z axis.
   *  Rotate \a rphi radian around the z axis. The Vertex is rotated as well.
   */

   void rotateY(HepDouble rphi);
  /** Rotate around z axis.
   *  Rotate \a rphi radian around the z axis. The Vertex is rotated as well.
   */

   void rotateZ(HepDouble rphi);

  //  inline RawParticle & transform(const HepRotation &rot);
  //  inline RawParticle & transform(const HepLorentzRotation &rot);

  /** Convert the particle to its charge conjugate state.
      This operation resets the particle ID to that of the charge conjugated 
      particle (if one exists). Also the measured charge is multiplied by -1.
   */
  void chargeConjugate();
  
  int       pid() const;         //!< get the HEP particle ID number
  
  int       status() const;      //!< get the particle status
  
  HepDouble charge() const;      //!< get the MEASURED charge 
  
  HepDouble PDGcharge() const;   //!< get the THEORETICAL charge
  
  HepDouble mass() const;        //!< get the MEASURED mass
  
  HepDouble PDGmass() const;     //!< get the THEORETICAL mass
  
  HepDouble PDGcTau() const;     //!< get the THEORETICAL lifetime
  
  /// get the PDG name
  std::string    PDGname() const;

  /** Get the pseudo rapidity of the particle.
   * \f$ \eta = -\log ( \tan ( \vartheta/2)) \f$
   */
   HepDouble eta() const; 

   HepDouble et() const;   //!< get the transverse energy 

   HepDouble x() const;  //!< x of vertex 

   HepDouble y() const;  //!< y of vertex 

   HepDouble z() const;  //!< z of vertex 

   HepDouble t() const;  //!< vertex time

   const HepLorentzVector& vertex() const;   //!< the vertex fourvector

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

  HepLorentzVector myVertex;   //!< the four vector of the vertex
  int myId;                    //!< the particle id number HEP-PID 
  int myStatus;                //!< the status code according to PYTHIA
  int myUsed;                  //!< status of the locking
  HepDouble myCharge;          //!< the MEASURED charge
  HepDouble myMass;            //!< the RECONSTRUCTED mass

private:
  ParticleTable* tab;
};


std::ostream& operator <<(std::ostream& o , const RawParticle& p); 

inline HepDouble RawParticle::eta() const { return -log(tan(this->theta()/2.)); }
inline HepDouble RawParticle::x() const { return myVertex.x(); }
inline HepDouble RawParticle::y() const { return myVertex.y(); }
inline HepDouble RawParticle::z() const { return myVertex.z(); }
inline HepDouble RawParticle::t() const { return myVertex.t(); }
inline int RawParticle::pid() const { return myId; }
inline int RawParticle::status() const { return myStatus; }
inline HepDouble RawParticle::charge() const { return myCharge; }
inline HepDouble RawParticle::mass() const { return myMass; }

#endif
