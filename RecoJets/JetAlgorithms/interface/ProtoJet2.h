#ifndef JetAlgorithms_ProtoJet2_h
#define JetAlgorithms_ProtoJet2_h

/** \class ProtoJet2
 *
 * \short Transient Jet class used by the reconstruction algorithms.
 *
 * ProtoJet2 is a transitent class used by the reconstruction algorithms. 
 * More to be added...
 *
 * \author Fernando Varela Rodriguez, Boston University
 *
 * \version   1st Version April 22, 2005.
 * \version   2nd Version Oct 19, 2005, R. Harris, modified to work with real CaloTowers from Jeremy Mans.
*
 ************************************************************/

#include "CLHEP/Vector/LorentzVector.h"
#include "PhysicsTools/Candidate/interface/CandidateFwd.h"

#include <vector>
#include <string>
#include <cmath>

int signum(double x);

class ProtoJet2 {
public:
  typedef std::vector <const reco::Candidate*> Candidates;
  /** Default Constructor */
  ProtoJet2();

  //**Constructor. Runs off an array of CaloTower* */
  ProtoJet2(const Candidates& theConstituents);

  /**  Destructor*/
  ~ProtoJet2();

  /// The Jet four-vector as a true Lorentz vector
  /** Returns the jet momentum component along the x axis */
  double px() const {return m_px;};
  /**Returns the jet momentum component along the y axis */
  double py() const {return m_py;};
  /** Returns the jet momentum component along the z axis*/
  double pz() const {return m_pz;};
  /** Returns the total energy of the jet*/
  double e() const {return m_e;};

  /// Standard quantities derived from the Jet Lorentz vector
  /** Returns the modulus of the momentum of the jet */
  double p() const {return sqrt(pow(m_px, 2) + pow(m_py, 2) + pow(m_pz, 2));};
  /** Returns the transverse momentum of the jet*/
  double pt() const {return sqrt(pow(m_px, 2) + pow(m_py, 2));};
  /** Returns the transverse energy of the jet*/
  double et() const {return p()!=0. ? m_e * pt()/p() : 0.;};
  /** Returns the jet mass of the jet*/
  double m() const {return sqrt(pow(e(), 2) - pow(p(), 2));};
  /** Returns the azimuthal angle of the jet, Phi*/
  double phi() const ;
  /** Returns the pseudorapidity of the jet*/
  double eta() const {return (p()-m_pz)!=0. ? 0.5*log((p() + m_pz)/(p() - m_pz)):9999.;};
  /** Returns the rapidity of the jet*/
  double y() const {return (m_e-m_pz)> 0. ? 0.5*log((m_e + m_pz)/(m_e - m_pz)):0.;};
  /** Returns the number of constituents of the Jet*/
  int numberOfConstituents() const {return m_constituents.size();};

  /** Returns the number of constituents carring the 90% of the jet energy*/
  int n90() const;

  /** Returns a Lorentz vector from the four-momentum components*/
  HepLorentzVector getLorentzVector() const;


   /** Returns the list of tower in a particular protojet */
   const Candidates& getTowerList() const {return m_constituents;} 
   /** Sets the list of towers in a protojet */
   void putTowers(const Candidates& towers) {
	m_constituents = towers;
	calculateLorentzVector(); 
   }



private:
  /** Jet constituents */
  Candidates m_constituents;

  /** Jet energy */
  double m_e;
  //The Jet four-vector as a true Lorentz vector: Px, Py, Pz, E
  /** Jet momentum component along the X axis */
  double m_px;
  /** Jet momentum component along the Y axis */
  double m_py;
  /** Jet momentum component along the Z axis */
  double m_pz;
  /** Private Method to calculate LorentzVector  */
  void calculateLorentzVector();
};

#endif
