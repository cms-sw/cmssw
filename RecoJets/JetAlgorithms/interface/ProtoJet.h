#ifndef JetAlgorithms_ProtoJet_h
#define JetAlgorithms_ProtoJet_h

/** \class ProtoJet
 *
 * \short Transient Jet class used by the reconstruction algorithms.
 *
 * ProtoJet is a transitent class used by the reconstruction algorithms. 
 *
 * \author Fernando Varela Rodriguez, Boston University
 *
 * \version   1st Version April 22, 2005.
 * \version   2nd Version Oct 19, 2005, R. Harris, modified to work with real CaloTowers from Jeremy Mans.
 * \version   3rd Version Mar 8, 2006, F.Ratnikov, use Candidate approach
*
 ************************************************************/

#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include "DataFormats/Math/interface/LorentzVector.h"

#include <vector>
#include <string>

class ProtoJet {
public:
  typedef reco::CandidateRef Constituent;
  typedef std::vector <Constituent> Constituents;

  typedef math::XYZTLorentzVector LorentzVector;
  /** Default Constructor */
  ProtoJet();

  //**Constructor. Runs off an array of CaloTower* */
  ProtoJet(const Constituents& theConstituents);

 //**Full Constructor */
  ProtoJet(const LorentzVector& fP4, const Constituents& fConstituents); 

  /**  Destructor*/
  ~ProtoJet() {}


  // The Jet four-vector as a true Lorentz vector
  /** Returns the jet momentum component along the x axis */
  double px() const {return mP4.Px();}
  /**Returns the jet momentum component along the y axis */
  double py() const {return mP4.Py();}
  /** Returns the jet momentum component along the z axis*/
  double pz() const {return mP4.Pz();}
  /** Returns the total energy of the jet*/
  double e() const {return mP4.E();}
  /** Returns the total energy of the jet*/
  double energy() const {return e();}
  
  // Standard quantities derived from the Jet Lorentz vector
  /** Returns the modulus of the momentum of the jet */
  double p() const {return mP4.P();}
  /** Returns the transverse momentum of the jet*/
  double pt() const {return mP4.Pt();}
  /** Returns the transverse energy of the jet*/
  double et() const {return mP4.Et();}
  /** Returns the jet mass of the jet*/
  double m() const {return mP4.M();}
  /** Returns the azimuthal angle of the jet, Phi*/
  double phi() const {return mP4.Phi();}
  /** Returns the pseudorapidity of the jet*/
  double eta() const {return mP4.Eta();}
  /** Returns the rapidity of the jet*/
  double y() const {return p() > 0 ? mP4.Rapidity() : 0;}
  /** Returns the number of constituents of the Jet*/
  int numberOfConstituents() const {return mConstituents.size();};


  /** Returns a Lorentz vector from the four-momentum components*/
  const LorentzVector& p4() const {return mP4;}
  
  
  /** Returns the list of tower in a particular protojet */
  const Constituents& getTowerList();
  const Constituents getTowerList() const;
  
  /** Sets the list of towers in a protojet */
  void putTowers(const Constituents& towers);

  /** Reorder towers by eT */
  void reorderTowers ();

  /** Make kinematics from constituents */
  void calculateLorentzVector() {calculateLorentzVectorERecombination();}
  void calculateLorentzVectorERecombination();
  void calculateLorentzVectorEtRecombination();
  
  


private:
  /** Jet kinematics */
  LorentzVector mP4;
  /** Jet constituents */
  Constituents mConstituents;
  bool mOrdered;
};

#endif
