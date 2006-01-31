#ifndef JetAlgorithms_ProtoJet_h
#define JetAlgorithms_ProtoJet_h
#include "CLHEP/Vector/LorentzVector.h"
#include "PhysicsTools/Candidate/interface/Candidate.h"

#include <vector>
#include <string>
#include <cmath>

int signum(double x);

class ProtoJet {
public:
  /** Default Constructor */
  ProtoJet();

  //**Constructor. Runs off an array of CaloTower* */
  ProtoJet(std::vector<const aod::Candidate *> theConstituents);

  /** Constructor. Runs off an array of CaloTower */
  ProtoJet(std::vector<const aod::Candidate> theConstituents);

  /** Constructor. Runs off an array of a CaloTowerCollection */
  ProtoJet(aod::CandidateCollection aTowerCollection);

  /**  Destructor*/
  ~ProtoJet();

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
  //  double phi() const ;
  /** Returns the pseudorapidity of the jet*/
  double eta() const {return (p()-m_pz)!=0. ? 0.5*log((p() + m_pz)/(p() - m_pz)):9999.;};
  /** Returns the rapidity of the jet*/
  double y() const {return (m_e-m_pz)> 0. ? 0.5*log((m_e + m_pz)/(m_e - m_pz)):0.;};
  /** Returns the number of constituents of the Jet*/
  int numberOfConstituents() const {return m_constituents.size();};

//All these methods must be moved to the the CaloJet class

  // Diagnostic Jet properties
  /** Returns the maximum energy in an EM tower*/
  //  double maxEInEmTowers() const;
  /** Returns the maximum energy in an HAD tower*/
  //  double maxEInHadTowers() const;

  /* To be implemented:
     - Will return the number of tracks pointing at the Jet, in the case
     the jet is made of tracks
     double getNumberOfTracks();

     - Will return the vector sum of PT of tracks that form the Jet
     divided by the Pt of the Jet
     double getChargedFraction();
  */
  /** Returns the energy fraction in the EM calorimeter normalized to the total
  energy*/
  //  double emFraction() const;

  /** Returns the number of constituents carring the 90% of the jet energy*/
  //  int n90() const;

  /** Returns a Lorentz vector from the four-momentum components*/
  HepLorentzVector getLorentzVector() const;

  // Returns the energy fraction deposited in HO
  //double energyFractionInHO() const;

  // Returns the energy fraction deposited in a particular subdetector layer within a detector
  // \param detector: name of the subdetector, i.e. HCAL or ECAL
  // \param subdetector: name of the subdetector, i.e. for HCAL the possibilities are: HB, HF, HED,
  // HES or ALL. Default value is ALL, i.e. calculate the energy fraction in a all subdetectors of a
  // given subdetector
  //   \param layer: Layer number. The default value is -1, i.e. look in all layers

  //double energyFraction(const std::string detector = "HCAL", const std::string subdetector = "ALL", int layer= -1) const;

   /** Returns the list of tower in a particular protojet */
   const std::vector<const aod::Candidate*> & getTowerList() const {return m_constituents;} 
   /** Sets the list of towers in a protojet */
  void putTowers(const std::vector<const aod::Candidate*> & towers) {
    m_constituents = towers;
    calculateLorentzVector(); 
   }

private:
  /** Jet constituents */
  std::vector<const aod::Candidate*> m_constituents;

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
