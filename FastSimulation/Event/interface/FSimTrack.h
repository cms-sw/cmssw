#ifndef FastSimulation_Event_FSimTrack_H
#define FastSimulation_Event_FSimTrack_H

// CLHEP Headers
#include "CLHEP/Vector/LorentzVector.h"
#include "SimGeneral/HepPDTRecord/interface/ParticleDataTable.h"

// CMSSW Headers
#include "SimDataFormats/Track/interface/SimTrack.h"

// FAMOS headers
#include "FastSimulation/Particle/interface/RawParticle.h"

#include <map>

class FSimVertex;
class FBaseSimEvent;

namespace HepMC {
  class GenParticle;
}

/** A class that mimics SimTrack, with enhanced features.
 *  Essentially an interface to SimTrack.
 * \author Patrick Janot, CERN 
 * $Date: 9-Dec-2003
 */

class FSimTrack : public SimTrack {

 public:
  /// Default constructor
  FSimTrack();
  
  /// Constructor from the EmmbSimTrack index in the FBaseSimEvent
  FSimTrack(const RawParticle* p, int iv, int ig, int id, FBaseSimEvent* mom);
  
  /// Destructor
  virtual ~FSimTrack();

  /// particle info...
  const DefaultConfig::ParticleData* particleInfo() const;
  
  /// four momentum
  //inline HepLorentzVector momentum() const { return me().momentum(); }

  /// particle type (HEP PDT convension)
  //inline int type() const { return me().type(); }

  /// charge
  float charge() const; 

  /// Origin vertex
  const FSimVertex& vertex() const;

  /// end vertex
  const FSimVertex& endVertex() const;

  /// mother
  const FSimTrack& mother() const; 

  /// Ith daughter
  const FSimTrack& daughter(int i) const; 

  /// Number of daughters
  int nDaughters() const; 

  /// Vector of daughter indices
  const std::vector<int>& daughters() const;

  /// no origin vertex...
  //inline bool  noVertex() const { return me().noVertex(); }

  /// no end vertex
  bool  noEndVertex() const; 

  /// Compare the end vertex position with another position.
  bool notYetToEndVertex(const HepLorentzVector& pos) const;

  /// no mother particle
  bool  noMother() const;

  /// no daughters
  bool  noDaughter() const;

  /// The original GenParticle
  const HepMC::GenParticle* genParticle() const;
   
  /// The attached SimTrack
  //const SimTrack& me() const;

  /// The original GenParticle in the original event
  //inline int genpartIndex() const { return me().genpartIndex(); }

  /// the generator particle index
  //  short int genpartIndex() const { return me().genpartIndex(); }

  /// no generated particle
  //  bool  noGenpart() const { return me().noGenpart();}

  /// the embedded Simulated Track
  //  const FSimTrack & me() const { return mom->embdTrack(id()); }

  /// the index in FBaseSimEvent and other vectors
  inline int id() const { return id_; }

  /// the distance to a reconstructed track
  //  double distFromRecTrack(const TTrack&) const; 

  /// The particle was propagated to the Preshower Layer1
  /// 2 : on the EndCaps; (no Barrel Preshower); no propagation possible
  /// 0 : not yet propagated or no pe
  inline int onLayer1() const { return layer1; }

  /// The particle was propagated to the Preshower Layer2 
  /// 2 : on the EndCaps; (no Barrel Preshower); 3 : No propagation possible
  /// 0 : not yet propagated
  inline int onLayer2() const { return layer2; }

  /// The particle was propagated to the ECAL front face
  /// 1 : on the barrel; 2 : on the EndCaps; 3 : no propagation possible
  /// 0 : not yet propagated
  inline int onEcal() const { return ecal; }

  /// The particle was propagated to the HCAL front face
  /// 1 : on the barrel; 2 : on the EndCaps; 3 : no propagation possible
  /// 0 : not yet propagated
  inline int onHcal() const { return hcal; }

  /// The particle was propagated to the VFCAL front face
  /// 2 : on the EndCaps (No VFCAL Barrel); 3 : no propagation possible
  /// 0 : not yet propagated
  inline int onVFcal() const { return vfcal; }

  /// The particle was tentatively propagated to calorimeters
  inline bool propagated() const { return prop; }

  /// The particle at Preshower Layer 1
  inline const RawParticle& layer1Entrance() const { return Layer1_Entrance; }

  /// The particle at Preshower Layer 2
  inline const RawParticle& layer2Entrance() const { return Layer2_Entrance; }

  /// The particle at ECAL entrance
  inline const RawParticle& ecalEntrance() const { return ECAL_Entrance; }

  /// The particle at HCAL entrance
  inline const RawParticle& hcalEntrance() const { return HCAL_Entrance; }

  /// The particle at VFCAL entrance
  inline const RawParticle& vfcalEntrance() const { return VFCAL_Entrance; }

  /// The RecHits in the Tracker
  // const map<unsigned,const FamosBasicRecHit*>& recHits() const {return theRecHits;}

  /// The SimHits in the Tracker
  //  inline const std::map<unsigned,RawParticle>& simHits() const {return theSimHits;}

  /// Is there a RecHit on this layer?
  //  bool isARecHit(const unsigned layer) const;

  /// If yes, here it is.
  //  const FamosBasicRecHit* recHit(unsigned layer) const; 

  /// Is there a SimHit on this layer?
  //  bool isASimHit(const unsigned layer) const;

  /// If yes, here is the corresponding RawParticle
  //  const RawParticle& simHit(unsigned layer) const;

  /// Set the end vertex
  void setEndVertex(int endv) { endv_ = endv; } 

  /// The particle has been propgated through the tracker
  void setPropagate();

  /// Set the preshower layer1 variables
  void setLayer1(const RawParticle& pp, int success);

  /// Set the preshower layer2 variables
  void setLayer2(const RawParticle& pp, int success);

  /// Set the ecal variables
  void setEcal(const RawParticle& pp,int success);

  /// Set the hcal variables
  void setHcal(const RawParticle& pp, int success);

  /// Set the hcal variables
  void setVFcal(const RawParticle& pp, int success);

  /// Add a RecHit for a track on a layer
  //  void addRecHit(const FamosBasicRecHit* hit, unsigned layer);

  /// Add a RecHit for a track on a layer
  //  void addSimHit(const RawParticle& pp, unsigned layer);

  /// Update the vactors of daughter's id
  inline void addDaughter(int i) { daugh_.push_back(i); }

 private:

  //  HepMC::GenParticle* me_;

  FBaseSimEvent* mom_;
  //  int embd_;   // The index in the SimTrack vector
  int id_; // The index in the FSimTrackVector
  
  int endv_; // The index of the end vertex in FSimVertex

  int layer1;// 1 if the particle was propagated to preshower layer1
  int layer2;// 1 if the particle was propagated to preshower layer2
  int ecal;  // 1 if the particle was propagated to ECAL/HCAL barrel
  int hcal;  // 2 if the particle was propagated to ECAL/HCAL endcap 
  int vfcal; // 1 if the particle was propagated to VFCAL 

  bool prop;     // true if the propagation to the calorimeters was done

  RawParticle Layer1_Entrance; // the particle at preshower Layer1
  RawParticle Layer2_Entrance; // the particle at preshower Layer2
  RawParticle ECAL_Entrance;   // the particle at ECAL entrance
  RawParticle HCAL_Entrance;   // the particle at HCAL entrance
  RawParticle VFCAL_Entrance;  // the particle at VFCAL entrance

  std::vector<int> daugh_; // The indices of the daughters in FSimTrack

  //  std::map<unsigned,const FamosBasicRecHit*> theRecHits;
  //  std::map<unsigned,RawParticle> theSimHits;
};

#include<iosfwd>
std::ostream& operator <<(std::ostream& o , const FSimTrack& t);

#endif // FSimTrack_H





