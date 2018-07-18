#ifndef FastSimulation_Event_FSimTrack_H
#define FastSimulation_Event_FSimTrack_H

// HepPDT Headers
#include "SimGeneral/HepPDTRecord/interface/ParticleDataTable.h"

// CMSSW Headers
#include "SimDataFormats/Track/interface/SimTrack.h"
#include "SimDataFormats/Vertex/interface/SimVertex.h"

// FAMOS headers
#include "FastSimulation/Particle/interface/RawParticle.h"

#include <vector>

class FSimVertex;
class FBaseSimEvent;

namespace HepMC {
  class GenParticle;
  class GenVertex;
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
  FSimTrack(const RawParticle* p, int iv, int ig, int id, FBaseSimEvent* mom, double dt=-1.);

  //! Hack to interface "old" calorimetry with "new" propagation in tracker (need to construct FSimTracks)
  FSimTrack(int ipart, const math::XYZTLorentzVector& p, int iv, int ig, int id, double charge, const math::XYZTLorentzVector& tkp, const math::XYZTLorentzVector& tkm, const SimVertex& tkv);
  
  /// Destructor
  virtual ~FSimTrack();

  /// particle info...
  inline const HepPDT::ParticleData* particleInfo() const {
    return info_;
  }
  
  /// charge
  inline float charge() const { 
    if(particleInfo() == nullptr) return charge_; 
    return particleInfo()->charge();
  }
  

  /// Origin vertex
  inline const FSimVertex vertex() const;

  /// end vertex
  inline const FSimVertex& endVertex() const;

  /// mother
  inline const FSimTrack& mother() const;

  /// Ith daughter
  inline const FSimTrack& daughter(int i) const;

  /// Number of daughters
  inline int nDaughters() const;

  /// Vector of daughter indices
  inline const std::vector<int>& daughters() const;

  /// no end vertex
  inline bool  noEndVertex() const;

  /// Compare the end vertex position with another position.
  bool notYetToEndVertex(const XYZTLorentzVector& pos) const;

  /// no mother particle
  inline bool  noMother() const; 

  /// no daughters
  inline bool  noDaughter() const;

  /// The original GenParticle
  inline const HepMC::GenParticle* genParticle() const;
   
  /// the index in FBaseSimEvent and other vectors
  inline int id() const { return id_; }

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
  
  /// The particle was propagated to the HCAL back face
  /// 1 : on the barrel; 2 : on the EndCaps; 3 : no propagation possible
  /// 0 : not yet propagated
  inline int outHcal() const { return hcalexit; }

  //The particle was propagated to the HO front face
  /// 1 : on the barrel; 2 : on the EndCaps; 3 : no propagation possible
  /// 0 : not yet propagated
  inline int onHO() const { return hoentr; }

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

  /// The particle at HCAL exir
  inline const RawParticle& hcalExit() const { return HCAL_Exit; }

  /// The particle at HCAL exir
  inline const RawParticle& hoEntrance() const { return HO_Entrance; }

  /// particle did not decay before more detectors (useful for newProducer)
  inline bool isGlobal() const { return isGlobal_; }

  /// particle did not decay before more detectors (useful for newProducer)
  inline void setGlobal() { isGlobal_ = true; }

  /// Set origin vertex
  inline void setOriginVertex(const SimVertex& v) { vertex_ = v; } 

  /// Set the end vertex
  inline void setEndVertex(int endv) { endv_ = endv; } 

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

  /// Set the hcal exit variables
  void setHcalExit(const RawParticle& pp, int success);

  /// Set the ho variables
  void setHO(const RawParticle& pp, int success);

  /// Add a RecHit for a track on a layer
  //  void addRecHit(const FamosBasicRecHit* hit, unsigned layer);

  /// Add a RecHit for a track on a layer
  //  void addSimHit(const RawParticle& pp, unsigned layer);

  /// Update the vactors of daughter's id
  inline void addDaughter(int i) { daugh_.push_back(i); }

  /// Set the index of the closest charged daughter
  inline void setClosestDaughterId(int id) { closestDaughterId_ = id; }

  /// Get the index of the closest charged daughter
  inline int closestDaughterId() const { return closestDaughterId_; }

  /// Temporary (until move of SimTrack to Mathcore) - No! Actually very useful
  const XYZTLorentzVector& momentum() const { return momentum_; }

  /// Reset the momentum (to be used with care)
  inline void setMomentum(const math::XYZTLorentzVector& newMomentum) {momentum_ = newMomentum; }

  /// Simply returns the SimTrack
  inline const SimTrack& simTrack() const { return *this; }

  /// Return the pre-defined decay time
  inline double decayTime() const { return properDecayTime; }

 private:

  //  HepMC::GenParticle* me_;
  SimVertex vertex_;

  FBaseSimEvent* mom_;
  //  int embd_;   // The index in the SimTrack vector
  int id_; // The index in the FSimTrackVector
  double charge_; // Charge of the particle
  
  int endv_; // The index of the end vertex in FSimVertex

  int layer1;// 1 if the particle was propagated to preshower layer1
  int layer2;// 1 if the particle was propagated to preshower layer2
  int ecal;  // 1 if the particle was propagated to ECAL/HCAL barrel
  int hcal;  // 2 if the particle was propagated to ECAL/HCAL endcap 
  int vfcal; // 1 if the particle was propagated to VFCAL 
  int hcalexit;  // 2 if the particle was propagated to HCAL Exit point
  int hoentr; // 1 if the particle was propagated to HO 


  bool prop;     // true if the propagation to the calorimeters was done

  RawParticle Layer1_Entrance; // the particle at preshower Layer1
  RawParticle Layer2_Entrance; // the particle at preshower Layer2
  RawParticle ECAL_Entrance;   // the particle at ECAL entrance
  RawParticle HCAL_Entrance;   // the particle at HCAL entrance
  RawParticle VFCAL_Entrance;  // the particle at VFCAL entrance
  RawParticle HCAL_Exit;       // the particle at HCAL ezit point
  RawParticle HO_Entrance;     // the particle at HO entrance


  std::vector<int> daugh_; // The indices of the daughters in FSimTrack
  int closestDaughterId_; // The index of the closest daughter id

  const HepPDT::ParticleData* info_; // The PDG info

  XYZTLorentzVector momentum_;

  double properDecayTime; // The proper decay time  (default is -1)

  bool isGlobal_; // needed for interfacing the new particle propagator with the old muon sim hit code

};

#include<iosfwd>
std::ostream& operator <<(std::ostream& o , const FSimTrack& t);

#include "FastSimulation/Event/interface/FSimTrack.icc"



#endif // FSimTrack_H





