#ifndef FSimTrack_H
#define FSimTrack_H

#include<cmath>
#include <CLHEP/Vector/LorentzVector.h>
#include "FastSimulation/Event/interface/FBaseSimEvent.h"
#include "FastSimulation/Particle/interface/RawParticle.h"

#include <map>

class FSimVertex;
class HepParticleData;

/** A class that mimics SimTrack, with enhanced features.
 *  Essentially an interface to FEmbdSimTrack.
 * \author Patrick Janot, CERN 
 * $Date: 9-Dec-2003
 */

class FSimTrack {

 public:
  /// Default constructor
  FSimTrack() : 
    me_(0), mom_(0), gen_(-1),
    layer1(0), layer2(0), ecal(0), hcal(0), vfcal(0), prop(false) {;}
  
  /// Constructor from the GenParticle index in the FBaseSimEvent
  FSimTrack(HepMC::GenParticle* ime, FBaseSimEvent* mom, int gen=-1) : 
    me_(ime), mom_(mom), gen_(gen),
    layer1(0), layer2(0), ecal(0), hcal(0), vfcal(0), prop(false) {;}
  
  /// Destructor
  virtual ~FSimTrack();

  /// particle info...
  const HepParticleData * particleInfo() const;
  
  /// four momentum
  inline HepLorentzVector momentum() const { 
    return me() ? me()->momentum() : HepLorentzVector();}

  /// particle type (HEP PDT convension)
  inline int type() const { return me() ? me()->pdg_id() : 0;}

  /// charge
  //  float charge() const{ return me().charge();}
  
  /// Origin vertex
  const FSimVertex& vertex() const;

  /// end vertex
  const FSimVertex& endVertex() const;

  /// mother
  const FSimTrack& mother() const; 

  /// first daughter
  const FSimTrack& daughter1() const;

  /// last daughter
  const FSimTrack& daughter2() const;

  /// no origin vertex...
  bool  noVertex() const { return !me() || !me()->production_vertex(); }

  /// no end vertex
  bool  noEndVertex() const { return !me() || !me()->end_vertex(); }

  /// Compare the end vertex position with another position.
  bool notYetToEndVertex(const HepLorentzVector& pos) const;

  /// no mother particle
  bool  noMother() const { return !me() || !me()->mother(); }

  /// no daughters
  bool  noDaughter() const { return !me() || !me()->listChildren().size(); }

  /// The original GenParticle
  const HepMC::GenParticle* me() const { return me_; }

  /// The original GenParticle in the original event
  int genpartIndex() const { return gen_; }

  /// the generator particle index
  //  short int genpartIndex() const { return me().genpartIndex(); }

  /// no generated particle
  //  bool  noGenpart() const { return me().noGenpart();}

  /// the embedded Simulated Track
  //  const FEmbdSimTrack & me() const { return mom->embdTrack(id()); }

  /// the index in FBaseSimEvent and other vectors
  int id() const { return me() ? me()->barcode()-1 : -1; }

  /// the distance to a reconstructed track
  //  double distFromRecTrack(const TTrack&) const; 

  /// The particle was propagated to the Preshower Layer1
  /// 2 : on the EndCaps; (no Barrel Preshower); no propagation possible
  /// 0 : not yet propagated or no pe
  int onLayer1() const { return layer1; }

  /// The particle was propagated to the Preshower Layer2 
  /// 2 : on the EndCaps; (no Barrel Preshower); 3 : No propagation possible
  /// 0 : not yet propagated
  int onLayer2() const { return layer2; }

  /// The particle was propagated to the ECAL front face
  /// 1 : on the barrel; 2 : on the EndCaps; 3 : no propagation possible
  /// 0 : not yet propagated
  int onEcal() const { return ecal; }

  /// The particle was propagated to the HCAL front face
  /// 1 : on the barrel; 2 : on the EndCaps; 3 : no propagation possible
  /// 0 : not yet propagated
  int onHcal() const { return hcal; }

  /// The particle was propagated to the VFCAL front face
  /// 2 : on the EndCaps (No VFCAL Barrel); 3 : no propagation possible
  /// 0 : not yet propagated
  int onVFcal() const { return vfcal; }

  /// The particle was tentatively propagated to calorimeters
  bool propagated() const { return prop; }

  /// The particle at Preshower Layer 1
  const RawParticle& layer1Entrance() const { return Layer1_Entrance; }

  /// The particle at Preshower Layer 2
  const RawParticle& layer2Entrance() const { return Layer2_Entrance; }

  /// The particle at ECAL entrance
  const RawParticle& ecalEntrance() const { return ECAL_Entrance; }

  /// The particle at HCAL entrance
  const RawParticle& hcalEntrance() const { return HCAL_Entrance; }

  /// The particle at VFCAL entrance
  const RawParticle& vfcalEntrance() const { return VFCAL_Entrance; }

  /// The RecHits in the Tracker
  // const map<unsigned,const FamosBasicRecHit*>& recHits() const {return theRecHits;}

  /// The SimHits in the Tracker
  const std::map<unsigned,RawParticle>& simHits() const {return theSimHits;}

  /// Is there a RecHit on this layer?
  //  bool isARecHit(const unsigned layer) const;

  /// If yes, here it is.
  //  const FamosBasicRecHit* recHit(unsigned layer) const; 

  /// Is there a SimHit on this layer?
  bool isASimHit(const unsigned layer) const;

  /// If yes, here is the corresponding RawParticle
  const RawParticle& simHit(unsigned layer) const;

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
  void addSimHit(const RawParticle& pp, unsigned layer);

 private:

  const HepMC::GenParticle* me_;
  const FBaseSimEvent* mom_;
  int gen_;

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

  //  std::map<unsigned,const FamosBasicRecHit*> theRecHits;
  std::map<unsigned,RawParticle> theSimHits;
};

#include<iosfwd>
std::ostream& operator <<(std::ostream& o , const FSimTrack& t);

#endif // FSimTrack_H





