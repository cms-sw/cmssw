#ifndef JetReco_PFJet_h
#define JetReco_PFJet_h

/** \class reco::PFJet
 *
 * \short Jets made from PFObjects
 *
 * PFJet represents Jets made from Particle Flow objects
 * Provide energy contributions from different PF types
 * in addition to generic Jet parameters
 *
 * \author Fedor Ratnikov, UMd, Apr 24, 2007
  * \version   $Id: PFJet.h,v 1.7 2007/08/01 23:03:26 fedor Exp $
 ************************************************************/


#include "DataFormats/JetReco/interface/Jet.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockFwd.h"


namespace reco {
class PFJet : public Jet {
 public:
  struct Specific {
    Specific () :
	 mChargedHadronEnergy (0),
	 mNeutralHadronEnergy (0),
	 mChargedEmEnergy (0),
	 mChargedMuEnergy (0),
	 mNeutralEmEnergy (0),
	 mChargedMultiplicity (0),
	 mNeutralMultiplicity (0),
	 mMuonMultiplicity (0)
    {}
    float mChargedHadronEnergy;
    float mNeutralHadronEnergy;
    float mChargedEmEnergy;
    float mChargedMuEnergy;
    float mNeutralEmEnergy;
    int mChargedMultiplicity;
    int mNeutralMultiplicity;
    int mMuonMultiplicity;
 };
  
  /** Default constructor*/
  PFJet() {}
  
  /** Constructor from values*/
  PFJet(const LorentzVector& fP4, const Point& fVertex, const Specific& fSpecific, 
	  const Jet::Constituents& fConstituents);

  /** backward compatible, vertex=(0,0,0) */
  PFJet(const LorentzVector& fP4, const Specific& fSpecific, 
	  const Jet::Constituents& fConstituents);

  
  virtual ~PFJet() {};

  /// chargedHadronEnergy 
  float chargedHadronEnergy () const {return m_specific.mChargedHadronEnergy;}
  ///  chargedHadronEnergyFraction
  float  chargedHadronEnergyFraction () const {return chargedHadronEnergy () / energy ();}
  /// neutralHadronEnergy
  float neutralHadronEnergy () const {return m_specific.mNeutralHadronEnergy;}
  /// neutralHadronEnergyFraction
  float neutralHadronEnergyFraction () const {return neutralHadronEnergy () / energy ();}
  /// chargedEmEnergy
  float chargedEmEnergy () const {return m_specific.mChargedEmEnergy;}
  /// chargedEmEnergyFraction
  float chargedEmEnergyFraction () const {return chargedEmEnergy () / energy ();}
  /// chargedMuEnergy
  float chargedMuEnergy () const {return m_specific.mChargedMuEnergy;}
  /// chargedMuEnergyFraction
  float chargedMuEnergyFraction () const {return chargedMuEnergy () / energy ();}
  /// neutralEmEnergy
  float neutralEmEnergy () const {return m_specific.mNeutralEmEnergy;}
  /// neutralEmEnergyFraction
  float neutralEmEnergyFraction () const {return neutralEmEnergy () / energy ();}
  /// chargedMultiplicity
  float chargedMultiplicity () const {return m_specific.mChargedMultiplicity;}
  /// neutralMultiplicity
  float neutralMultiplicity () const {return m_specific.mNeutralMultiplicity;}
  /// muonMultiplicity
  float muonMultiplicity () const {return m_specific.mMuonMultiplicity;}

 
  /// convert generic constituent to specific type
  static reco::PFBlockRef getPFBlock (const reco::Candidate* fConstituent);
  /// get specific constituent
  reco::PFBlockRef getConstituent (unsigned fIndex) const;
  /// get all constituents
  std::vector <reco::PFBlockRef> getConstituents () const;
  
  // block accessors
  
  const Specific& getSpecific () const {return m_specific;}

  /// Polymorphic clone
  virtual PFJet* clone () const;

  /// Print object
  virtual std::string print () const;

 private:
  /// Polymorphic overlap
  virtual bool overlap( const Candidate & ) const;
  
  //Variables specific to to the PFJet class
  Specific m_specific;
};
}
// temporary fix before include_checcker runs globally
#include "DataFormats/JetReco/interface/PFJetCollection.h" //INCLUDECHECKER:SKIP
#endif
