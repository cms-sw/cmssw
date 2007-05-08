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
  * \version   $Id: PFJet.h,v 1.1 2007/04/24 21:53:08 fedor Exp $
 ************************************************************/


#include "DataFormats/JetReco/interface/Jet.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockFwd.h"

#include "DataFormats/JetReco/interface/PFJetfwd.h"

namespace reco {
class PFJet : public Jet {
 public:
  struct Specific {
    Specific () :
	 mChargedHadronPt (0),
	 mNeutralHadronPt (0),
	 mChargedEmPt (0),
	 mChargedMuPt (0),
	 mNeutralEmPt (0),
	 mChargedMultiplicity (0),
	 mNeutralMultiplicity (0)
    {}
    float mChargedHadronPt;
    float mNeutralHadronPt;
    float mChargedEmPt;
    float mChargedMuPt;
    float mNeutralEmPt;
    int mChargedMultiplicity;
    int mNeutralMultiplicity;
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

  /// chargedHadronPt 
  float chargedHadronPt () const {return m_specific.mChargedHadronPt;}
  ///  chargedHadronPtFraction
  float  chargedHadronPtFraction () const {return chargedHadronPt () / pt ();}
  /// neutralHadronPt
  float neutralHadronPt () const {return m_specific.mNeutralHadronPt;}
  /// neutralHadronPtFraction
  float neutralHadronPtFraction () const {return neutralHadronPt () / pt ();}
  /// chargedEmPt
  float chargedEmPt () const {return m_specific.mChargedEmPt;}
  /// chargedEmPtFraction
  float chargedEmPtFraction () const {return chargedEmPt () / pt ();}
  /// chargedMuPt
  float chargedMuPt () const {return m_specific.mChargedMuPt;}
  /// chargedMuPtFraction
  float chargedMuPtFraction () const {return chargedMuPt () / pt ();}
  /// neutralEmPt
  float neutralEmPt () const {return m_specific.mNeutralEmPt;}
  /// neutralEmPtFraction
  float neutralEmPtFraction () const {return neutralEmPt () / pt ();}
  /// chargedMultiplicity
  float chargedMultiplicity () const {return m_specific.mChargedMultiplicity;}
  /// neutralMultiplicity
  float neutralMultiplicity () const {return m_specific.mNeutralMultiplicity;}

 
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
#endif
