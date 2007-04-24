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
  * \version   $Id: PFJet.h,v 1.21 2007/02/22 19:17:35 fedor Exp $
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
	 mNutralHadronPt (0),
	 mChargedEmPt (0),
	 mNutralEmPt (0),
	 mChargedMultiplicity (0),
	 mNeutralMultiplicity (0)
    {}
    float mChargedHadronPt;
    float mNutralHadronPt;
    float mChargedEmPt;
    float mNutralEmPt;
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

  double chargedHadronPt () {return m_specific.mChargedHadronPt;}
  double neutralHadronPt () {return m_specific.mNeutralHadronPt;}
  double chargedEmPt () {return m_specific.mChargedEmPt;}
  double neutralEmPt () {return m_specific.mNeutralEmPt;}
  double chargedMultiplicity () {return m_specific.mChargedMultiplicity;}
  double neutralMultiplicity () {return m_specific.mNeutralMultiplicity;}

 
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
