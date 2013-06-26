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
  * \version   $Id: PFJet.h,v 1.21 2013/05/01 13:53:47 srappocc Exp $
 ************************************************************/


#include "DataFormats/JetReco/interface/Jet.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"

namespace reco {
class PFJet : public Jet {
 public:

    typedef reco::PFCandidatePtr ConstituentTypePtr;
    typedef reco::PFCandidateFwdPtr ConstituentTypeFwdPtr;

  struct Specific {
    Specific () :
       mChargedHadronEnergy (0),
       mNeutralHadronEnergy (0),
       mPhotonEnergy (0),
       mElectronEnergy (0),
       mMuonEnergy (0),
       mHFHadronEnergy (0),
       mHFEMEnergy (0),

       mChargedHadronMultiplicity (0),
       mNeutralHadronMultiplicity (0),
       mPhotonMultiplicity (0),
       mElectronMultiplicity (0),
       mMuonMultiplicity (0),
       mHFHadronMultiplicity (0),
       mHFEMMultiplicity (0),

       mChargedEmEnergy (0),
       mChargedMuEnergy (0),
       mNeutralEmEnergy (0),
       
       mChargedMultiplicity (0),
       mNeutralMultiplicity (0)
    {}
    float mChargedHadronEnergy;
    float mNeutralHadronEnergy;
    float mPhotonEnergy;
    float mElectronEnergy;
    float mMuonEnergy;
    float mHFHadronEnergy;
    float mHFEMEnergy;

    int mChargedHadronMultiplicity;
    int mNeutralHadronMultiplicity;
    int mPhotonMultiplicity;
    int mElectronMultiplicity;
    int mMuonMultiplicity;
    int mHFHadronMultiplicity;
    int mHFEMMultiplicity;

    //old (deprecated) data members
    //kept only for backwards compatibility:
    float mChargedEmEnergy;
    float mChargedMuEnergy;
    float mNeutralEmEnergy;
    int mChargedMultiplicity;
    int mNeutralMultiplicity;
 };
  
  /** Default constructor*/
  PFJet() {}
  
  /** Constructor from values*/
  PFJet(const LorentzVector& fP4, const Point& fVertex, const Specific& fSpecific, 
	  const Jet::Constituents& fConstituents);

  PFJet(const LorentzVector& fP4, const Point& fVertex, const Specific& fSpecific); 

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
  /// photonEnergy 
  float photonEnergy () const {return m_specific.mPhotonEnergy;}
  /// photonEnergyFraction
  float photonEnergyFraction () const {return photonEnergy () / energy ();}
  /// electronEnergy 
  float electronEnergy () const {return m_specific.mElectronEnergy;}
  /// electronEnergyFraction
  float electronEnergyFraction () const {return electronEnergy () / energy ();}
  /// muonEnergy 
  float muonEnergy () const {return m_specific.mMuonEnergy;}
  /// muonEnergyFraction
  float muonEnergyFraction () const {return muonEnergy () / energy ();}
  /// HFHadronEnergy 
  float HFHadronEnergy () const {return m_specific.mHFHadronEnergy;}
  /// HFHadronEnergyFraction
  float HFHadronEnergyFraction () const {return HFHadronEnergy () / energy ();}
  /// HFEMEnergy 
  float HFEMEnergy () const {return m_specific.mHFEMEnergy;}
  /// HFEMEnergyFraction
  float HFEMEnergyFraction () const {return HFEMEnergy () / energy ();}

  /// chargedHadronMultiplicity
  int chargedHadronMultiplicity () const {return m_specific.mChargedHadronMultiplicity;}
  /// neutralHadronMultiplicity
  int neutralHadronMultiplicity () const {return m_specific.mNeutralHadronMultiplicity;}
  /// photonMultiplicity
  int photonMultiplicity () const {return m_specific.mPhotonMultiplicity;}
  /// electronMultiplicity
  int electronMultiplicity () const {return m_specific.mElectronMultiplicity;}
  /// muonMultiplicity
  int muonMultiplicity () const {return m_specific.mMuonMultiplicity;}
  /// HFHadronMultiplicity
  int HFHadronMultiplicity () const {return m_specific.mHFHadronMultiplicity;}
  /// HFEMMultiplicity
  int HFEMMultiplicity () const {return m_specific.mHFEMMultiplicity;}

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
  int chargedMultiplicity () const {return m_specific.mChargedMultiplicity;}
  /// neutralMultiplicity
  int neutralMultiplicity () const {return m_specific.mNeutralMultiplicity;}


  /// get specific constituent
  virtual reco::PFCandidatePtr getPFConstituent (unsigned fIndex) const;

  /// get all constituents
  virtual std::vector <reco::PFCandidatePtr> getPFConstituents () const;

  /// \ brief get all tracks in the jets
  /// All PFCandidates hold a reference to a track. All the non-null
  /// references are added to the returned TrackRefVector
  reco::TrackRefVector getTrackRefs() const; 
  
  // block accessors
  
  const Specific& getSpecific () const {return m_specific;}

  /// Polymorphic clone
  virtual PFJet* clone () const;

  /// Print object in details
  virtual std::string print () const;


 private:
  /// Polymorphic overlap
  virtual bool overlap( const Candidate & ) const;
  
  //Variables specific to to the PFJet class
  Specific m_specific;
};

// streamer
 std::ostream& operator<<(std::ostream& out, const reco::PFJet& jet);
}
// temporary fix before include_checcker runs globally
#include "DataFormats/JetReco/interface/PFJetCollection.h" //INCLUDECHECKER:SKIP 
#endif
