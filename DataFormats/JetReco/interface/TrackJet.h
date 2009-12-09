#ifndef DataFormats_JetReco_TrackJet_h
#define DataFormats_JetReco_TrackJet_h


/** \class reco::TrackJet
 *
 * \short Jets made out of tracks
 *
 * TrackJet represents Jets with tracks as constituents.
 * The usual way of jets having candidates as constituents is preserved
 * through a caching mechanism with transient pointers, such that all
 * usual jet shape methods, which rely on the getJetConstituents() method,
 * are still functional.
 *
 * \author Steven Lowette
 *
 * $Id: TrackJet.h,v 1.1 2009/11/25 19:07:49 srappocc Exp $
 *
 ************************************************************/


#include "DataFormats/JetReco/interface/Jet.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedRefCandidate.h"


namespace reco {

  class TrackJet : public Jet {

    public:

      /// Default constructor
      TrackJet();
      /// Constructor without constituents
      TrackJet(const LorentzVector & fP4, const Point & fVertex);
      /// Constructor from RecoChargedRefCandidate constituents
      TrackJet(const LorentzVector & fP4, const Point & fVertex, const Jet::Constituents & fConstituents);
      /// Destructor
      virtual ~TrackJet() {}
      /// Polymorphic clone
      virtual TrackJet * clone () const;

      /// Number of track daughters
      virtual size_t numberOfTracks() const { return numberOfDaughters(); }
      /// Return Ptr to the track costituent
      virtual edm::Ptr<reco::Track> track(size_t i) const;
      /// Return pointers to all track costituents
      virtual std::vector<edm::Ptr<reco::Track> > tracks() const;

      /// calculate and set the charge by adding up the constituting track charges
      void resetCharge();
      /// Associated PV
      // to be implemented

      /// Print object
      virtual std::string print () const;

  private:
      /// Polymorphic overlap
      virtual bool overlap(const Candidate & dummy) const;

    private:


  };

}

#endif
