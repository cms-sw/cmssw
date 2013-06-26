#ifndef DataFormats_JetReco_TrackJet_h
#define DataFormats_JetReco_TrackJet_h


/** \class reco::TrackJet
 *
 * \short Jets made out of tracks
 *
 * TrackJet represents Jets with tracks as constituents.
 * The consitutents are in this case RecoChargedRefCandidates, that
 * preserve the Refs to the original tracks. Those Refs are used to
 * provide transparent access to the tracks.
 *
 * \author Steven Lowette
 *
 * $Id: TrackJet.h,v 1.4 2010/06/29 09:56:50 lowette Exp $
 *
 ************************************************************/


#include "DataFormats/JetReco/interface/Jet.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedRefCandidate.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"


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
      size_t numberOfTracks() const { return numberOfDaughters(); }
      /// Return Ptr to the track costituent
      virtual edm::Ptr<reco::Track> track(size_t i) const;
      /// Return pointers to all track costituents
      std::vector<edm::Ptr<reco::Track> > tracks() const;

      /// calculate and set the charge by adding up the constituting track charges
      void resetCharge();
      /// get associated primary vertex
      const reco::VertexRef primaryVertex() const;
      /// set associated primary vertex
      void setPrimaryVertex(const reco::VertexRef & vtx);
      /// check jet to be associated to the hard primary vertex
      bool fromHardVertex() const { return (this->primaryVertex().index() == 0); }

      /// Print object
      virtual std::string print () const;

    private:

      /// Polymorphic overlap
      virtual bool overlap(const Candidate & dummy) const;

    private:

      /// Associated primary vertex
      reco::VertexRef vtx_;

  };

}

#endif
