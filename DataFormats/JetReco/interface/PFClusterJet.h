#ifndef DataFormats_JetReco_PFClusterJet_h
#define DataFormats_JetReco_PFClusterJet_h


/** \class reco::PFClusterJet
 *
 * \short Jets made out of PFClusters
 *
 * PFClusterJet represents Jets with PFClusters as constituents.
 * The consitutents are in this case RecoPFClusterRefCandidates, that
 * preserve the Refs to the original PFClusters. Those Refs are used to
 * provide transparent access to the PFClusters.
 *
 * \author Salvatore Rappoccio
 *
 * $Id: PFClusterJet.h,v 1.2 2012/03/21 22:09:44 slava77 Exp $
 *
 ************************************************************/


#include "DataFormats/JetReco/interface/Jet.h"
#include "DataFormats/ParticleFlowReco/interface/RecoPFClusterRefCandidate.h"


namespace reco {

  class PFClusterJet : public Jet {

    public:

      /// Default constructor
      PFClusterJet();
      /// Constructor without constituents
      PFClusterJet(const LorentzVector & fP4, const Point & fVertex );
      /// Constructor from RecoChargedRefCandidate constituents
      PFClusterJet(const LorentzVector & fP4, const Point & fVertex, const Jet::Constituents & fConstituents);
      /// Destructor
      virtual ~PFClusterJet() {}
      /// Polymorphic clone
      virtual PFClusterJet * clone () const;

      /// Print object
      virtual std::string print () const;
      
      /// Easy Constituent access
      reco::PFClusterRef   pfCluster( size_t i) const;

    private:

      /// Polymorphic overlap
      virtual bool overlap(const Candidate & dummy) const;

    private:


  };

}

#endif
