#ifndef RecoTauTag_RecoTau_RecoTauConstructor_h
#define RecoTauTag_RecoTau_RecoTauConstructor_h

/*
 * RecoTauConstructor
 *
 * A generalized builder of reco::PFTau objects.  Takes a variety of
 * different collections and converts them to the proper Ref format
 * needed for PFTau storage.  Automatically sets the p4, charge, and
 * other properties correctly.  Optionally, it can determine the
 * lead track information, and copy the gamma candidates owned by the
 * reconstructed pi zeros into the appropriate PiZero collection.
 *
 * If the gammas are copied from the PiZeros, the four vector will be
 * built using the PiZeros + Charged Hadrons.  This can be different than
 * the Gammas + Charged Hadrons, as the PiZero may have a mass hypothesis set by
 * the RecoTauPiZeroProducer
 *
 * Note that the p4 of the tau is *always* set as the sum of objects in
 * signal cone.
 *
 * Author: Evan K. Friis, UC Davis
 *
 * $Id $
 */

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"
#include "DataFormats/TauReco/interface/PFTau.h"
#include "DataFormats/JetReco/interface/PFJetCollection.h"
#include "DataFormats/TauReco/interface/PFRecoTauChargedHadron.h"
#include "DataFormats/TauReco/interface/RecoTauPiZero.h"

#include <vector>

namespace reco { namespace tau {

class RecoTauConstructor {
  public:
    enum Region {
      kSignal,
      kIsolation
    };

    enum ParticleType {
      kChargedHadron,
      kGamma,
      kNeutralHadron,
      kAll
    };

    /// Constructor with PFCandidate Handle
    RecoTauConstructor(const PFJetRef& jetRef,
        const edm::Handle<PFCandidateCollection>& pfCands,
        bool copyGammasFromPiZeros=false);

    /*
     * Code to set leading candidates.  These are just wrappers about
     * the embedded taus methods, but with the ability to convert Ptrs
     * to Refs.
     */

    /// Set leading PFChargedHadron candidate
    template<typename T> void setleadPFChargedHadrCand(const T& cand) {
      tau_->setleadPFChargedHadrCand(convertToPtr(cand));
    }

    /// Set leading PFGamma candidate
    template<typename T> void setleadPFNeutralCand(const T& cand) {
      tau_->setleadPFNeutralCand(convertToPtr(cand));
    }

    /// Set leading PF candidate
    template<typename T> void setleadPFCand(const T& cand) {
      tau_->setleadPFCand(convertToPtr(cand));
    }

    /// Append a PFCandidateRef/Ptr to a given collection
    void addPFCand(Region region, ParticleType type, const PFCandidateRef& ref, bool skipAddToP4 = false);
    void addPFCand(Region region, ParticleType type, const PFCandidatePtr& ptr, bool skipAddToP4 = false);

    /// Reserve a set amount of space for a given RefVector
    void reserve(Region region, ParticleType type, size_t size);

    // Add a collection of objects to a given collection
    template<typename InputIterator>
    void addPFCands(Region region, ParticleType type, const InputIterator& begin, const InputIterator& end) 
    {
      for(InputIterator iter = begin; iter != end; ++iter) {
	addPFCand(region, type, convertToPtr(*iter));
      }
    }
    
    /// Reserve a set amount of space for the ChargedHadrons
    void reserveTauChargedHadron(Region region, size_t size);

    /// Add a ChargedHadron to the given collection
    void addTauChargedHadron(Region region, const PFRecoTauChargedHadron& chargedHadron);

    /// Add a list of charged hadrons to the input collection
    template<typename InputIterator> void addTauChargedHadrons(Region region, const InputIterator& begin, const InputIterator& end)
    {
      for ( InputIterator iter = begin; iter != end; ++iter ) {
	addTauChargedHadron(region, *iter);
      }
    }

    /// Reserve a set amount of space for the PiZeros
    void reservePiZero(Region region, size_t size);

    /// Add a PiZero to the given collection
    void addPiZero(Region region, const RecoTauPiZero& piZero);

    /// Add a list of pizeros to the input collection
    template<typename InputIterator> void addPiZeros(Region region, const InputIterator& begin, const InputIterator& end)
    {
      for ( InputIterator iter = begin; iter != end; ++iter ) {
	addPiZero(region, *iter);
      }
    }

    // Build and return the associated tau
    std::auto_ptr<reco::PFTau> get(bool setupLeadingCandidates=true);

    // Get the four vector of the signal objects added so far
    const reco::Candidate::LorentzVector& p4() const { return p4_; }

  private:
    typedef std::pair<Region, ParticleType> CollectionKey;
    typedef std::map<CollectionKey, std::vector<PFCandidatePtr>*> CollectionMap;
    typedef boost::shared_ptr<std::vector<PFCandidatePtr> > SortedListPtr;
    typedef std::map<CollectionKey, SortedListPtr> SortedCollectionMap;

    bool copyGammas_;
    // Retrieve collection associated to signal/iso and type
    std::vector<PFCandidatePtr>* getCollection(Region region, ParticleType type);
    SortedListPtr getSortedCollection(Region region, ParticleType type);

    // Sort all our collections by PT and copy them into the tau
    void sortAndCopyIntoTau();

    // Helper functions for dealing with refs
    PFCandidatePtr convertToPtr(const PFCandidatePtr& pfPtr) const;
    PFCandidatePtr convertToPtr(const CandidatePtr& candPtr) const;
    PFCandidatePtr convertToPtr(const PFCandidateRef& pfRef) const;

    const edm::Handle<PFCandidateCollection>& pfCands_;
    std::auto_ptr<reco::PFTau> tau_;
    CollectionMap collections_;

    // Keep sorted (by descending pt) collections
    SortedCollectionMap sortedCollections_;

    // Keep track of the signal cone four vector in case we want it
    reco::Candidate::LorentzVector p4_;
};
} } // end reco::tau namespace
#endif
