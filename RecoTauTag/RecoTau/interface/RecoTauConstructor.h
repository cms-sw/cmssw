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
 */

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"
#include "DataFormats/TauReco/interface/PFTau.h"
#include "DataFormats/JetReco/interface/JetCollection.h"
#include "DataFormats/TauReco/interface/PFRecoTauChargedHadron.h"
#include "DataFormats/TauReco/interface/RecoTauPiZero.h"
#include "CommonTools/Utils/interface/StringObjectFunction.h"

#include <vector>

namespace reco {
  namespace tau {

    class RecoTauConstructor {
    public:
      enum Region { kSignal, kIsolation };

      enum ParticleType { kChargedHadron, kGamma, kNeutralHadron, kAll };

      /// Constructor with PFCandidate Handle
      RecoTauConstructor(const JetBaseRef& jetRef,
                         const edm::Handle<edm::View<reco::Candidate> >& pfCands,
                         bool copyGammasFromPiZeros = false,
                         const StringObjectFunction<reco::PFTau>* signalConeSize = nullptr,
                         double minAbsPhotonSumPt_insideSignalCone = 2.5,
                         double minRelPhotonSumPt_insideSignalCone = 0.,
                         double minAbsPhotonSumPt_outsideSignalCone = 1.e+9,
                         double minRelPhotonSumPt_outsideSignalCone = 1.e+9);

      /*
     * Code to set leading candidates.  These are just wrappers about
     * the embedded taus methods, but with the ability to convert Ptrs
     * to Refs.
     */

      /// Set leading PFChargedHadron candidate
      template <typename T>
      void setleadChargedHadrCand(const T& cand) {
        tau_->setleadChargedHadrCand(convertToPtr(cand));
      }

      /// Set leading PFGamma candidate
      template <typename T>
      void setleadNeutralCand(const T& cand) {
        tau_->setleadNeutralCand(convertToPtr(cand));
      }

      /// Set leading PF candidate
      template <typename T>
      void setleadCand(const T& cand) {
        tau_->setleadCand(convertToPtr(cand));
      }

      /// Append a PFCandidateRef/Ptr to a given collection
      void addPFCand(Region region, ParticleType type, const CandidatePtr& ptr, bool skipAddToP4 = false);

      /// Reserve a set amount of space for a given RefVector
      void reserve(Region region, ParticleType type, size_t size);

      // Add a collection of objects to a given collection
      template <typename InputIterator>
      void addPFCands(Region region, ParticleType type, const InputIterator& begin, const InputIterator& end) {
        for (InputIterator iter = begin; iter != end; ++iter) {
          addPFCand(region, type, convertToPtr(*iter));
        }
      }

      /// Reserve a set amount of space for the ChargedHadrons
      void reserveTauChargedHadron(Region region, size_t size);

      /// Add a ChargedHadron to the given collection
      void addTauChargedHadron(Region region, const PFRecoTauChargedHadron& chargedHadron);

      /// Add a list of charged hadrons to the input collection
      template <typename InputIterator>
      void addTauChargedHadrons(Region region, const InputIterator& begin, const InputIterator& end) {
        for (InputIterator iter = begin; iter != end; ++iter) {
          addTauChargedHadron(region, *iter);
        }
      }

      /// Reserve a set amount of space for the PiZeros
      void reservePiZero(Region region, size_t size);

      /// Add a PiZero to the given collection
      void addPiZero(Region region, const RecoTauPiZero& piZero);

      /// Add a list of pizeros to the input collection
      template <typename InputIterator>
      void addPiZeros(Region region, const InputIterator& begin, const InputIterator& end) {
        for (InputIterator iter = begin; iter != end; ++iter) {
          addPiZero(region, *iter);
        }
      }

      // Build and return the associated tau
      std::unique_ptr<reco::PFTau> get(bool setupLeadingCandidates = true);

      // Get the four vector of the signal objects added so far
      const reco::Candidate::LorentzVector& p4() const { return p4_; }

    private:
      typedef std::pair<Region, ParticleType> CollectionKey;
      typedef std::map<CollectionKey, std::vector<CandidatePtr>*> CollectionMap;
      typedef std::shared_ptr<std::vector<CandidatePtr> > SortedListPtr;
      typedef std::map<CollectionKey, SortedListPtr> SortedCollectionMap;

      bool copyGammas_;

      const StringObjectFunction<reco::PFTau>* signalConeSize_;
      double minAbsPhotonSumPt_insideSignalCone_;
      double minRelPhotonSumPt_insideSignalCone_;
      double minAbsPhotonSumPt_outsideSignalCone_;
      double minRelPhotonSumPt_outsideSignalCone_;

      // Retrieve collection associated to signal/iso and type
      std::vector<CandidatePtr>* getCollection(Region region, ParticleType type);
      SortedListPtr getSortedCollection(Region region, ParticleType type);

      // Sort all our collections by PT and copy them into the tau
      void sortAndCopyIntoTau();

      // Helper functions for dealing with refs
      CandidatePtr convertToPtr(const PFCandidatePtr& pfPtr) const;
      CandidatePtr convertToPtr(const CandidatePtr& candPtr) const;

      const edm::Handle<edm::View<reco::Candidate> >& pfCands_;
      std::unique_ptr<reco::PFTau> tau_;
      CollectionMap collections_;

      // Keep sorted (by descending pt) collections
      SortedCollectionMap sortedCollections_;

      // Keep track of the signal cone four vector in case we want it
      reco::Candidate::LorentzVector p4_;
    };
  }  // namespace tau
}  // namespace reco
#endif
