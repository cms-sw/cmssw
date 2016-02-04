#include "RecoTauTag/RecoTau/interface/RecoTauConstructor.h"
#include "RecoTauTag/RecoTau/interface/RecoTauCommonUtilities.h"
#include <boost/foreach.hpp>
#include <boost/bind.hpp>

namespace reco { namespace tau {



RecoTauConstructor::RecoTauConstructor(const PFJetRef& jet,
    const edm::Handle<PFCandidateCollection>& pfCands,
    bool copyGammasFromPiZeros):pfCands_(pfCands) {

  // Initialize tau
  tau_.reset(new PFTau());

  copyGammas_ = copyGammasFromPiZeros;
  // Initialize our Accessors
  collections_[std::make_pair(kSignal, kChargedHadron)] =
      &tau_->selectedSignalPFChargedHadrCands_;
  collections_[std::make_pair(kSignal, kGamma)] =
      &tau_->selectedSignalPFGammaCands_;
  collections_[std::make_pair(kSignal, kNeutralHadron)] =
      &tau_->selectedSignalPFNeutrHadrCands_;
  collections_[std::make_pair(kSignal, kAll)] =
      &tau_->selectedSignalPFCands_;

  collections_[std::make_pair(kIsolation, kChargedHadron)] =
      &tau_->selectedIsolationPFChargedHadrCands_;
  collections_[std::make_pair(kIsolation, kGamma)] =
      &tau_->selectedIsolationPFGammaCands_;
  collections_[std::make_pair(kIsolation, kNeutralHadron)] =
      &tau_->selectedIsolationPFNeutrHadrCands_;
  collections_[std::make_pair(kIsolation, kAll)] =
      &tau_->selectedIsolationPFCands_;

  // Build our temporary sorted collections, since you can't use stl sorts on
  // RefVectors
  BOOST_FOREACH(const CollectionMap::value_type &colkey, collections_) {
    // Build an empty list for each collection
    sortedCollections_[colkey.first] = SortedListPtr(
        new SortedListPtr::element_type);
  }

  tau_->setjetRef(jet);
}

void RecoTauConstructor::addPFCand(Region region, ParticleType type,
    const PFCandidateRef& ref) {
  if (region == kSignal) {
    // Keep track of the four vector of the signal vector products added so far.
    // If a photon add it if we are not using PiZeros to build the gammas
    if ( (type != kGamma) || !copyGammas_ )
      p4_ += ref->p4();
  }
  getSortedCollection(region, type)->push_back(ref);
  // Add to global collection
  getSortedCollection(region, kAll)->push_back(ref);
}

void RecoTauConstructor::reserve(Region region, ParticleType type, size_t size){
  getSortedCollection(region, type)->reserve(size);
  getCollection(region, type)->reserve(size);
  // Reserve global collection as well
  getSortedCollection(region, kAll)->reserve(
      getSortedCollection(region, kAll)->size() + size);
  getCollection(region, kAll)->reserve(
      getCollection(region, kAll)->size() + size);
}

void RecoTauConstructor::reservePiZero(Region region, size_t size) {
  if(region == kSignal) {
    tau_->signalPiZeroCandidates_.reserve(size);
    // If we are building the gammas with the pizeros, resize that
    // vector as well
    if(copyGammas_)
      reserve(kSignal, kGamma, 2*size);
  } else {
    tau_->isolationPiZeroCandidates_.reserve(size);
    if(copyGammas_)
      reserve(kIsolation, kGamma, 2*size);
  }
}

void RecoTauConstructor::addPiZero(Region region, const RecoTauPiZero& piZero) {
  if(region == kSignal) {
    tau_->signalPiZeroCandidates_.push_back(piZero);
    // Copy the daughter gammas into the gamma collection if desired
    if(copyGammas_) {
      // If we are using the pizeros to build the gammas, make sure we update
      // the four vector correctly.
      p4_ += piZero.p4();
      addPFCands(kSignal, kGamma, piZero.daughterPtrVector().begin(),
          piZero.daughterPtrVector().end());
    }
  } else {
    tau_->isolationPiZeroCandidates_.push_back(piZero);
    if(copyGammas_) {
      addPFCands(kIsolation, kGamma, piZero.daughterPtrVector().begin(),
          piZero.daughterPtrVector().end());
    }
  }
}

PFCandidateRefVector*
RecoTauConstructor::getCollection(Region region, ParticleType type) {
    return collections_[std::make_pair(region, type)];
}

RecoTauConstructor::SortedListPtr
RecoTauConstructor::getSortedCollection(Region region, ParticleType type) {
  return sortedCollections_[std::make_pair(region, type)];
}

// Trivial converter needed for polymorphism
PFCandidateRef RecoTauConstructor::convertToRef(
    const PFCandidateRef& pfRef) const {
  return pfRef;
}

namespace {
// Make sure the two products come from the same EDM source
template<typename T1, typename T2>
void checkMatchedProductIds(const T1& t1, const T2& t2) {
    if (t1.id() != t2.id()) {
      throw cms::Exception("MismatchedPFCandSrc") << "Error: the input tag"
          << " for the PF candidate collection provided to the RecoTauBuilder "
          << " does not match the one that was used to build the source jets."
          << " Please update the pfCandSrc paramters for the PFTau builders.";
    }
}
}

// Convert from a Ptr to a Ref
PFCandidateRef RecoTauConstructor::convertToRef(
    const PFCandidatePtr& pfPtr) const {
  if(pfPtr.isNonnull()) {
    checkMatchedProductIds(pfPtr, pfCands_);
    return PFCandidateRef(pfCands_, pfPtr.key());
  } else return PFCandidateRef();
}

// Convert from a CandidatePtr to a Ref
PFCandidateRef RecoTauConstructor::convertToRef(
    const CandidatePtr& candPtr) const {
  if(candPtr.isNonnull()) {
    checkMatchedProductIds(candPtr, pfCands_);
    return PFCandidateRef(pfCands_, candPtr.key());
  } else return PFCandidateRef();
}

namespace {
template<typename T> bool ptDescending(const T& a, const T& b) {
  return a.pt() > b.pt();
}
template<typename T> bool ptDescendingRef(const T& a, const T& b) {
  return a->pt() > b->pt();
}
}

void RecoTauConstructor::sortAndCopyIntoTau() {
  // The pizeros are a special case, as we can sort them in situ
  std::sort(tau_->signalPiZeroCandidates_.begin(),
            tau_->signalPiZeroCandidates_.end(),
            ptDescending<RecoTauPiZero>);
  std::sort(tau_->isolationPiZeroCandidates_.begin(),
            tau_->isolationPiZeroCandidates_.end(),
            ptDescending<RecoTauPiZero>);

  // Sort each of our sortable collections, and copy them into the final
  // tau RefVector.
  BOOST_FOREACH(const CollectionMap::value_type &colkey, collections_) {
    SortedListPtr sortedCollection = sortedCollections_[colkey.first];
    std::sort(sortedCollection->begin(),
              sortedCollection->end(),
              ptDescendingRef<PFCandidateRef>);
    // Copy into the real tau collection
    std::for_each(
        sortedCollection->begin(), sortedCollection->end(),
        boost::bind(&PFCandidateRefVector::push_back, colkey.second, _1));
  }
}

std::auto_ptr<reco::PFTau> RecoTauConstructor::get(bool setupLeadingObjects) {
  // Copy the sorted collections into the interal tau refvectors
  sortAndCopyIntoTau();

  // Setup all the important member variables of the tau
  // Set charge of tau
  tau_->setCharge(
      sumPFCandCharge(getCollection(kSignal, kChargedHadron)->begin(),
                      getCollection(kSignal, kChargedHadron)->end()));

  // Set PDG id
  tau_->setPdgId(tau_->charge() > 0 ? 15 : -15);

  // Set P4
  tau_->setP4(p4_);
//  tau_->setP4(
//      sumPFCandP4(
//        getCollection(kSignal, kAll)->begin(),
//        getCollection(kSignal, kAll)->end()
//        )
//      );

  // Set charged isolation quantities
  tau_->setisolationPFChargedHadrCandsPtSum(
      sumPFCandPt(
        getCollection(kIsolation, kChargedHadron)->begin(),
        getCollection(kIsolation, kChargedHadron)->end()
        )
      );

  // Set gamma isolation quantities
  tau_->setisolationPFGammaCandsEtSum(
      sumPFCandPt(
        getCollection(kIsolation, kGamma)->begin(),
        getCollection(kIsolation, kGamma)->end()
        )
      );
  // Set em fraction
  tau_->setemFraction(sumPFCandPt(
          getCollection(kSignal, kGamma)->begin(),
          getCollection(kSignal, kGamma)->end()) / tau_->pt());

  if(setupLeadingObjects)
  {
    typedef PFCandidateRefVector::const_iterator Iter;
    // Find the highest PT object in the signal cone
    Iter leadingCand = leadPFCand(
        getCollection(kSignal, kAll)->begin(),
        getCollection(kSignal, kAll)->end()
        );

    if(leadingCand != getCollection(kSignal, kAll)->end())
      tau_->setleadPFCand(*leadingCand);

    // Hardest charged object in signal cone
    Iter leadingChargedCand = leadPFCand(
        getCollection(kSignal, kChargedHadron)->begin(),
        getCollection(kSignal, kChargedHadron)->end()
        );

    if(leadingChargedCand != getCollection(kSignal, kChargedHadron)->end())
      tau_->setleadPFChargedHadrCand(*leadingChargedCand);

    // Hardest gamma object in signal cone
    Iter leadingGammaCand = leadPFCand(
        getCollection(kSignal, kGamma)->begin(),
        getCollection(kSignal, kGamma)->end()
        );

    if(leadingGammaCand != getCollection(kSignal, kGamma)->end())
      tau_->setleadPFNeutralCand(*leadingGammaCand);
  }
  return tau_;
}
}} // end namespace reco::tau
