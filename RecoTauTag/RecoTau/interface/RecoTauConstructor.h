#ifndef RecoTauTag_RecoTau_RecoTauConstructor_h
#define RecoTauTag_RecoTau_RecoTauConstructor_h

/*
 * RecoTauConstructor
 *
 * A generalized builder of TauType objects.  Takes a variety of
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
#include "DataFormats/Common/interface/RefToPtr.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"
#include "DataFormats/PatCandidates/interface/PackedCandidate.h"
#include "DataFormats/TauReco/interface/PFTau.h"
#include "DataFormats/TauReco/interface/PFBaseTau.h"
#include "DataFormats/JetReco/interface/JetCollection.h"
#include "DataFormats/TauReco/interface/PFRecoTauChargedHadron.h"
#include "DataFormats/TauReco/interface/RecoTauPiZero.h"
#include "CommonTools/Utils/interface/StringObjectFunction.h"

#include "RecoTauTag/RecoTau/interface/pfRecoTauChargedHadronAuxFunctions.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "RecoTauTag/RecoTau/interface/RecoTauCommonUtilities.h"

#include <boost/foreach.hpp>
#include <boost/bind.hpp>
#include "boost/shared_ptr.hpp"
#include <vector>

namespace reco { namespace tau {

template<class TauType, class PFCollType, class PFType>
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
    RecoTauConstructor(const JetBaseRef& jetRef,
        const edm::Handle<std::vector<PFCollType> >& pfCands,
	bool copyGammasFromPiZeros = false,
	const StringObjectFunction<TauType>* signalConeSize = nullptr,
	double minAbsPhotonSumPt_insideSignalCone = 2.5, double minRelPhotonSumPt_insideSignalCone = 0.,
	double minAbsPhotonSumPt_outsideSignalCone = 1.e+9, double minRelPhotonSumPt_outsideSignalCone = 1.e+9);

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

    /// Append a edm::Ref<std::vector<PFType> >/Ptr to a given collection
    void addPFCand(Region region, ParticleType type, const edm::Ref<std::vector<PFType> >& ref, bool skipAddToP4 = false);
    void addPFCand(Region region, ParticleType type, const edm::Ptr<PFType>& ptr, bool skipAddToP4 = false);

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
    std::auto_ptr<TauType> get(bool setupLeadingCandidates=true);

    // Get the four vector of the signal objects added so far
    const reco::Candidate::LorentzVector& p4() const { return p4_; }

  private:
    typedef std::pair<Region, ParticleType> CollectionKey;
    typedef std::map<CollectionKey, std::vector<edm::Ptr<PFType>>*> CollectionMap;
    typedef boost::shared_ptr<std::vector<edm::Ptr<PFType>> > SortedListPtr;
    typedef std::map<CollectionKey, SortedListPtr> SortedCollectionMap;

    bool copyGammas_;

    const StringObjectFunction<TauType>* signalConeSize_;
    double minAbsPhotonSumPt_insideSignalCone_;
    double minRelPhotonSumPt_insideSignalCone_;
    double minAbsPhotonSumPt_outsideSignalCone_; 
    double minRelPhotonSumPt_outsideSignalCone_;

    // Retrieve collection associated to signal/iso and type
    std::vector<edm::Ptr<PFType>>* getCollection(Region region, ParticleType type);
    SortedListPtr getSortedCollection(Region region, ParticleType type);

    // Sort all our collections by PT and copy them into the tau
    void sortAndCopyIntoTau();

    // Allow template specialization (PFJet for PFTau, JetBaseRef for PFBaseTau)
    void setJetRef(const JetBaseRef& jet);

    // Helper functions for dealing with refs
    template<typename T>
    edm::Ptr<T> convertToPtr(const edm::Ptr<T>& pfPtr) const{
      return pfPtr;
    }
    edm::Ptr<PFType> convertToPtr(const CandidatePtr& candPtr) const;
    edm::Ptr<PFType> convertToPtr(const edm::Ref<std::vector<PFType> >& pfRef) const;

    const edm::Handle<std::vector<PFCollType> >& pfCands_;
    std::auto_ptr<TauType> tau_;
    CollectionMap collections_;

    // Keep sorted (by descending pt) collections
    SortedCollectionMap sortedCollections_;

    // Keep track of the signal cone four vector in case we want it
    reco::Candidate::LorentzVector p4_;
};

namespace {
  template<typename T, typename U>
  std::vector<edm::Ptr<T> > convertPtrVector(const std::vector<edm::Ptr<U> >& cands) {
    std::vector<edm::Ptr<T> > newSignalCands;
    for (auto& cand : cands) {
      const auto& newPtr = cand->masterClone().template castTo<edm::Ptr<U> >();
      newSignalCands.push_back(newPtr);
    }
    return std::move(newSignalCands);
  }
  
  template<typename T>
  std::vector<edm::Ptr<T> > convertPtrVector(const std::vector<edm::Ptr<T> >& cands) {
    return cands;
  }
}

template<class TauType, class PFCollType, class PFType>
RecoTauConstructor<TauType, PFCollType, PFType>::RecoTauConstructor(const JetBaseRef& jet, const edm::Handle<std::vector<PFCollType> >& pfCands, 
               bool copyGammasFromPiZeros,
               const StringObjectFunction<TauType>* signalConeSize,
               double minAbsPhotonSumPt_insideSignalCone, double minRelPhotonSumPt_insideSignalCone,
               double minAbsPhotonSumPt_outsideSignalCone, double minRelPhotonSumPt_outsideSignalCone)
  : signalConeSize_(signalConeSize),
    minAbsPhotonSumPt_insideSignalCone_(minAbsPhotonSumPt_insideSignalCone),
    minRelPhotonSumPt_insideSignalCone_(minRelPhotonSumPt_insideSignalCone),
    minAbsPhotonSumPt_outsideSignalCone_(minAbsPhotonSumPt_outsideSignalCone),
    minRelPhotonSumPt_outsideSignalCone_(minRelPhotonSumPt_outsideSignalCone),
    pfCands_(pfCands)
{
  // Initialize tau
  tau_.reset(new TauType());

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
  BOOST_FOREACH(const typename CollectionMap::value_type& colkey, collections_) {
    // Build an empty list for each collection
    sortedCollections_[colkey.first] = SortedListPtr(
        new typename SortedListPtr::element_type);
  }

  setJetRef(jet);
}

template<class TauType, class PFCollType, class PFType>
void RecoTauConstructor<TauType, PFCollType, PFType>::addPFCand(Region region, ParticleType type, const edm::Ref<std::vector<PFType> >& ref, bool skipAddToP4) {
  LogDebug("TauConstructorAddPFCand") << " region = " << region << ", type = " << type << ": Pt = " << ref->pt() << ", eta = " << ref->eta() << ", phi = " << ref->phi();
  if ( region == kSignal ) {
    // Keep track of the four vector of the signal vector products added so far.
    // If a photon add it if we are not using PiZeros to build the gammas
    if ( ((type != kGamma) || !copyGammas_) && !skipAddToP4 ) {
      LogDebug("TauConstructorAddPFCand") << "--> adding PFCand to tauP4." ;
      p4_ += ref->p4();
    }
  }
  getSortedCollection(region, type)->push_back(edm::refToPtr<std::vector<PFType> >(ref));
  // Add to global collection
  getSortedCollection(region, kAll)->push_back(edm::refToPtr<std::vector<PFType> >(ref));
}

template<class TauType, class PFCollType, class PFType>
void RecoTauConstructor<TauType, PFCollType, PFType>::addPFCand(Region region, ParticleType type, const edm::Ptr<PFType>& ptr, bool skipAddToP4) {
  LogDebug("TauConstructorAddPFCand") << " region = " << region << ", type = " << type << ": Pt = " << ptr->pt() << ", eta = " << ptr->eta() << ", phi = " << ptr->phi();
  if ( region == kSignal ) {
    // Keep track of the four vector of the signal vector products added so far.
    // If a photon add it if we are not using PiZeros to build the gammas
    if ( ((type != kGamma) || !copyGammas_) && !skipAddToP4 ) {
      LogDebug("TauConstructorAddPFCand") << "--> adding PFCand to tauP4." ;
      p4_ += ptr->p4();
    }
  }
  getSortedCollection(region, type)->push_back(ptr);
  // Add to global collection
  getSortedCollection(region, kAll)->push_back(ptr);
}

template<class TauType, class PFCollType, class PFType>
void RecoTauConstructor<TauType, PFCollType, PFType>::reserve(Region region, ParticleType type, size_t size){
  getSortedCollection(region, type)->reserve(size);
  getCollection(region, type)->reserve(size);
  // Reserve global collection as well
  getSortedCollection(region, kAll)->reserve(
      getSortedCollection(region, kAll)->size() + size);
  getCollection(region, kAll)->reserve(
      getCollection(region, kAll)->size() + size);
}

template<class TauType, class PFCollType, class PFType>
void RecoTauConstructor<TauType, PFCollType, PFType>::reserveTauChargedHadron(Region region, size_t size) 
{
  if ( region == kSignal ) {
    tau_->signalTauChargedHadronCandidatesRestricted().reserve(size);
    tau_->selectedSignalPFChargedHadrCands_.reserve(size);
  } else {
    tau_->isolationTauChargedHadronCandidatesRestricted().reserve(size);
    tau_->selectedIsolationPFChargedHadrCands_.reserve(size);
  }
}

namespace
{
  template<typename PFType>
  void checkOverlap(const CandidatePtr& neutral, const std::vector<edm::Ptr<PFType>>& pfGammas, bool& isUnique)
  {
    LogDebug("TauConstructorCheckOverlap") << " pfGammas: #entries = " << pfGammas.size();
    for ( typename std::vector<edm::Ptr<PFType>>::const_iterator pfGamma = pfGammas.begin();
    pfGamma != pfGammas.end(); ++pfGamma ) {
      LogDebug("TauConstructorCheckOverlap") << "pfGamma = " << pfGamma->id() << ":" << pfGamma->key();
      if ( (*pfGamma).refCore() == neutral.refCore() && (*pfGamma).key() == neutral.key() ) isUnique = false;
    }
  }


  void checkOverlapPiZero(const CandidatePtr& neutral, const std::vector<reco::RecoTauPiZero>& piZeros, bool& isUnique)
  {
    LogDebug("TauConstructorCheckOverlap") << " piZeros: #entries = " << piZeros.size();
    for ( std::vector<reco::RecoTauPiZero>::const_iterator piZero = piZeros.begin();
    piZero != piZeros.end(); ++piZero ) {
      size_t numPFGammas = piZero->numberOfDaughters();
      for ( size_t iPFGamma = 0; iPFGamma < numPFGammas; ++iPFGamma ) {
  reco::CandidatePtr pfGamma = piZero->daughterPtr(iPFGamma);
  LogDebug("TauConstructorCheckOverlap") << "pfGamma = " << pfGamma.id() << ":" << pfGamma.key();
  if ( pfGamma.id() == neutral.id() && pfGamma.key() == neutral.key() ) isUnique = false;
      }
    }
  }
}

template<class TauType, class PFCollType, class PFType>
void RecoTauConstructor<TauType, PFCollType, PFType>::addTauChargedHadron(Region region, const PFRecoTauChargedHadron& chargedHadron) 
{
  LogDebug("TauConstructorAddChH") << " region = " << region << ": Pt = " << chargedHadron.pt() << ", eta = " << chargedHadron.eta() << ", phi = " << chargedHadron.phi();
  // CV: need to make sure that PFGammas merged with ChargedHadrons are not part of PiZeros
  const std::vector<CandidatePtr>& neutrals = chargedHadron.getNeutralPFCandidates();
  std::vector<CandidatePtr> neutrals_cleaned;
  for ( std::vector<CandidatePtr>::const_iterator neutral = neutrals.begin();
  neutral != neutrals.end(); ++neutral ) {
    LogDebug("TauConstructorAddChH") << "neutral = " << neutral->id() << ":" << neutral->key();
    bool isUnique = true;
    if ( copyGammas_ ) checkOverlap(*neutral, *getSortedCollection(kSignal, kGamma), isUnique);
    else checkOverlapPiZero(*neutral, tau_->signalPiZeroCandidatesRestricted(), isUnique);
    if ( region == kIsolation ) {
      if ( copyGammas_ ) checkOverlap(*neutral, *getSortedCollection(kIsolation, kGamma), isUnique);
      else checkOverlapPiZero(*neutral, tau_->isolationPiZeroCandidatesRestricted(), isUnique);      
    }
    LogDebug("TauConstructorAddChH") << "--> isUnique = " << isUnique;
    if ( isUnique ) neutrals_cleaned.push_back(*neutral);
  }
  PFRecoTauChargedHadron chargedHadron_cleaned = chargedHadron;
  if ( neutrals_cleaned.size() != neutrals.size() ) {
    chargedHadron_cleaned.neutralPFCandidates_ = neutrals_cleaned;
    setChargedHadronP4(chargedHadron_cleaned);
  }
  if ( region == kSignal ) {
    tau_->signalTauChargedHadronCandidatesRestricted().push_back(chargedHadron_cleaned);
    p4_ += chargedHadron_cleaned.p4();
    if ( chargedHadron_cleaned.getChargedPFCandidate().isNonnull() ) {
      addPFCand(kSignal, kChargedHadron, convertToPtr(chargedHadron_cleaned.getChargedPFCandidate()), true);
    }
    const std::vector<CandidatePtr>& neutrals = chargedHadron_cleaned.getNeutralPFCandidates();
    for ( std::vector<CandidatePtr>::const_iterator neutral = neutrals.begin();
    neutral != neutrals.end(); ++neutral ) {
      if      ( std::abs((*neutral)->pdgId()) == 22 ) addPFCand(kSignal, kGamma, convertToPtr(*neutral), true);
      else if ( std::abs((*neutral)->pdgId()) == 130 ) addPFCand(kSignal, kNeutralHadron, convertToPtr(*neutral), true);
    };
  } else {
    tau_->isolationTauChargedHadronCandidatesRestricted().push_back(chargedHadron_cleaned);
    if ( chargedHadron_cleaned.getChargedPFCandidate().isNonnull() ) {
      if      ( std::abs(chargedHadron_cleaned.getChargedPFCandidate()->pdgId()) == 211  ) addPFCand(kIsolation, kChargedHadron, convertToPtr(chargedHadron_cleaned.getChargedPFCandidate()));
      else if ( std::abs(chargedHadron_cleaned.getChargedPFCandidate()->pdgId()) == 130 ) addPFCand(kIsolation, kNeutralHadron, convertToPtr(chargedHadron_cleaned.getChargedPFCandidate()));
    }
    const std::vector<CandidatePtr>& neutrals = chargedHadron_cleaned.getNeutralPFCandidates();
    for ( std::vector<CandidatePtr>::const_iterator neutral = neutrals.begin();
    neutral != neutrals.end(); ++neutral ) {
      if      ( std::abs((*neutral)->pdgId()) == 22 ) addPFCand(kIsolation, kGamma, convertToPtr(*neutral));
      else if ( std::abs((*neutral)->pdgId()) == 130 ) addPFCand(kIsolation, kNeutralHadron, convertToPtr(*neutral));
    };
  }
}

template<class TauType, class PFCollType, class PFType>
void RecoTauConstructor<TauType, PFCollType, PFType>::reservePiZero(Region region, size_t size) 
{
  if ( region == kSignal ) {
    tau_->signalPiZeroCandidatesRestricted().reserve(size);
    // If we are building the gammas with the pizeros, resize that
    // vector as well
    if ( copyGammas_ ) reserve(kSignal, kGamma, 2*size);
  } else {
    tau_->isolationPiZeroCandidatesRestricted().reserve(size);
    if ( copyGammas_ ) reserve(kIsolation, kGamma, 2*size);
  }
}

template<class TauType, class PFCollType, class PFType>
void RecoTauConstructor<TauType, PFCollType, PFType>::addPiZero(Region region, const RecoTauPiZero& piZero) 
{
  LogDebug("TauConstructorAddPi0") << " region = " << region << ": Pt = " << piZero.pt() << ", eta = " << piZero.eta() << ", phi = " << piZero.phi();
  if ( region == kSignal ) {
    tau_->signalPiZeroCandidatesRestricted().push_back(piZero);
    // Copy the daughter gammas into the gamma collection if desired
    if ( copyGammas_ ) {
      // If we are using the pizeros to build the gammas, make sure we update
      // the four vector correctly.
      p4_ += piZero.p4();
      addPFCands(kSignal, kGamma, piZero.daughterPtrVector().begin(), 
          piZero.daughterPtrVector().end());
    }
  } else {
    tau_->isolationPiZeroCandidatesRestricted().push_back(piZero);
    if ( copyGammas_ ) {
      addPFCands(kIsolation, kGamma, piZero.daughterPtrVector().begin(),
          piZero.daughterPtrVector().end());
    }
  }
}

template<class TauType, class PFCollType, class PFType>
std::vector<edm::Ptr<PFType>>*
RecoTauConstructor<TauType, PFCollType, PFType>::getCollection(Region region, ParticleType type) {
    return collections_[std::make_pair(region, type)];
}

template<class TauType, class PFCollType, class PFType>
typename RecoTauConstructor<TauType, PFCollType, PFType>::SortedListPtr
RecoTauConstructor<TauType, PFCollType, PFType>::getSortedCollection(Region region, ParticleType type) {
  return sortedCollections_[std::make_pair(region, type)];
}

// // Trivial converter needed for polymorphism
// template<class TauType, class PFCollType, class PFType>
// edm::Ptr<PFType> RecoTauConstructor<TauType, PFCollType, PFType>::convertToPtr(
//     const edm::Ptr<PFType>& pfPtr) const 

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

// Convert from a Ref to a Ptr
template<class TauType, class PFCollType, class PFType>
edm::Ptr<PFType> RecoTauConstructor<TauType, PFCollType, PFType>::convertToPtr(
    const edm::Ref<std::vector<PFType> >& pfRef) const {
  if(pfRef.isNonnull()) {
    checkMatchedProductIds(pfRef, pfCands_);
    return edm::Ptr<PFType>(pfCands_, pfRef.key());
  } else return edm::Ptr<PFType>();
}


template<class TauType, class PFCollType, class PFType>
edm::Ptr<PFType> RecoTauConstructor<TauType, PFCollType, PFType>::convertToPtr(
    const CandidatePtr& candPtr) const {
  if(candPtr.isNonnull()) {
    checkMatchedProductIds(candPtr, pfCands_);
    return edm::Ptr<PFType>(pfCands_, candPtr.key());
  } else return edm::Ptr<PFType>();
}

namespace {
template<typename T> bool ptDescending(const T& a, const T& b) {
  return a.pt() > b.pt();
}
template<typename T> bool ptDescendingPtr(const T& a, const T& b) {
  return a->pt() > b->pt();
}
}

template<class TauType, class PFCollType, class PFType>
void RecoTauConstructor<TauType, PFCollType, PFType>::sortAndCopyIntoTau() {
  // The charged hadrons and pizeros are a special case, as we can sort them in situ
  std::sort(tau_->signalTauChargedHadronCandidatesRestricted().begin(),
            tau_->signalTauChargedHadronCandidatesRestricted().end(),
            ptDescending<PFRecoTauChargedHadron>);
  std::sort(tau_->isolationTauChargedHadronCandidatesRestricted().begin(),
            tau_->isolationTauChargedHadronCandidatesRestricted().end(),
            ptDescending<PFRecoTauChargedHadron>);
  std::sort(tau_->signalPiZeroCandidatesRestricted().begin(),
            tau_->signalPiZeroCandidatesRestricted().end(),
            ptDescending<RecoTauPiZero>);
  std::sort(tau_->isolationPiZeroCandidatesRestricted().begin(),
            tau_->isolationPiZeroCandidatesRestricted().end(),
            ptDescending<RecoTauPiZero>);

  // Sort each of our sortable collections, and copy them into the final
  // tau RefVector.
  BOOST_FOREACH ( const typename CollectionMap::value_type& colkey, collections_ ) {
    SortedListPtr sortedCollection = sortedCollections_[colkey.first];
    std::sort(sortedCollection->begin(),
              sortedCollection->end(),
              ptDescendingPtr<edm::Ptr<PFType>>);
    // Copy into the real tau collection
    for ( typename std::vector<edm::Ptr<PFType>>::const_iterator particle = sortedCollection->begin();
    particle != sortedCollection->end(); ++particle ) {
      colkey.second->push_back(*particle);
    }
  }
}

namespace
{
  template<typename TauType>
  typename TauType::hadronicDecayMode calculateDecayMode(const TauType& tau, double dRsignalCone, 
                double minAbsPhotonSumPt_insideSignalCone, double minRelPhotonSumPt_insideSignalCone,
                double minAbsPhotonSumPt_outsideSignalCone, double minRelPhotonSumPt_outsideSignalCone) 
  {
    unsigned int nCharged = tau.signalTauChargedHadronCandidates().size();
    // If no tracks exist, this is definitely not a tau!
    if ( !nCharged ) return TauType::kNull;

    unsigned int nPiZeros = 0;
    const std::vector<RecoTauPiZero>& piZeros = tau.signalPiZeroCandidates();
    for ( std::vector<RecoTauPiZero>::const_iterator piZero = piZeros.begin();
    piZero != piZeros.end(); ++piZero ) {
      double photonSumPt_insideSignalCone = 0.;
      double photonSumPt_outsideSignalCone = 0.;
      int numPhotons = piZero->numberOfDaughters();
      for ( int idxPhoton = 0; idxPhoton < numPhotons; ++idxPhoton ) {
  const reco::Candidate* photon = piZero->daughter(idxPhoton);
  double dR = deltaR(photon->p4(), tau.p4());
  if ( dR < dRsignalCone ) {
    photonSumPt_insideSignalCone += photon->pt();
  } else {
    photonSumPt_outsideSignalCone += photon->pt();
  }
      }
      if ( photonSumPt_insideSignalCone  > minAbsPhotonSumPt_insideSignalCone  || photonSumPt_insideSignalCone  > (minRelPhotonSumPt_insideSignalCone*tau.pt())  ||
     photonSumPt_outsideSignalCone > minAbsPhotonSumPt_outsideSignalCone || photonSumPt_outsideSignalCone > (minRelPhotonSumPt_outsideSignalCone*tau.pt()) ) ++nPiZeros;
    }
    
    // Find the maximum number of PiZeros our parameterization can hold
    const unsigned int maxPiZeros = TauType::kOneProngNPiZero;
    
    // Determine our track index
    unsigned int trackIndex = (nCharged - 1)*(maxPiZeros + 1);
    
    // Check if we handle the given number of tracks
    if ( trackIndex >= TauType::kRareDecayMode ) return TauType::kRareDecayMode;
    
    if ( nPiZeros > maxPiZeros ) nPiZeros = maxPiZeros;
    return static_cast<typename TauType::hadronicDecayMode>(trackIndex + nPiZeros);
  }
}

template<class TauType, class PFCollType, class PFType>
std::auto_ptr<TauType> RecoTauConstructor<TauType, PFCollType, PFType>::get(bool setupLeadingObjects) 
{
  LogDebug("TauConstructorGet") << "Start getting" ;

  // Copy the sorted collections into the interal tau refvectors
  sortAndCopyIntoTau();

  // Setup all the important member variables of the tau
  // Set charge of tau
//  tau_->setCharge(
//      sumPFCandCharge(getCollection(kSignal, kChargedHadron)->begin(),
//                      getCollection(kSignal, kChargedHadron)->end()));
  // CV: take charge of highest pT charged hadron as charge of tau,
  //     in case tau does not have three reconstructed tracks
  //    (either because tau is reconstructed as 2prong or because PFRecoTauChargedHadron is built from a PFNeutralHadron)
  unsigned int nCharged = 0;
  int charge = 0;
  double leadChargedHadronPt = 0.;
  int leadChargedHadronCharge = 0;
  for ( std::vector<PFRecoTauChargedHadron>::const_iterator chargedHadron = tau_->signalTauChargedHadronCandidatesRestricted().begin();
  chargedHadron != tau_->signalTauChargedHadronCandidatesRestricted().end(); ++chargedHadron ) {
    if ( chargedHadron->algoIs(PFRecoTauChargedHadron::kChargedPFCandidate) || chargedHadron->algoIs(PFRecoTauChargedHadron::kTrack) ) {
      ++nCharged;
      charge += chargedHadron->charge();
      if ( chargedHadron->pt() > leadChargedHadronPt ) {  
  leadChargedHadronPt = chargedHadron->pt();
  leadChargedHadronCharge = chargedHadron->charge();
      }
    }
  }
  if ( nCharged == 3 ) tau_->setCharge(charge);
  else tau_->setCharge(leadChargedHadronCharge);

  // Set PDG id
  tau_->setPdgId(tau_->charge() < 0 ? 15 : -15);

  // Set P4
  tau_->setP4(p4_);

  // Set Decay Mode
  double dRsignalCone = ( signalConeSize_ ) ? (*signalConeSize_)(*tau_) : 0.5;
  tau_->setSignalConeSize(dRsignalCone);
  typename TauType::hadronicDecayMode dm = calculateDecayMode(
      *tau_, 
      dRsignalCone, 
      minAbsPhotonSumPt_insideSignalCone_, minRelPhotonSumPt_insideSignalCone_, minAbsPhotonSumPt_outsideSignalCone_, minRelPhotonSumPt_outsideSignalCone_);
  tau_->setDecayMode(dm);

  LogDebug("TauConstructorGet") << "Pt = " << tau_->pt() << ", eta = " << tau_->eta() << ", phi = " << tau_->phi() << ", mass = " << tau_->mass() << ", dm = " << tau_->decayMode() ;

  // Set charged isolation quantities
  tau_->setisolationPFChargedHadrCandsPtSum(
      sumPFCandPt(
        getCollection(kIsolation, kChargedHadron)->begin(),
        getCollection(kIsolation, kChargedHadron)->end()));

  // Set gamma isolation quantities
  tau_->setisolationPFGammaCandsEtSum(
      sumPFCandPt(
        getCollection(kIsolation, kGamma)->begin(),
        getCollection(kIsolation, kGamma)->end()));

  // Set em fraction
  tau_->setemFraction(sumPFCandPt(
          getCollection(kSignal, kGamma)->begin(),
          getCollection(kSignal, kGamma)->end()) / tau_->pt());

  if ( setupLeadingObjects ) {
    typedef typename std::vector<edm::Ptr<PFType>>::const_iterator Iter;
    // Find the highest PT object in the signal cone
    Iter leadingCand = leadPFCand(
        getCollection(kSignal, kAll)->begin(),
        getCollection(kSignal, kAll)->end());

    if ( leadingCand != getCollection(kSignal, kAll)->end() )
      tau_->setleadPFCand(*leadingCand);

    // Hardest charged object in signal cone
    Iter leadingChargedCand = leadPFCand(
        getCollection(kSignal, kChargedHadron)->begin(),
        getCollection(kSignal, kChargedHadron)->end());

    if ( leadingChargedCand != getCollection(kSignal, kChargedHadron)->end() )
      tau_->setleadPFChargedHadrCand(*leadingChargedCand);

    // Hardest gamma object in signal cone
    Iter leadingGammaCand = leadPFCand(
        getCollection(kSignal, kGamma)->begin(),
        getCollection(kSignal, kGamma)->end());

    if(leadingGammaCand != getCollection(kSignal, kGamma)->end())
      tau_->setleadPFNeutralCand(*leadingGammaCand);
  }
  return tau_;
}

// Specializations
template<>
void RecoTauConstructor<reco::PFTau, reco::PFCandidate, reco::PFCandidate>::setJetRef(const JetBaseRef& jet);

template<>
void RecoTauConstructor<reco::PFBaseTau, pat::PackedCandidate, reco::Candidate>::setJetRef(const JetBaseRef& jet);


} } // end reco::tau namespace
#endif
