#include "RecoTauTag/RecoTau/interface/RecoTauConstructor.h"

#include "RecoTauTag/RecoTau/interface/RecoTauCommonUtilities.h"
#include "DataFormats/Common/interface/RefToPtr.h"

#include "RecoTauTag/RecoTau/interface/pfRecoTauChargedHadronAuxFunctions.h"

#include <boost/foreach.hpp>
#include <boost/bind.hpp>

namespace reco { namespace tau {

RecoTauConstructor::RecoTauConstructor(const PFJetRef& jet, const edm::Handle<PFCandidateCollection>& pfCands, bool copyGammasFromPiZeros)
  : pfCands_(pfCands) 
{
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
  BOOST_FOREACH(const CollectionMap::value_type& colkey, collections_) {
    // Build an empty list for each collection
    sortedCollections_[colkey.first] = SortedListPtr(
        new SortedListPtr::element_type);
  }

  tau_->setjetRef(jet);
}

void RecoTauConstructor::addPFCand(Region region, ParticleType type, const PFCandidateRef& ref, bool skipAddToP4) {
  LogDebug("TauConstructorAddPFCand") << " region = " << region << ", type = " << type << ": Pt = " << ref->pt() << ", eta = " << ref->eta() << ", phi = " << ref->phi();
  if ( region == kSignal ) {
    // Keep track of the four vector of the signal vector products added so far.
    // If a photon add it if we are not using PiZeros to build the gammas
    if ( ((type != kGamma) || !copyGammas_) && !skipAddToP4 ) {
      LogDebug("TauConstructorAddPFCand") << "--> adding PFCand to tauP4." ;
      p4_ += ref->p4();
    }
  }
  getSortedCollection(region, type)->push_back(edm::refToPtr<PFCandidateCollection>(ref));
  // Add to global collection
  getSortedCollection(region, kAll)->push_back(edm::refToPtr<PFCandidateCollection>(ref));
}

void RecoTauConstructor::addPFCand(Region region, ParticleType type, const PFCandidatePtr& ptr, bool skipAddToP4) {
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

void RecoTauConstructor::reserve(Region region, ParticleType type, size_t size){
  getSortedCollection(region, type)->reserve(size);
  getCollection(region, type)->reserve(size);
  // Reserve global collection as well
  getSortedCollection(region, kAll)->reserve(
      getSortedCollection(region, kAll)->size() + size);
  getCollection(region, kAll)->reserve(
      getCollection(region, kAll)->size() + size);
}

void RecoTauConstructor::reserveTauChargedHadron(Region region, size_t size) 
{
  if ( region == kSignal ) {
    tau_->signalTauChargedHadronCandidates_.reserve(size);
    tau_->selectedSignalPFChargedHadrCands_.reserve(size);
  } else {
    tau_->isolationTauChargedHadronCandidates_.reserve(size);
    tau_->selectedIsolationPFChargedHadrCands_.reserve(size);
  }
}

namespace
{
  void checkOverlap(const PFCandidatePtr& neutral, const std::vector<PFCandidatePtr>& pfGammas, bool& isUnique)
  {
    LogDebug("TauConstructorCheckOverlap") << " pfGammas: #entries = " << pfGammas.size();
    for ( std::vector<PFCandidatePtr>::const_iterator pfGamma = pfGammas.begin();
	  pfGamma != pfGammas.end(); ++pfGamma ) {
      LogDebug("TauConstructorCheckOverlap") << "pfGamma = " << pfGamma->id() << ":" << pfGamma->key();
      if ( (*pfGamma) == neutral ) isUnique = false;
    }
  }

  void checkOverlap(const PFCandidatePtr& neutral, const std::vector<reco::RecoTauPiZero>& piZeros, bool& isUnique)
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

void RecoTauConstructor::addTauChargedHadron(Region region, const PFRecoTauChargedHadron& chargedHadron) 
{
  LogDebug("TauConstructorAddChH") << " region = " << region << ": Pt = " << chargedHadron.pt() << ", eta = " << chargedHadron.eta() << ", phi = " << chargedHadron.phi();
  // CV: need to make sure that PFGammas merged with ChargedHadrons are not part of PiZeros
  const std::vector<PFCandidatePtr>& neutrals = chargedHadron.getNeutralPFCandidates();
  std::vector<PFCandidatePtr> neutrals_cleaned;
  for ( std::vector<PFCandidatePtr>::const_iterator neutral = neutrals.begin();
	neutral != neutrals.end(); ++neutral ) {
    LogDebug("TauConstructorAddChH") << "neutral = " << neutral->id() << ":" << neutral->key();
    bool isUnique = true;
    if ( copyGammas_ ) checkOverlap(*neutral, *getSortedCollection(kSignal, kGamma), isUnique);
    else checkOverlap(*neutral, tau_->signalPiZeroCandidates_, isUnique);
    if ( region == kIsolation ) {
      if ( copyGammas_ ) checkOverlap(*neutral, *getSortedCollection(kIsolation, kGamma), isUnique);
      else checkOverlap(*neutral, tau_->isolationPiZeroCandidates_, isUnique);      
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
    tau_->signalTauChargedHadronCandidates_.push_back(chargedHadron_cleaned);
    p4_ += chargedHadron_cleaned.p4();
    if ( chargedHadron_cleaned.getChargedPFCandidate().isNonnull() ) {
      addPFCand(kSignal, kChargedHadron, chargedHadron_cleaned.getChargedPFCandidate(), true);
    }
    const std::vector<PFCandidatePtr>& neutrals = chargedHadron_cleaned.getNeutralPFCandidates();
    for ( std::vector<PFCandidatePtr>::const_iterator neutral = neutrals.begin();
	  neutral != neutrals.end(); ++neutral ) {
      if      ( (*neutral)->particleId() == reco::PFCandidate::gamma ) addPFCand(kSignal, kGamma, *neutral, true);
      else if ( (*neutral)->particleId() == reco::PFCandidate::h0    ) addPFCand(kSignal, kNeutralHadron, *neutral, true);
    };
  } else {
    tau_->isolationTauChargedHadronCandidates_.push_back(chargedHadron_cleaned);
    if ( chargedHadron_cleaned.getChargedPFCandidate().isNonnull() ) {
      if      ( chargedHadron_cleaned.getChargedPFCandidate()->particleId() == reco::PFCandidate::h  ) addPFCand(kIsolation, kChargedHadron, chargedHadron_cleaned.getChargedPFCandidate());
      else if ( chargedHadron_cleaned.getChargedPFCandidate()->particleId() == reco::PFCandidate::h0 ) addPFCand(kIsolation, kNeutralHadron, chargedHadron_cleaned.getChargedPFCandidate());
    }
    const std::vector<PFCandidatePtr>& neutrals = chargedHadron_cleaned.getNeutralPFCandidates();
    for ( std::vector<PFCandidatePtr>::const_iterator neutral = neutrals.begin();
	  neutral != neutrals.end(); ++neutral ) {
      if      ( (*neutral)->particleId() == reco::PFCandidate::gamma ) addPFCand(kIsolation, kGamma, *neutral);
      else if ( (*neutral)->particleId() == reco::PFCandidate::h0    ) addPFCand(kIsolation, kNeutralHadron, *neutral);
    };
  }
}

void RecoTauConstructor::reservePiZero(Region region, size_t size) 
{
  if ( region == kSignal ) {
    tau_->signalPiZeroCandidates_.reserve(size);
    // If we are building the gammas with the pizeros, resize that
    // vector as well
    if ( copyGammas_ ) reserve(kSignal, kGamma, 2*size);
  } else {
    tau_->isolationPiZeroCandidates_.reserve(size);
    if ( copyGammas_ ) reserve(kIsolation, kGamma, 2*size);
  }
}

void RecoTauConstructor::addPiZero(Region region, const RecoTauPiZero& piZero) 
{
  LogDebug("TauConstructorAddPi0") << " region = " << region << ": Pt = " << piZero.pt() << ", eta = " << piZero.eta() << ", phi = " << piZero.phi();
  if ( region == kSignal ) {
    tau_->signalPiZeroCandidates_.push_back(piZero);
    // Copy the daughter gammas into the gamma collection if desired
    if ( copyGammas_ ) {
      // If we are using the pizeros to build the gammas, make sure we update
      // the four vector correctly.
      p4_ += piZero.p4();
      addPFCands(kSignal, kGamma, piZero.daughterPtrVector().begin(), 
          piZero.daughterPtrVector().end());
    }
  } else {
    tau_->isolationPiZeroCandidates_.push_back(piZero);
    if ( copyGammas_ ) {
      addPFCands(kIsolation, kGamma, piZero.daughterPtrVector().begin(),
          piZero.daughterPtrVector().end());
    }
  }
}

std::vector<PFCandidatePtr>*
RecoTauConstructor::getCollection(Region region, ParticleType type) {
    return collections_[std::make_pair(region, type)];
}

RecoTauConstructor::SortedListPtr
RecoTauConstructor::getSortedCollection(Region region, ParticleType type) {
  return sortedCollections_[std::make_pair(region, type)];
}

// Trivial converter needed for polymorphism
PFCandidatePtr RecoTauConstructor::convertToPtr(
    const PFCandidatePtr& pfPtr) const {
  return pfPtr;
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

PFCandidatePtr RecoTauConstructor::convertToPtr(
    const PFCandidateRef& pfRef) const {
  if(pfRef.isNonnull()) {
    checkMatchedProductIds(pfRef, pfCands_);
    return PFCandidatePtr(pfCands_, pfRef.key());
  } else return PFCandidatePtr();
}

// Convert from a CandidateRef to a Ptr
PFCandidatePtr RecoTauConstructor::convertToPtr(
    const CandidatePtr& candPtr) const {
  if(candPtr.isNonnull()) {
    checkMatchedProductIds(candPtr, pfCands_);
    return PFCandidatePtr(pfCands_, candPtr.key());
  } else return PFCandidatePtr();
}

namespace {
template<typename T> bool ptDescending(const T& a, const T& b) {
  return a.pt() > b.pt();
}
template<typename T> bool ptDescendingPtr(const T& a, const T& b) {
  return a->pt() > b->pt();
}
}

void RecoTauConstructor::sortAndCopyIntoTau() {
  // The charged hadrons and pizeros are a special case, as we can sort them in situ
  std::sort(tau_->signalTauChargedHadronCandidates_.begin(),
            tau_->signalTauChargedHadronCandidates_.end(),
            ptDescending<PFRecoTauChargedHadron>);
  std::sort(tau_->isolationTauChargedHadronCandidates_.begin(),
            tau_->isolationTauChargedHadronCandidates_.end(),
            ptDescending<PFRecoTauChargedHadron>);
  std::sort(tau_->signalPiZeroCandidates_.begin(),
            tau_->signalPiZeroCandidates_.end(),
            ptDescending<RecoTauPiZero>);
  std::sort(tau_->isolationPiZeroCandidates_.begin(),
            tau_->isolationPiZeroCandidates_.end(),
            ptDescending<RecoTauPiZero>);

  // Sort each of our sortable collections, and copy them into the final
  // tau RefVector.
  BOOST_FOREACH ( const CollectionMap::value_type& colkey, collections_ ) {
    SortedListPtr sortedCollection = sortedCollections_[colkey.first];
    std::sort(sortedCollection->begin(),
              sortedCollection->end(),
              ptDescendingPtr<PFCandidatePtr>);
    // Copy into the real tau collection
    for ( std::vector<PFCandidatePtr>::const_iterator particle = sortedCollection->begin();
	  particle != sortedCollection->end(); ++particle ) {
      colkey.second->push_back(*particle);
    }
  }
}

std::auto_ptr<reco::PFTau> RecoTauConstructor::get(bool setupLeadingObjects) 
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
  for ( std::vector<PFRecoTauChargedHadron>::const_iterator chargedHadron = tau_->signalTauChargedHadronCandidates_.begin();
	chargedHadron != tau_->signalTauChargedHadronCandidates_.end(); ++chargedHadron ) {
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
//  tau_->setP4(
//      sumPFCandP4(
//        getCollection(kSignal, kAll)->begin(),
//        getCollection(kSignal, kAll)->end()
//        )
//      );
  LogDebug("TauConstructorGet") << "Pt = " << tau_->pt() << ", eta = " << tau_->eta() << ", phi = " << tau_->phi() << ", mass = " << tau_->mass() ;

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
    typedef std::vector<PFCandidatePtr>::const_iterator Iter;
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
}} // end namespace reco::tau
