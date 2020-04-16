#include <memory>

#include "RecoTauTag/RecoTau/interface/RecoTauConstructor.h"

#include "RecoTauTag/RecoTau/interface/RecoTauCommonUtilities.h"
#include "DataFormats/Common/interface/RefToPtr.h"

#include "DataFormats/Math/interface/deltaR.h"

#include "RecoTauTag/RecoTau/interface/pfRecoTauChargedHadronAuxFunctions.h"

namespace reco::tau {

  RecoTauConstructor::RecoTauConstructor(const JetBaseRef& jet,
                                         const edm::Handle<edm::View<reco::Candidate> >& pfCands,
                                         bool copyGammasFromPiZeros,
                                         const StringObjectFunction<reco::PFTau>* signalConeSize,
                                         double minAbsPhotonSumPt_insideSignalCone,
                                         double minRelPhotonSumPt_insideSignalCone,
                                         double minAbsPhotonSumPt_outsideSignalCone,
                                         double minRelPhotonSumPt_outsideSignalCone)
      : signalConeSize_(signalConeSize),
        minAbsPhotonSumPt_insideSignalCone_(minAbsPhotonSumPt_insideSignalCone),
        minRelPhotonSumPt_insideSignalCone_(minRelPhotonSumPt_insideSignalCone),
        minAbsPhotonSumPt_outsideSignalCone_(minAbsPhotonSumPt_outsideSignalCone),
        minRelPhotonSumPt_outsideSignalCone_(minRelPhotonSumPt_outsideSignalCone),
        pfCands_(pfCands) {
    // Initialize tau
    tau_.reset(new PFTau());

    copyGammas_ = copyGammasFromPiZeros;
    // Initialize our Accessors
    collections_[std::make_pair(kSignal, kChargedHadron)] = &tau_->selectedSignalChargedHadrCands_;
    collections_[std::make_pair(kSignal, kGamma)] = &tau_->selectedSignalGammaCands_;
    collections_[std::make_pair(kSignal, kNeutralHadron)] = &tau_->selectedSignalNeutrHadrCands_;
    collections_[std::make_pair(kSignal, kAll)] = &tau_->selectedSignalCands_;

    collections_[std::make_pair(kIsolation, kChargedHadron)] = &tau_->selectedIsolationChargedHadrCands_;
    collections_[std::make_pair(kIsolation, kGamma)] = &tau_->selectedIsolationGammaCands_;
    collections_[std::make_pair(kIsolation, kNeutralHadron)] = &tau_->selectedIsolationNeutrHadrCands_;
    collections_[std::make_pair(kIsolation, kAll)] = &tau_->selectedIsolationCands_;

    // Build our temporary sorted collections, since you can't use stl sorts on
    // RefVectors
    for (auto const& colkey : collections_) {
      // Build an empty list for each collection
      sortedCollections_[colkey.first] = std::make_shared<SortedListPtr::element_type>();
    }

    tau_->setjetRef(jet);
  }

  void RecoTauConstructor::addPFCand(Region region, ParticleType type, const CandidatePtr& ptr, bool skipAddToP4) {
    LogDebug("TauConstructorAddPFCand") << " region = " << region << ", type = " << type << ": Pt = " << ptr->pt()
                                        << ", eta = " << ptr->eta() << ", phi = " << ptr->phi();
    if (region == kSignal) {
      // Keep track of the four vector of the signal vector products added so far.
      // If a photon add it if we are not using PiZeros to build the gammas
      if (((type != kGamma) || !copyGammas_) && !skipAddToP4) {
        LogDebug("TauConstructorAddPFCand") << "--> adding PFCand to tauP4.";
        p4_ += ptr->p4();
      }
    }
    getSortedCollection(region, type)->push_back(ptr);
    // Add to global collection
    getSortedCollection(region, kAll)->push_back(ptr);
  }

  void RecoTauConstructor::reserve(Region region, ParticleType type, size_t size) {
    getSortedCollection(region, type)->reserve(size);
    getCollection(region, type)->reserve(size);
    // Reserve global collection as well
    getSortedCollection(region, kAll)->reserve(getSortedCollection(region, kAll)->size() + size);
    getCollection(region, kAll)->reserve(getCollection(region, kAll)->size() + size);
  }

  void RecoTauConstructor::reserveTauChargedHadron(Region region, size_t size) {
    if (region == kSignal) {
      tau_->signalTauChargedHadronCandidatesRestricted().reserve(size);
      tau_->selectedSignalChargedHadrCands_.reserve(size);
    } else {
      tau_->isolationTauChargedHadronCandidatesRestricted().reserve(size);
      tau_->selectedIsolationChargedHadrCands_.reserve(size);
    }
  }

  namespace {
    void checkOverlap(const CandidatePtr& neutral, const std::vector<CandidatePtr>& pfGammas, bool& isUnique) {
      LogDebug("TauConstructorCheckOverlap") << " pfGammas: #entries = " << pfGammas.size();
      for (std::vector<CandidatePtr>::const_iterator pfGamma = pfGammas.begin(); pfGamma != pfGammas.end(); ++pfGamma) {
        LogDebug("TauConstructorCheckOverlap") << "pfGamma = " << pfGamma->id() << ":" << pfGamma->key();
        if ((*pfGamma).refCore() == neutral.refCore() && (*pfGamma).key() == neutral.key())
          isUnique = false;
      }
    }

    void checkOverlap(const CandidatePtr& neutral, const std::vector<reco::RecoTauPiZero>& piZeros, bool& isUnique) {
      LogDebug("TauConstructorCheckOverlap") << " piZeros: #entries = " << piZeros.size();
      for (std::vector<reco::RecoTauPiZero>::const_iterator piZero = piZeros.begin(); piZero != piZeros.end();
           ++piZero) {
        size_t numPFGammas = piZero->numberOfDaughters();
        for (size_t iPFGamma = 0; iPFGamma < numPFGammas; ++iPFGamma) {
          reco::CandidatePtr pfGamma = piZero->daughterPtr(iPFGamma);
          LogDebug("TauConstructorCheckOverlap") << "pfGamma = " << pfGamma.id() << ":" << pfGamma.key();
          if (pfGamma.id() == neutral.id() && pfGamma.key() == neutral.key())
            isUnique = false;
        }
      }
    }
  }  // namespace

  void RecoTauConstructor::addTauChargedHadron(Region region, const PFRecoTauChargedHadron& chargedHadron) {
    LogDebug("TauConstructorAddChH") << " region = " << region << ": Pt = " << chargedHadron.pt()
                                     << ", eta = " << chargedHadron.eta() << ", phi = " << chargedHadron.phi();
    // CV: need to make sure that PFGammas merged with ChargedHadrons are not part of PiZeros
    const std::vector<CandidatePtr>& neutrals = chargedHadron.getNeutralPFCandidates();
    std::vector<CandidatePtr> neutrals_cleaned;
    for (std::vector<CandidatePtr>::const_iterator neutral = neutrals.begin(); neutral != neutrals.end(); ++neutral) {
      LogDebug("TauConstructorAddChH") << "neutral = " << neutral->id() << ":" << neutral->key();
      bool isUnique = true;
      if (copyGammas_)
        checkOverlap(*neutral, *getSortedCollection(kSignal, kGamma), isUnique);
      else
        checkOverlap(*neutral, tau_->signalPiZeroCandidatesRestricted(), isUnique);
      if (region == kIsolation) {
        if (copyGammas_)
          checkOverlap(*neutral, *getSortedCollection(kIsolation, kGamma), isUnique);
        else
          checkOverlap(*neutral, tau_->isolationPiZeroCandidatesRestricted(), isUnique);
      }
      LogDebug("TauConstructorAddChH") << "--> isUnique = " << isUnique;
      if (isUnique)
        neutrals_cleaned.push_back(*neutral);
    }
    PFRecoTauChargedHadron chargedHadron_cleaned = chargedHadron;
    if (neutrals_cleaned.size() != neutrals.size()) {
      chargedHadron_cleaned.neutralPFCandidates_ = neutrals_cleaned;
      setChargedHadronP4(chargedHadron_cleaned);
    }
    if (region == kSignal) {
      tau_->signalTauChargedHadronCandidatesRestricted().push_back(chargedHadron_cleaned);
      p4_ += chargedHadron_cleaned.p4();
      if (chargedHadron_cleaned.getChargedPFCandidate().isNonnull()) {
        addPFCand(kSignal, kChargedHadron, convertToPtr(chargedHadron_cleaned.getChargedPFCandidate()), true);
      }
      const std::vector<CandidatePtr>& neutrals = chargedHadron_cleaned.getNeutralPFCandidates();
      for (std::vector<CandidatePtr>::const_iterator neutral = neutrals.begin(); neutral != neutrals.end(); ++neutral) {
        if (std::abs((*neutral)->pdgId()) == 22)
          addPFCand(kSignal, kGamma, convertToPtr(*neutral), true);
        else if (std::abs((*neutral)->pdgId()) == 130)
          addPFCand(kSignal, kNeutralHadron, convertToPtr(*neutral), true);
      };
    } else {
      tau_->isolationTauChargedHadronCandidatesRestricted().push_back(chargedHadron_cleaned);
      if (chargedHadron_cleaned.getChargedPFCandidate().isNonnull()) {
        if (std::abs(chargedHadron_cleaned.getChargedPFCandidate()->pdgId()) == 211)
          addPFCand(kIsolation, kChargedHadron, convertToPtr(chargedHadron_cleaned.getChargedPFCandidate()));
        else if (std::abs(chargedHadron_cleaned.getChargedPFCandidate()->pdgId()) == 130)
          addPFCand(kIsolation, kNeutralHadron, convertToPtr(chargedHadron_cleaned.getChargedPFCandidate()));
      }
      const std::vector<CandidatePtr>& neutrals = chargedHadron_cleaned.getNeutralPFCandidates();
      for (std::vector<CandidatePtr>::const_iterator neutral = neutrals.begin(); neutral != neutrals.end(); ++neutral) {
        if (std::abs((*neutral)->pdgId()) == 22)
          addPFCand(kIsolation, kGamma, convertToPtr(*neutral));
        else if (std::abs((*neutral)->pdgId()) == 130)
          addPFCand(kIsolation, kNeutralHadron, convertToPtr(*neutral));
      };
    }
  }

  void RecoTauConstructor::reservePiZero(Region region, size_t size) {
    if (region == kSignal) {
      tau_->signalPiZeroCandidatesRestricted().reserve(size);
      // If we are building the gammas with the pizeros, resize that
      // vector as well
      if (copyGammas_)
        reserve(kSignal, kGamma, 2 * size);
    } else {
      tau_->isolationPiZeroCandidatesRestricted().reserve(size);
      if (copyGammas_)
        reserve(kIsolation, kGamma, 2 * size);
    }
  }

  void RecoTauConstructor::addPiZero(Region region, const RecoTauPiZero& piZero) {
    LogDebug("TauConstructorAddPi0") << " region = " << region << ": Pt = " << piZero.pt() << ", eta = " << piZero.eta()
                                     << ", phi = " << piZero.phi();
    if (region == kSignal) {
      tau_->signalPiZeroCandidatesRestricted().push_back(piZero);
      // Copy the daughter gammas into the gamma collection if desired
      if (copyGammas_) {
        // If we are using the pizeros to build the gammas, make sure we update
        // the four vector correctly.
        p4_ += piZero.p4();
        addPFCands(kSignal, kGamma, piZero.daughterPtrVector().begin(), piZero.daughterPtrVector().end());
      }
    } else {
      tau_->isolationPiZeroCandidatesRestricted().push_back(piZero);
      if (copyGammas_) {
        addPFCands(kIsolation, kGamma, piZero.daughterPtrVector().begin(), piZero.daughterPtrVector().end());
      }
    }
  }

  std::vector<CandidatePtr>* RecoTauConstructor::getCollection(Region region, ParticleType type) {
    return collections_[std::make_pair(region, type)];
  }

  RecoTauConstructor::SortedListPtr RecoTauConstructor::getSortedCollection(Region region, ParticleType type) {
    return sortedCollections_[std::make_pair(region, type)];
  }

  // Trivial converter needed for polymorphism
  CandidatePtr RecoTauConstructor::convertToPtr(const CandidatePtr& pfPtr) const { return pfPtr; }

  namespace {
    // Make sure the two products come from the same EDM source
    template <typename T1, typename T2>
    void checkMatchedProductIds(const T1& t1, const T2& t2) {
      if (t1.id() != t2.id()) {
        throw cms::Exception("MismatchedPFCandSrc")
            << "Error: the input tag"
            << " for the PF candidate collection provided to the RecoTauBuilder "
            << " does not match the one that was used to build the source jets."
            << " Please update the pfCandSrc paramters for the PFTau builders.";
      }
    }
  }  // namespace

  // Convert from a CandidateRef to a Ptr
  CandidatePtr RecoTauConstructor::convertToPtr(const PFCandidatePtr& candPtr) const {
    if (candPtr.isNonnull()) {
      checkMatchedProductIds(candPtr, pfCands_);
      return CandidatePtr(pfCands_, candPtr.key());
    } else
      return PFCandidatePtr();
  }

  namespace {
    template <typename T>
    bool ptDescending(const T& a, const T& b) {
      return a.pt() > b.pt();
    }
    template <typename T>
    bool ptDescendingPtr(const T& a, const T& b) {
      return a->pt() > b->pt();
    }
  }  // namespace

  void RecoTauConstructor::sortAndCopyIntoTau() {
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
    for (auto const& colkey : collections_) {
      SortedListPtr sortedCollection = sortedCollections_[colkey.first];
      std::sort(sortedCollection->begin(), sortedCollection->end(), ptDescendingPtr<CandidatePtr>);
      // Copy into the real tau collection
      for (std::vector<CandidatePtr>::const_iterator particle = sortedCollection->begin();
           particle != sortedCollection->end();
           ++particle) {
        colkey.second->push_back(*particle);
      }
    }
  }

  namespace {
    PFTau::hadronicDecayMode calculateDecayMode(const reco::PFTau& tau,
                                                double dRsignalCone,
                                                double minAbsPhotonSumPt_insideSignalCone,
                                                double minRelPhotonSumPt_insideSignalCone,
                                                double minAbsPhotonSumPt_outsideSignalCone,
                                                double minRelPhotonSumPt_outsideSignalCone) {
      unsigned int nCharged = tau.signalTauChargedHadronCandidates().size();
      // If no tracks exist, this is definitely not a tau!
      if (!nCharged)
        return PFTau::kNull;

      unsigned int nPiZeros = 0;
      const std::vector<RecoTauPiZero>& piZeros = tau.signalPiZeroCandidates();
      for (std::vector<RecoTauPiZero>::const_iterator piZero = piZeros.begin(); piZero != piZeros.end(); ++piZero) {
        double photonSumPt_insideSignalCone = 0.;
        double photonSumPt_outsideSignalCone = 0.;
        int numPhotons = piZero->numberOfDaughters();
        for (int idxPhoton = 0; idxPhoton < numPhotons; ++idxPhoton) {
          const reco::Candidate* photon = piZero->daughter(idxPhoton);
          double dR = deltaR(photon->p4(), tau.p4());
          if (dR < dRsignalCone) {
            photonSumPt_insideSignalCone += photon->pt();
          } else {
            photonSumPt_outsideSignalCone += photon->pt();
          }
        }
        if (photonSumPt_insideSignalCone > minAbsPhotonSumPt_insideSignalCone ||
            photonSumPt_insideSignalCone > (minRelPhotonSumPt_insideSignalCone * tau.pt()) ||
            photonSumPt_outsideSignalCone > minAbsPhotonSumPt_outsideSignalCone ||
            photonSumPt_outsideSignalCone > (minRelPhotonSumPt_outsideSignalCone * tau.pt()))
          ++nPiZeros;
      }

      // Find the maximum number of PiZeros our parameterization can hold
      const unsigned int maxPiZeros = PFTau::kOneProngNPiZero;

      // Determine our track index
      unsigned int trackIndex = (nCharged - 1) * (maxPiZeros + 1);

      // Check if we handle the given number of tracks
      if (trackIndex >= PFTau::kRareDecayMode)
        return PFTau::kRareDecayMode;

      if (nPiZeros > maxPiZeros)
        nPiZeros = maxPiZeros;
      return static_cast<PFTau::hadronicDecayMode>(trackIndex + nPiZeros);
    }
  }  // namespace

  std::unique_ptr<reco::PFTau> RecoTauConstructor::get(bool setupLeadingObjects) {
    LogDebug("TauConstructorGet") << "Start getting";

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
    for (std::vector<PFRecoTauChargedHadron>::const_iterator chargedHadron =
             tau_->signalTauChargedHadronCandidatesRestricted().begin();
         chargedHadron != tau_->signalTauChargedHadronCandidatesRestricted().end();
         ++chargedHadron) {
      if (chargedHadron->algoIs(PFRecoTauChargedHadron::kChargedPFCandidate) ||
          chargedHadron->algoIs(PFRecoTauChargedHadron::kTrack)) {
        ++nCharged;
        charge += chargedHadron->charge();
        if (chargedHadron->pt() > leadChargedHadronPt) {
          leadChargedHadronPt = chargedHadron->pt();
          leadChargedHadronCharge = chargedHadron->charge();
        }
      }
    }
    if (nCharged == 3)
      tau_->setCharge(charge);
    else
      tau_->setCharge(leadChargedHadronCharge);

    // Set PDG id
    tau_->setPdgId(tau_->charge() < 0 ? 15 : -15);

    // Set P4
    tau_->setP4(p4_);

    // Set Decay Mode
    double dRsignalCone = (signalConeSize_) ? (*signalConeSize_)(*tau_) : 0.5;
    tau_->setSignalConeSize(dRsignalCone);
    PFTau::hadronicDecayMode dm = calculateDecayMode(*tau_,
                                                     dRsignalCone,
                                                     minAbsPhotonSumPt_insideSignalCone_,
                                                     minRelPhotonSumPt_insideSignalCone_,
                                                     minAbsPhotonSumPt_outsideSignalCone_,
                                                     minRelPhotonSumPt_outsideSignalCone_);
    tau_->setDecayMode(dm);

    LogDebug("TauConstructorGet") << "Pt = " << tau_->pt() << ", eta = " << tau_->eta() << ", phi = " << tau_->phi()
                                  << ", mass = " << tau_->mass() << ", dm = " << tau_->decayMode();

    // Set charged isolation quantities
    tau_->setisolationPFChargedHadrCandsPtSum(sumPFCandPt(getCollection(kIsolation, kChargedHadron)->begin(),
                                                          getCollection(kIsolation, kChargedHadron)->end()));

    // Set gamma isolation quantities
    tau_->setisolationPFGammaCandsEtSum(
        sumPFCandPt(getCollection(kIsolation, kGamma)->begin(), getCollection(kIsolation, kGamma)->end()));

    // Set em fraction
    tau_->setemFraction(sumPFCandPt(getCollection(kSignal, kGamma)->begin(), getCollection(kSignal, kGamma)->end()) /
                        tau_->pt());

    if (setupLeadingObjects) {
      typedef std::vector<CandidatePtr>::const_iterator Iter;
      // Find the highest PT object in the signal cone
      Iter leadingCand = leadCand(getCollection(kSignal, kAll)->begin(), getCollection(kSignal, kAll)->end());

      if (leadingCand != getCollection(kSignal, kAll)->end())
        tau_->setleadCand(*leadingCand);

      // Hardest charged object in signal cone
      Iter leadingChargedCand =
          leadCand(getCollection(kSignal, kChargedHadron)->begin(), getCollection(kSignal, kChargedHadron)->end());

      if (leadingChargedCand != getCollection(kSignal, kChargedHadron)->end())
        tau_->setleadChargedHadrCand(*leadingChargedCand);

      // Hardest gamma object in signal cone
      Iter leadingGammaCand = leadCand(getCollection(kSignal, kGamma)->begin(), getCollection(kSignal, kGamma)->end());

      if (leadingGammaCand != getCollection(kSignal, kGamma)->end())
        tau_->setleadNeutralCand(*leadingGammaCand);
    }
    return std::move(tau_);
  }
}  // end namespace reco::tau
