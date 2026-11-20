// -*- C++ -*-
//
// Package:     DataFormats/PatCandidates
// Class:       pat::GenJet (pat::io_v1::GenJet)
//
// Analysis-level generator jet: PATObject<reco::GenJet> with the parts of the pat::Jet API
// (DataFormats/PatCandidates/interface/Jet.h, CMSSW_20_1_0_pre3) that make sense for generator jets.
// Not provided (no generator meaning): b tagging and tag infos, jet ID, calo/JPT/PF specifics and the
// jet-type queries, calorimeter, HF/HO and JPT track quantities, calo-tower and PF-candidate constituents,
// associated tracks, HF multiplicities, jet flavour (available from the JetFlavourInfoMatchingCollection of the
// generator jets).
// Behaviour on missing data:
//   - JEC: factor 1 at every level, correctedJet/correctedP4 return the jet itself
//   - groomedMass -1, missing subjet collections empty
//   - PF-named energies/fractions/multiplicities are filled from reco::GenJet::Specific
//   - genJet() returns the jet itself; genJetFwdRef() points to the reco::GenJet the jet was made from
//   - genCandidates(): generator particles stored with setGenCandidates, else empty
//   - no scouting constructor
// Comments "copy" / "mapped" say where a body comes from.

#ifndef DataFormats_PatCandidates_GenJet_h
#define DataFormats_PatCandidates_GenJet_h

#include "DataFormats/PatCandidates/interface/PATObject.h"
#include "DataFormats/PatCandidates/interface/JetCorrFactors.h"
#include "DataFormats/JetReco/interface/GenJet.h"
#include "DataFormats/JetReco/interface/GenJetCollection.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/HepMCCandidate/interface/GenParticleFwd.h"
#include "DataFormats/Common/interface/Ptr.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefVector.h"
#include "DataFormats/Common/interface/FwdPtr.h"
#include "DataFormats/Common/interface/FwdRef.h"
#include "DataFormats/Common/interface/AtomicPtrCache.h"

#include <algorithm>
#include <numeric>
#include <ostream>
#include <string>
#include <vector>

namespace pat {
  namespace io_v1 {
    class GenJet;
  }
  using GenJet = io_v1::GenJet;
  typedef std::vector<GenJet> GenJetCollection;
  typedef edm::Ref<GenJetCollection> GenJetRef;
  typedef edm::RefVector<GenJetCollection> GenJetRefVector;
  /// counterpart of pat::JetPtrCollection (Jet.h:83) for generator subjets
  typedef std::vector<edm::Ptr<pat::GenJet> > GenJetPtrCollection;
}  // namespace pat

namespace reco {
  /// pipe operator, counterpart of reco::operator<<(std::ostream&, const pat::Jet&) (Jet.h:67)
  std::ostream& operator<<(std::ostream& out, const pat::GenJet& obj);
}  // namespace reco

namespace pat {

  namespace io_v1 {
    class GenJet : public PATObject<reco::GenJet> {
    public:
      // ---------------------------------------------------------------- constructors
      GenJet();
      GenJet(const reco::GenJet& aJet);  // implicit, as pat::Jet
      /// from a reco::GenJet in a std::vector<reco::GenJet>: genJetFwdRef() points to it
      GenJet(const edm::RefToBase<reco::GenJet>& aJetRef);  // implicit, as pat::Jet
      GenJet(const edm::Ptr<reco::GenJet>& aJetRef);        // implicit, as pat::Jet
      GenJet(const edm::RefToBase<pat::GenJet>& aJetRef);   // implicit, as pat::Jet
      GenJet(const edm::Ptr<pat::GenJet>& aJetRef);         // implicit, as pat::Jet
      // no Run3ScoutingPFJet constructor
      ~GenJet() override;
      GenJet* clone() const override { return new GenJet(*this); }

      // ---------------------------------------------------------------- MC matching
      const reco::GenParticle* genParton() const { return genParticle(); }
      /// the jet itself
      const reco::GenJet* genJet() const { return this; }

      // ---------------------------------------------------------------- JEC
      // A generator jet is its own uncorrected level: factor 1 at every level, for every set.
      const std::vector<std::string> availableJECSets() const { return {}; }
      /// pat::Jet throws std::out_of_range for set>=0 (Jet.cc:344); here: empty list
      const std::vector<std::string> availableJECLevels(const int& /*set*/ = 0) const { return {}; }
      const std::vector<std::string> availableJECLevels(const std::string& /*set*/) const { return {}; }
      bool jecSetsAvailable() const { return false; }
      bool jecSetAvailable(const std::string& /*set*/) const { return false; }
      bool jecSetAvailable(const unsigned int& /*set*/) const { return false; }
      /// same missing-data value as pat::Jet (Jet.h:149, Jet.h:153)
      std::string currentJECSet() const { return std::string("ERROR"); }
      std::string currentJECLevel() const { return std::string("ERROR"); }
      /// pat::Jet returns an uninitialised member (Jet.h:641); here a defined NONE
      JetCorrFactors::Flavor currentJECFlavor() const { return JetCorrFactors::NONE; }
      float jecFactor(const std::string& /*level*/,
                      const std::string& /*flavor*/ = "none",
                      const std::string& /*set*/ = "") const {
        return 1.f;
      }
      float jecFactor(const unsigned int& /*level*/,
                      const JetCorrFactors::Flavor& /*flavor*/ = JetCorrFactors::NONE,
                      const unsigned int& /*set*/ = 0) const {
        return 1.f;
      }
      /// returns a pat::GenJet copy (signature adapted from pat::Jet)
      GenJet correctedJet(const std::string& level,
                          const std::string& flavor = "none",
                          const std::string& set = "") const;
      GenJet correctedJet(const unsigned int& level,
                          const JetCorrFactors::Flavor& flavor = JetCorrFactors::NONE,
                          const unsigned int& set = 0) const;
      const LorentzVector correctedP4(const std::string& /*level*/,
                                      const std::string& /*flavor*/ = "none",
                                      const std::string& /*set*/ = "") const {
        return p4();
      }
      const LorentzVector correctedP4(const unsigned int& /*level*/,
                                      const JetCorrFactors::Flavor& /*flavor*/ = JetCorrFactors::NONE,
                                      const unsigned int& /*set*/ = 0) const {
        return p4();
      }
      /// no JEC: only rescales the four-momentum, as pat::Jet does without JEC (Jet.cc:313)
      void scaleEnergy(double fScale) override { scaleEnergy(fScale, "Unscaled"); }
      void scaleEnergy(double fScale, const std::string& /*level*/) { setP4(p4() * fScale); }

      // ---------------------------------------------------------------- jet charge (copy; never filled)
      float jetCharge() const { return jetCharge_; }
      void setJetCharge(float jetCharge) { jetCharge_ = jetCharge; }

      // ---------------------------------------------------------------- embedding setters
      /// generator-particle counterpart of pat::Jet::setPFCandidates (Jet.cc)
      void setGenCandidates(const reco::GenParticleFwdPtrVector& genCandidates);
      void setGenParton(const reco::GenParticleRef& gp, bool embed = false) { setGenParticleRef(gp, embed); }

      // ---------------------------------------------------------------- PF-named
      // NOT redeclared (inherited from reco::GenJet with identical signature, GenJet.h:95-104):
      //   chargedHadronEnergy, neutralHadronEnergy, chargedEmEnergy, neutralEmEnergy, muonEnergy,
      //   chargedHadronMultiplicity, neutralHadronMultiplicity, muonMultiplicity
      /// mapped: m_ChargedHadronMultiplicity + m_ChargedEmMultiplicity + m_MuonMultiplicity
      int chargedMultiplicity() const {
        return chargedHadronMultiplicity() + chargedEmMultiplicity() + muonMultiplicity();
      }
      /// copy of pat::Jet bodies (Jet.h:409-423); jecSetsAvailable() is false, so the denominator is energy()
      float chargedHadronEnergyFraction() const { return chargedHadronEnergy() / energy(); }
      float neutralHadronEnergyFraction() const { return neutralHadronEnergy() / energy(); }
      float chargedEmEnergyFraction() const { return chargedEmEnergy() / energy(); }
      float neutralEmEnergyFraction() const { return neutralEmEnergy() / energy(); }
      /// mapped: photons = m_NeutralEmEnergy, electrons = m_ChargedEmEnergy
      float photonEnergy() const { return neutralEmEnergy(); }
      float photonEnergyFraction() const { return photonEnergy() / energy(); }
      float electronEnergy() const { return chargedEmEnergy(); }
      float electronEnergyFraction() const { return electronEnergy() / energy(); }
      float muonEnergyFraction() const { return muonEnergy() / energy(); }
      /// mapped: photon = neutral EM, electron = charged EM multiplicity (GenJet.h:102-103)
      int photonMultiplicity() const { return neutralEmMultiplicity(); }
      int electronMultiplicity() const { return chargedEmMultiplicity(); }
      /// mapped: m_MuonEnergy
      float chargedMuEnergy() const { return muonEnergy(); }
      float chargedMuEnergyFraction() const { return chargedMuEnergy() / energy(); }
      /// mapped: m_NeutralHadronMultiplicity + m_NeutralEmMultiplicity
      int neutralMultiplicity() const { return neutralHadronMultiplicity() + neutralEmMultiplicity(); }

      // ---------------------------------------------------------------- constituents (copy, subjet-aware)
      const reco::Candidate* daughter(size_t i) const override;
      reco::CandidatePtr daughterPtr(size_t i) const override;
      const reco::CompositePtrCandidate::daughters& daughterPtrVector() const override;
      using reco::LeafCandidate::daughter;
      size_t numberOfDaughters() const override;
      void clearDaughters() override {
        PATObject<reco::GenJet>::clearDaughters();
        daughtersTemp_.reset();
      }

      // ---------------------------------------------------------------- generator particles, forward pointers
      /// forward Ptrs of the stored generator particles, as pat::Jet::getPFConstituents (dereference
      /// genCandidatesFwdPtr() to fall back to the back Ptr); empty if none were set
      std::vector<reco::GenParticlePtr> const& genCandidates() const;
      reco::GenParticleFwdPtrVector const& genCandidatesFwdPtr() const { return genCandidatesFwdPtr_; }
      /// forward and back ref: the reco::GenJet the jet was made from (null for a jet built from an object)
      edm::FwdRef<reco::GenJetCollection> const& genJetFwdRef() const { return genJetFwdRef_; }
      /// as pat::Jet inside the range; out of range: no-op instead of OutOfRange
      void updateFwdGenCandidateFwdPtr(unsigned int index, const reco::GenParticlePtr& updateFwd);
      void updateFwdGenJetFwdRef(reco::GenJetRef updateRef) {
        genJetFwdRef_ = edm::FwdRef<reco::GenJetCollection>(updateRef, genJetFwdRef_.backRef());
      }

      friend std::ostream& reco::operator<<(std::ostream& out, const pat::GenJet& obj);

      // ---------------------------------------------------------------- subjets
      /// missing index or label: static empty collection, not OutOfRange/SubjetsNotFound
      pat::GenJetPtrCollection const& subjets(unsigned int index = 0) const;
      pat::GenJetPtrCollection const& subjets(std::string const& label) const;
      void addSubjets(pat::GenJetPtrCollection const& pieces, std::string const& label = "");
      bool hasSubjets(std::string const& label) const {
        return std::find(subjetLabels_.begin(), subjetLabels_.end(), label) != subjetLabels_.end();
      }
      unsigned int nSubjetCollections() const { return subjetCollections_.size(); }
      std::vector<std::string> const& subjetCollectionNames() const { return subjetLabels_; }
      /// guard evaluated first; -1 if the collection is missing or empty
      double groomedMass(unsigned int index = 0) const {
        return index < nSubjetCollections() ? sumMass(subjetCollections_[index]) : -1.0;
      }
      double groomedMass(std::string const& label) const { return hasSubjets(label) ? sumMass(subjets(label)) : -1.0; }

      // ---------------------------------------------------------------- inherited, listed for the reader
      // PATObject<reco::GenJet>: user data, trigger matches, efficiencies, overlaps, resolutions,
      //   genParticle refs (userFloat/userInt with a missing key, efficiency, getKinResolution and
      //   resol* keep throwing:  exception).
      // reco::GenJet: emEnergy, hadEnergy, invisibleEnergy, auxiliaryEnergy, chargedEmMultiplicity,
      //   neutralEmMultiplicity, detectorEta, getGenConstituent(s), getSpecific, setSpecific.
      //   Note: PATObject::genParticle(size_t) hides the static reco::GenJet::genParticle(const Candidate*);
      //   call it as reco::GenJet::genParticle(c).

    protected:
      // ---- embedding
      edm::AtomicPtrCache<std::vector<reco::GenParticlePtr> > genCandidatesTemp_;  // transient
      reco::GenParticleFwdPtrVector genCandidatesFwdPtr_;

      // ---- substructure
      std::vector<pat::GenJetPtrCollection> subjetCollections_;
      std::vector<std::string> subjetLabels_;
      edm::AtomicPtrCache<std::vector<reco::CandidatePtr> > daughtersTemp_;  // transient

      // ---- MC
      edm::FwdRef<reco::GenJetCollection> genJetFwdRef_;  // the reco::GenJet the jet was made from

      // ---- jet charge
      float jetCharge_;

    private:
      static double sumMass(pat::GenJetPtrCollection const& sub) {
        return sub.empty() ? -1.0
                           : std::accumulate(sub.begin(),
                                             sub.end(),
                                             reco::Candidate::LorentzVector(),
                                             [](reco::Candidate::LorentzVector const& a,
                                                edm::Ptr<pat::GenJet> const& b) { return a + b->p4(); })
                                 .mass();
      }
      void cacheGenCandidates() const;  // stored FwdPtrs only; no throw
      void cacheDaughters() const;      // copy of Jet.cc:616
      static pat::GenJetPtrCollection const& emptySubjets();
    };
  }  // namespace io_v1
}  // namespace pat

#endif
