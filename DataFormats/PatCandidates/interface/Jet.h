//
//

#ifndef DataFormats_PatCandidates_Jet_h
#define DataFormats_PatCandidates_Jet_h

/**
  \class    pat::Jet Jet.h "DataFormats/PatCandidates/interface/Jet.h"
  \brief    Analysis-level calorimeter jet class

   Jet implements the analysis-level calorimeter jet class within the
   'pat' namespace

  \author   Steven Lowette, Giovanni Petrucciani, Roger Wolf, Christian Autermann
*/

#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/JetReco/interface/BasicJet.h"
#include "DataFormats/JetReco/interface/JPTJet.h"
#include "DataFormats/JetReco/interface/PFJet.h"
#include "DataFormats/CaloTowers/interface/CaloTower.h"
#include "DataFormats/JetReco/interface/GenJet.h"
#include "DataFormats/Candidate/interface/Particle.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/BTauReco/interface/JetTag.h"
#include "DataFormats/PatCandidates/interface/PATObject.h"
#include "DataFormats/BTauReco/interface/CandIPTagInfo.h"
#include "DataFormats/BTauReco/interface/TrackIPTagInfo.h"
#include "DataFormats/BTauReco/interface/TrackProbabilityTagInfo.h"
#include "DataFormats/BTauReco/interface/TrackCountingTagInfo.h"
#include "DataFormats/BTauReco/interface/CandSoftLeptonTagInfo.h"
#include "DataFormats/BTauReco/interface/SoftLeptonTagInfo.h"
#include "DataFormats/JetMatching/interface/JetFlavourInfo.h"
#include "DataFormats/BTauReco/interface/PixelClusterTagInfo.h"
#include "DataFormats/BTauReco/interface/CandSecondaryVertexTagInfo.h"
#include "DataFormats/BTauReco/interface/SecondaryVertexTagInfo.h"
#include "DataFormats/BTauReco/interface/BoostedDoubleSVTagInfo.h"
#include "DataFormats/PatCandidates/interface/JetCorrFactors.h"
#include "DataFormats/JetReco/interface/JetID.h"

#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"

#include "DataFormats/Common/interface/Ptr.h"
#include "DataFormats/Common/interface/OwnVector.h"
#include "DataFormats/Common/interface/AtomicPtrCache.h"

#include <numeric>

// Define typedefs for convenience
namespace pat {
  class Jet;
  typedef std::vector<Jet> JetCollection;
  typedef edm::Ref<JetCollection> JetRef;
  typedef edm::RefVector<JetCollection> JetRefVector;
}  // namespace pat

namespace reco {
  /// pipe operator (introduced to use pat::Jet with PFTopProjectors)
  std::ostream& operator<<(std::ostream& out, const pat::Jet& obj);
}  // namespace reco

// Class definition
namespace pat {

  class PATJetSlimmer;

  typedef reco::CaloJet::Specific CaloSpecific;
  typedef reco::JPTJet::Specific JPTSpecific;
  typedef reco::PFJet::Specific PFSpecific;
  typedef std::vector<edm::FwdPtr<reco::BaseTagInfo> > TagInfoFwdPtrCollection;
  typedef std::vector<edm::FwdPtr<reco::PFCandidate> > PFCandidateFwdPtrCollection;
  typedef std::vector<edm::FwdPtr<CaloTower> > CaloTowerFwdPtrCollection;
  typedef std::vector<edm::Ptr<pat::Jet> > JetPtrCollection;

  class Jet : public PATObject<reco::Jet> {
    /// make friends with PATJetProducer so that it can set the an initial
    /// jet energy scale unequal to raw calling the private initializeJEC
    /// function, which should be non accessible to any other user
    friend class PATJetProducer;
    friend class PATJetSlimmer;
    friend class PATJetUpdater;

  public:
    /// default constructor
    Jet();
    /// constructor from a reco::Jet
    Jet(const reco::Jet& aJet);
    /// constructor from ref to reco::Jet
    Jet(const edm::RefToBase<reco::Jet>& aJetRef);
    /// constructor from ref to reco::Jet
    Jet(const edm::Ptr<reco::Jet>& aJetRef);
    /// constructure from ref to pat::Jet
    Jet(const edm::RefToBase<pat::Jet>& aJetRef);
    /// constructure from ref to pat::Jet
    Jet(const edm::Ptr<pat::Jet>& aJetRef);
    /// destructor
    ~Jet() override;
    /// required reimplementation of the Candidate's clone method
    Jet* clone() const override { return new Jet(*this); }

    /// ---- methods for MC matching ----

    /// return the matched generated parton
    const reco::GenParticle* genParton() const { return genParticle(); }
    /// return the matched generated jet
    const reco::GenJet* genJet() const;
    /// return the parton-based flavour of the jet
    int partonFlavour() const;
    /// return the hadron-based flavour of the jet
    int hadronFlavour() const;
    /// return the JetFlavourInfo of the jet
    const reco::JetFlavourInfo& jetFlavourInfo() const;

  public:
    /// ---- methods for jet corrections ----

    /// returns the labels of all available sets of jet energy corrections
    const std::vector<std::string> availableJECSets() const;
    // returns the available JEC Levels for a given jecSet
    const std::vector<std::string> availableJECLevels(const int& set = 0) const;
    // returns the available JEC Levels for a given jecSet
    const std::vector<std::string> availableJECLevels(const std::string& set) const {
      return availableJECLevels(jecSet(set));
    };
    /// returns true if the jet carries jet energy correction information
    /// at all
    bool jecSetsAvailable() const { return !jec_.empty(); }
    /// returns true if the jet carries a set of jet energy correction
    /// factors with the given label
    bool jecSetAvailable(const std::string& set) const { return (jecSet(set) >= 0); };
    /// returns true if the jet carries a set of jet energy correction
    /// factors with the given label
    bool jecSetAvailable(const unsigned int& set) const { return (set < jec_.size()); };
    /// returns the label of the current set of jet energy corrections
    std::string currentJECSet() const {
      return currentJECSet_ < jec_.size() ? jec_.at(currentJECSet_).jecSet() : std::string("ERROR");
    }
    /// return the name of the current step of jet energy corrections
    std::string currentJECLevel() const {
      return currentJECSet_ < jec_.size() ? jec_.at(currentJECSet_).jecLevel(currentJECLevel_) : std::string("ERROR");
    };
    /// return flavour of the current step of jet energy corrections
    JetCorrFactors::Flavor currentJECFlavor() const { return currentJECFlavor_; };
    /// correction factor to the given level for a specific set
    /// of correction factors, starting from the current level
    float jecFactor(const std::string& level, const std::string& flavor = "none", const std::string& set = "") const;
    /// correction factor to the given level for a specific set
    /// of correction factors, starting from the current level
    float jecFactor(const unsigned int& level,
                    const JetCorrFactors::Flavor& flavor = JetCorrFactors::NONE,
                    const unsigned int& set = 0) const;
    /// copy of the jet corrected up to the given level for the set
    /// of jet energy correction factors, which is currently in use
    Jet correctedJet(const std::string& level, const std::string& flavor = "none", const std::string& set = "") const;
    /// copy of the jet corrected up to the given level for the set
    /// of jet energy correction factors, which is currently in use
    Jet correctedJet(const unsigned int& level,
                     const JetCorrFactors::Flavor& flavor = JetCorrFactors::NONE,
                     const unsigned int& set = 0) const;
    /// p4 of the jet corrected up to the given level for the set
    /// of jet energy correction factors, which is currently in use
    const LorentzVector correctedP4(const std::string& level,
                                    const std::string& flavor = "none",
                                    const std::string& set = "") const {
      return correctedJet(level, flavor, set).p4();
    };
    /// p4 of the jet corrected up to the given level for the set
    /// of jet energy correction factors, which is currently in use
    const LorentzVector correctedP4(const unsigned int& level,
                                    const JetCorrFactors::Flavor& flavor = JetCorrFactors::NONE,
                                    const unsigned int& set = 0) const {
      return correctedJet(level, flavor, set).p4();
    };
    /// Scale energy and correspondingly adjust raw jec factors
    void scaleEnergy(double fScale) override { scaleEnergy(fScale, "Unscaled"); }
    void scaleEnergy(double fScale, const std::string& level);

  private:
    /// index of the set of jec factors with given label; returns -1 if no set
    /// of jec factors exists with the given label
    int jecSet(const std::string& label) const;
    /// update the current JEC set; used by correctedJet
    void currentJECSet(const unsigned int& set) { currentJECSet_ = set; };
    /// update the current JEC level; used by correctedJet
    void currentJECLevel(const unsigned int& level) { currentJECLevel_ = level; };
    /// update the current JEC flavor; used by correctedJet
    void currentJECFlavor(const JetCorrFactors::Flavor& flavor) { currentJECFlavor_ = flavor; };
    /// add more sets of energy correction factors
    void addJECFactors(const JetCorrFactors& jec) { jec_.push_back(jec); };
    /// initialize the jet to a given JEC level during creation starting from Uncorrected
    void initializeJEC(unsigned int level,
                       const JetCorrFactors::Flavor& flavor = JetCorrFactors::NONE,
                       unsigned int set = 0);

  public:
    /// ---- methods for accessing b-tagging info ----

    /// get b discriminant from label name
    float bDiscriminator(const std::string& theLabel) const;
    /// get vector of paire labelname-disciValue
    const std::vector<std::pair<std::string, float> >& getPairDiscri() const;
    /// get list of tag info labels
    std::vector<std::string> const& tagInfoLabels() const { return tagInfoLabels_; }
    /// check to see if the given tag info is nonzero
    bool hasTagInfo(const std::string label) const { return tagInfo(label) != nullptr; }
    /// get a tagInfo with the given name, or NULL if none is found.
    /// You should omit the 'TagInfos' part from the label
    const reco::BaseTagInfo* tagInfo(const std::string& label) const;
    /// get a tagInfo with the given name and type or NULL if none is found.
    /// If the label is empty or not specified, it returns the first tagInfo of that type (if any one exists)
    /// you should omit the 'TagInfos' part from the label
    const reco::CandIPTagInfo* tagInfoCandIP(const std::string& label = "") const;
    const reco::TrackIPTagInfo* tagInfoTrackIP(const std::string& label = "") const;
    /// get a tagInfo with the given name and type or NULL if none is found.
    /// If the label is empty or not specified, it returns the first tagInfo of that type (if any one exists)
    /// you should omit the 'TagInfos' part from the label
    const reco::CandSoftLeptonTagInfo* tagInfoCandSoftLepton(const std::string& label = "") const;
    const reco::SoftLeptonTagInfo* tagInfoSoftLepton(const std::string& label = "") const;
    /// get a tagInfo with the given name and type or NULL if none is found.
    /// If the label is empty or not specified, it returns the first tagInfo of that type (if any one exists)
    /// you should omit the 'TagInfos' part from the label
    const reco::CandSecondaryVertexTagInfo* tagInfoCandSecondaryVertex(const std::string& label = "") const;
    const reco::SecondaryVertexTagInfo* tagInfoSecondaryVertex(const std::string& label = "") const;
    const reco::BoostedDoubleSVTagInfo* tagInfoBoostedDoubleSV(const std::string& label = "") const;
    /// get a tagInfo with the given name and type or NULL if none is found.
    /// If the label is empty or not specified, it returns the first tagInfo of that type (if any one exists)
    /// you should omit the 'TagInfos' part from the label
    const reco::PixelClusterTagInfo* tagInfoPixelCluster(const std::string& label = "") const;
    /// method to add a algolabel-discriminator pair
    void addBDiscriminatorPair(const std::pair<std::string, float>& thePair);
    /// sets a tagInfo with the given name from an edm::Ptr<T> to it.
    /// If the label ends with 'TagInfos', the 'TagInfos' is stripped out.
    void addTagInfo(const std::string& label, const TagInfoFwdPtrCollection::value_type& info);

    // ---- track related methods ----

    /// method to return the JetCharge computed when creating the Jet
    float jetCharge() const;
    /// method to return a vector of refs to the tracks associated to this jet
    const reco::TrackRefVector& associatedTracks() const;
    /// method to set the jet charge
    void setJetCharge(float jetCharge);
    /// method to set the vector of refs to the tracks associated to this jet
    void setAssociatedTracks(const reco::TrackRefVector& tracks);

    // ---- methods for content embedding ----

    /// method to store the CaloJet constituents internally
    void setCaloTowers(const CaloTowerFwdPtrCollection& caloTowers);
    /// method to store the PFCandidate constituents internally
    void setPFCandidates(const PFCandidateFwdPtrCollection& pfCandidates);
    /// method to set the matched parton
    void setGenParton(const reco::GenParticleRef& gp, bool embed = false) { setGenParticleRef(gp, embed); }
    /// method to set the matched generated jet reference, embedding if requested
    void setGenJetRef(const edm::FwdRef<reco::GenJetCollection>& gj);
    /// method to set the parton-based flavour of the jet
    void setPartonFlavour(int partonFl);
    /// method to set the hadron-based flavour of the jet
    void setHadronFlavour(int hadronFl);
    /// method to set the JetFlavourInfo of the jet
    void setJetFlavourInfo(const reco::JetFlavourInfo& jetFlavourInfo);

    /// methods for jet ID
    void setJetID(reco::JetID const& id) { jetID_ = id; }

    // ---- jet specific methods ----

    /// check to see if the jet is a reco::CaloJet
    bool isCaloJet() const { return !specificCalo_.empty() && !isJPTJet(); }
    /// check to see if the jet is a reco::JPTJet
    bool isJPTJet() const { return !specificJPT_.empty(); }
    /// check to see if the jet is a reco::PFJet
    bool isPFJet() const { return !specificPF_.empty(); }
    /// check to see if the jet is no more than a reco::BasicJet
    bool isBasicJet() const { return !(isCaloJet() || isPFJet() || isJPTJet()); }
    /// retrieve the calo specific part of the jet
    const CaloSpecific& caloSpecific() const {
      if (specificCalo_.empty())
        throw cms::Exception("Type Mismatch") << "This PAT jet was not made from a CaloJet.\n";
      return specificCalo_[0];
    }
    /// retrieve the jpt specific part of the jet
    const JPTSpecific& jptSpecific() const {
      if (specificJPT_.empty())
        throw cms::Exception("Type Mismatch") << "This PAT jet was not made from a JPTJet.\n";
      return specificJPT_[0];
    }
    /// check to see if the PFSpecific object is stored
    bool hasPFSpecific() const { return !specificPF_.empty(); }
    /// retrieve the pf specific part of the jet
    const PFSpecific& pfSpecific() const {
      if (specificPF_.empty())
        throw cms::Exception("Type Mismatch") << "This PAT jet was not made from a PFJet.\n";
      return specificPF_[0];
    }
    /// set the calo specific part of the jet
    void setCaloSpecific(const CaloSpecific& newCaloSpecific) {
      if (specificCalo_.empty())
        throw cms::Exception("Type Mismatch") << "This PAT jet was not made from a CaloJet.\n";
      specificCalo_[0] = newCaloSpecific;
    }
    /// set the jpt specific part of the jet
    void setJPTSpecific(const JPTSpecific& newJPTSpecific) {
      if (specificJPT_.empty())
        throw cms::Exception("Type Mismatch") << "This PAT jet was not made from a JPTJet.\n";
      specificJPT_[0] = newJPTSpecific;
    }
    /// set the pf specific part of the jet
    void setPFSpecific(const PFSpecific& newPFSpecific) {
      if (specificPF_.empty())
        throw cms::Exception("Type Mismatch") << "This PAT jet was not made from a PFJet.\n";
      specificPF_[0] = newPFSpecific;
    }

    // ---- Calo Jet specific information ----

    /// returns the maximum energy deposited in ECAL towers
    float maxEInEmTowers() const { return caloSpecific().mMaxEInEmTowers; }
    /// returns the maximum energy deposited in HCAL towers
    float maxEInHadTowers() const { return caloSpecific().mMaxEInHadTowers; }
    /// returns the jet hadronic energy fraction
    float energyFractionHadronic() const { return caloSpecific().mEnergyFractionHadronic; }
    /// returns the jet electromagnetic energy fraction
    float emEnergyFraction() const { return caloSpecific().mEnergyFractionEm; }
    /// returns the jet hadronic energy in HB
    float hadEnergyInHB() const { return caloSpecific().mHadEnergyInHB; }
    /// returns the jet hadronic energy in HO
    float hadEnergyInHO() const { return caloSpecific().mHadEnergyInHO; }
    /// returns the jet hadronic energy in HE
    float hadEnergyInHE() const { return caloSpecific().mHadEnergyInHE; }
    /// returns the jet hadronic energy in HF
    float hadEnergyInHF() const { return caloSpecific().mHadEnergyInHF; }
    /// returns the jet electromagnetic energy in EB
    float emEnergyInEB() const { return caloSpecific().mEmEnergyInEB; }
    /// returns the jet electromagnetic energy in EE
    float emEnergyInEE() const { return caloSpecific().mEmEnergyInEE; }
    /// returns the jet electromagnetic energy extracted from HF
    float emEnergyInHF() const { return caloSpecific().mEmEnergyInHF; }
    /// returns area of contributing towers
    float towersArea() const { return caloSpecific().mTowersArea; }
    /// returns the number of constituents carrying a 90% of the total Jet energy*/
    int n90() const { return nCarrying(0.9); }
    /// returns the number of constituents carrying a 60% of the total Jet energy*/
    int n60() const { return nCarrying(0.6); }

    /// convert generic constituent to specific type
    //  static CaloTowerPtr caloTower (const reco::Candidate* fConstituent);
    /// get specific constituent of the CaloJet.
    /// if the caloTowers were embedded, this reference is transient only and must not be persisted
    CaloTowerPtr getCaloConstituent(unsigned fIndex) const;
    /// get the constituents of the CaloJet.
    /// If the caloTowers were embedded, these reference are transient only and must not be persisted
    std::vector<CaloTowerPtr> const& getCaloConstituents() const;

    // ---- JPT Jet specific information ----

    /// pions fully contained in cone
    const reco::TrackRefVector& pionsInVertexInCalo() const { return jptSpecific().pionsInVertexInCalo; }
    /// pions that curled out
    const reco::TrackRefVector& pionsInVertexOutCalo() const { return jptSpecific().pionsInVertexOutCalo; }
    /// pions that curled in
    const reco::TrackRefVector& pionsOutVertexInCalo() const { return jptSpecific().pionsOutVertexInCalo; }
    /// muons fully contained in cone
    const reco::TrackRefVector& muonsInVertexInCalo() const { return jptSpecific().muonsInVertexInCalo; }
    /// muons that curled out
    const reco::TrackRefVector& muonsInVertexOutCalo() const { return jptSpecific().muonsInVertexOutCalo; }
    /// muons that curled in
    const reco::TrackRefVector& muonsOutVertexInCalo() const { return jptSpecific().muonsOutVertexInCalo; }
    /// electrons fully contained in cone
    const reco::TrackRefVector& elecsInVertexInCalo() const { return jptSpecific().elecsInVertexInCalo; }
    /// electrons that curled out
    const reco::TrackRefVector& elecsInVertexOutCalo() const { return jptSpecific().elecsInVertexOutCalo; }
    /// electrons that curled in
    const reco::TrackRefVector& elecsOutVertexInCalo() const { return jptSpecific().elecsOutVertexInCalo; }
    /// chargedMultiplicity
    float elecMultiplicity() const {
      return jptSpecific().elecsInVertexInCalo.size() + jptSpecific().elecsInVertexOutCalo.size();
    }

    // ---- JPT or PF Jet specific information ----

    /// muonMultiplicity
    int muonMultiplicity() const;
    /// chargedMultiplicity
    int chargedMultiplicity() const;
    /// chargedEmEnergy
    float chargedEmEnergy() const;
    /// neutralEmEnergy
    float neutralEmEnergy() const;
    /// chargedHadronEnergy
    float chargedHadronEnergy() const;
    /// neutralHadronEnergy
    float neutralHadronEnergy() const;

    /// chargedHadronEnergyFraction (relative to uncorrected jet energy)
    float chargedHadronEnergyFraction() const {
      return chargedHadronEnergy() / ((jecSetsAvailable() ? jecFactor(0) : 1.) * energy());
    }
    /// neutralHadronEnergyFraction (relative to uncorrected jet energy)
    float neutralHadronEnergyFraction() const {
      return neutralHadronEnergy() / ((jecSetsAvailable() ? jecFactor(0) : 1.) * energy());
    }
    /// chargedEmEnergyFraction (relative to uncorrected jet energy)
    float chargedEmEnergyFraction() const {
      return chargedEmEnergy() / ((jecSetsAvailable() ? jecFactor(0) : 1.) * energy());
    }
    /// neutralEmEnergyFraction (relative to uncorrected jet energy)
    float neutralEmEnergyFraction() const {
      return neutralEmEnergy() / ((jecSetsAvailable() ? jecFactor(0) : 1.) * energy());
    }

    // ---- PF Jet specific information ----
    /// photonEnergy
    float photonEnergy() const { return pfSpecific().mPhotonEnergy; }
    /// photonEnergyFraction (relative to corrected jet energy)
    float photonEnergyFraction() const {
      return photonEnergy() / ((jecSetsAvailable() ? jecFactor(0) : 1.) * energy());
    }
    /// electronEnergy
    float electronEnergy() const { return pfSpecific().mElectronEnergy; }
    /// electronEnergyFraction (relative to corrected jet energy)
    float electronEnergyFraction() const {
      return electronEnergy() / ((jecSetsAvailable() ? jecFactor(0) : 1.) * energy());
    }
    /// muonEnergy
    float muonEnergy() const { return pfSpecific().mMuonEnergy; }
    /// muonEnergyFraction (relative to corrected jet energy)
    float muonEnergyFraction() const { return muonEnergy() / ((jecSetsAvailable() ? jecFactor(0) : 1.) * energy()); }
    /// HFHadronEnergy
    float HFHadronEnergy() const { return pfSpecific().mHFHadronEnergy; }
    /// HFHadronEnergyFraction (relative to corrected jet energy)
    float HFHadronEnergyFraction() const {
      return HFHadronEnergy() / ((jecSetsAvailable() ? jecFactor(0) : 1.) * energy());
    }
    /// HFEMEnergy
    float HFEMEnergy() const { return pfSpecific().mHFEMEnergy; }
    /// HFEMEnergyFraction (relative to corrected jet energy)
    float HFEMEnergyFraction() const { return HFEMEnergy() / ((jecSetsAvailable() ? jecFactor(0) : 1.) * energy()); }

    /// chargedHadronMultiplicity
    int chargedHadronMultiplicity() const { return pfSpecific().mChargedHadronMultiplicity; }
    /// neutralHadronMultiplicity
    int neutralHadronMultiplicity() const { return pfSpecific().mNeutralHadronMultiplicity; }
    /// photonMultiplicity
    int photonMultiplicity() const { return pfSpecific().mPhotonMultiplicity; }
    /// electronMultiplicity
    int electronMultiplicity() const { return pfSpecific().mElectronMultiplicity; }

    /// HFHadronMultiplicity
    int HFHadronMultiplicity() const { return pfSpecific().mHFHadronMultiplicity; }
    /// HFEMMultiplicity
    int HFEMMultiplicity() const { return pfSpecific().mHFEMMultiplicity; }

    /// chargedMuEnergy
    float chargedMuEnergy() const { return pfSpecific().mChargedMuEnergy; }
    /// chargedMuEnergyFraction
    float chargedMuEnergyFraction() const {
      return chargedMuEnergy() / ((jecSetsAvailable() ? jecFactor(0) : 1.) * energy());
    }

    /// neutralMultiplicity
    int neutralMultiplicity() const { return pfSpecific().mNeutralMultiplicity; }

    /// hoEnergy
    float hoEnergy() const { return pfSpecific().mHOEnergy; }
    /// hoEnergyFraction (relative to corrected jet energy)
    float hoEnergyFraction() const { return hoEnergy() / ((jecSetsAvailable() ? jecFactor(0) : 1.) * energy()); }
    /// convert generic constituent to specific type

    //  static CaloTowerPtr caloTower (const reco::Candidate* fConstituent);
    /// get specific constituent of the CaloJet.
    /// if the caloTowers were embedded, this reference is transient only and must not be persisted
    reco::PFCandidatePtr getPFConstituent(unsigned fIndex) const;
    /// get the constituents of the CaloJet.
    /// If the caloTowers were embedded, these reference are transient only and must not be persisted
    std::vector<reco::PFCandidatePtr> const& getPFConstituents() const;

    /// get a pointer to a Candididate constituent of the jet
    ///    If using refactorized PAT, return that. (constituents size > 0)
    ///    Else check the old version of PAT (embedded constituents size > 0)
    ///    Else return the reco Jet number of constituents
    const reco::Candidate* daughter(size_t i) const override;

    reco::CandidatePtr daughterPtr(size_t i) const override;
    const reco::CompositePtrCandidate::daughters& daughterPtrVector() const override;

    using reco::LeafCandidate::daughter;  // avoid hiding the base implementation

    /// Return number of daughters:
    ///    If using refactorized PAT, return that. (constituents size > 0)
    ///    Else check the old version of PAT (embedded constituents size > 0)
    ///    Else return the reco Jet number of constituents
    size_t numberOfDaughters() const override;

    /// clear daughter references
    void clearDaughters() override {
      PATObject<reco::Jet>::clearDaughters();
      daughtersTemp_.reset();  // need to reset daughtersTemp_ as well
    }

    /// accessing Jet ID information
    reco::JetID const& jetID() const { return jetID_; }

    /// Access to bare FwdPtr collections
    CaloTowerFwdPtrVector const& caloTowersFwdPtr() const { return caloTowersFwdPtr_; }
    reco::PFCandidateFwdPtrVector const& pfCandidatesFwdPtr() const { return pfCandidatesFwdPtr_; }
    edm::FwdRef<reco::GenJetCollection> const& genJetFwdRef() const { return genJetFwdRef_; }
    TagInfoFwdPtrCollection const& tagInfosFwdPtr() const { return tagInfosFwdPtr_; }

    /// Update bare FwdPtr and FwdRef "forward" pointers while keeping the
    /// "back" pointers the same (i.e. the ref "forwarding")
    void updateFwdCaloTowerFwdPtr(unsigned int index, const edm::Ptr<CaloTower>& updateFwd) {
      if (index < caloTowersFwdPtr_.size()) {
        caloTowersFwdPtr_[index] = CaloTowerFwdPtrVector::value_type(updateFwd, caloTowersFwdPtr_[index].backPtr());
      } else {
        throw cms::Exception("OutOfRange") << "Index " << index << " is out of range" << std::endl;
      }
    }

    void updateFwdPFCandidateFwdPtr(unsigned int index, const edm::Ptr<reco::PFCandidate>& updateFwd) {
      if (index < pfCandidatesFwdPtr_.size()) {
        pfCandidatesFwdPtr_[index] =
            reco::PFCandidateFwdPtrVector::value_type(updateFwd, pfCandidatesFwdPtr_[index].backPtr());
      } else {
        throw cms::Exception("OutOfRange") << "Index " << index << " is out of range" << std::endl;
      }
    }

    void updateFwdTagInfoFwdPtr(unsigned int index, const edm::Ptr<reco::BaseTagInfo>& updateFwd) {
      if (index < tagInfosFwdPtr_.size()) {
        tagInfosFwdPtr_[index] = TagInfoFwdPtrCollection::value_type(updateFwd, tagInfosFwdPtr_[index].backPtr());
      } else {
        throw cms::Exception("OutOfRange") << "Index " << index << " is out of range" << std::endl;
      }
    }

    void updateFwdGenJetFwdRef(edm::Ref<reco::GenJetCollection> updateRef) {
      genJetFwdRef_ = edm::FwdRef<reco::GenJetCollection>(updateRef, genJetFwdRef_.backRef());
    }

    /// pipe operator (introduced to use pat::Jet with PFTopProjectors)
    friend std::ostream& reco::operator<<(std::ostream& out, const pat::Jet& obj);

    /// Access to subjet list
    pat::JetPtrCollection const& subjets(unsigned int index = 0) const;

    /// String access to subjet list
    pat::JetPtrCollection const& subjets(std::string const& label) const;

    /// Add new set of subjets
    void addSubjets(pat::JetPtrCollection const& pieces, std::string const& label = "");

    /// Check to see if the subjet collection exists
    bool hasSubjets(std::string const& label) const {
      return find(subjetLabels_.begin(), subjetLabels_.end(), label) != subjetLabels_.end();
    }

    /// Number of subjet collections
    unsigned int nSubjetCollections() const { return subjetCollections_.size(); }

    /// Subjet collection names
    std::vector<std::string> const& subjetCollectionNames() const { return subjetLabels_; }

    /// Access to mass of subjets
    double groomedMass(unsigned int index = 0) const {
      auto const& sub = subjets(index);
      return nSubjetCollections() > index && !sub.empty()
                 ? std::accumulate(
                       sub.begin(),
                       sub.end(),
                       reco::Candidate::LorentzVector(),
                       [](reco::Candidate::LorentzVector const& a, reco::CandidatePtr const& b) { return a + b->p4(); })
                       .mass()
                 : -1.0;
    }
    double groomedMass(std::string const& label) const {
      auto const& sub = subjets(label);
      return hasSubjets(label) && !sub.empty()
                 ? std::accumulate(
                       sub.begin(),
                       sub.end(),
                       reco::Candidate::LorentzVector(),
                       [](reco::Candidate::LorentzVector const& a, reco::CandidatePtr const& b) { return a + b->p4(); })
                       .mass()
                 : -1.0;
    }

  protected:
    // ---- for content embedding ----

    bool embeddedCaloTowers_;
    edm::AtomicPtrCache<std::vector<CaloTowerPtr> > caloTowersTemp_;  // to simplify user interface
    CaloTowerCollection caloTowers_;                                  // Compatibility embedding
    CaloTowerFwdPtrVector caloTowersFwdPtr_;                          // Refactorized content embedding

    bool embeddedPFCandidates_;
    edm::AtomicPtrCache<std::vector<reco::PFCandidatePtr> > pfCandidatesTemp_;  // to simplify user interface
    reco::PFCandidateCollection pfCandidates_;                                  // Compatibility embedding
    reco::PFCandidateFwdPtrVector pfCandidatesFwdPtr_;                          // Refactorized content embedding

    // ---- Jet Substructure ----
    std::vector<pat::JetPtrCollection> subjetCollections_;
    std::vector<std::string> subjetLabels_;
    edm::AtomicPtrCache<std::vector<reco::CandidatePtr> > daughtersTemp_;

    // ---- MC info ----

    std::vector<reco::GenJet> genJet_;
    reco::GenJetRefVector genJetRef_;
    edm::FwdRef<reco::GenJetCollection> genJetFwdRef_;
    reco::JetFlavourInfo jetFlavourInfo_;

    // ---- energy scale correction factors ----

    // energy scale correction factors; the string carries a potential label if
    // more then one set of correction factors is embedded. The label corresponds
    // to the label of the jetCorrFactors module that has been embedded.
    std::vector<pat::JetCorrFactors> jec_;
    // currently applied set of jet energy correction factors (i.e. the index in
    // jetEnergyCorrections_)
    unsigned int currentJECSet_;
    // currently applied jet energy correction level
    unsigned int currentJECLevel_;
    // currently applied jet energy correction flavor (can be NONE, GLUON, UDS,
    // CHARM or BOTTOM)
    JetCorrFactors::Flavor currentJECFlavor_;

    // ---- b-tag related members ----

    std::vector<std::pair<std::string, float> > pairDiscriVector_;
    std::vector<std::string> tagInfoLabels_;
    edm::OwnVector<reco::BaseTagInfo> tagInfos_;  // Compatibility embedding
    TagInfoFwdPtrCollection tagInfosFwdPtr_;      // Refactorized embedding

    // ---- track related members ----

    float jetCharge_;
    reco::TrackRefVector associatedTracks_;

    // ---- specific members ----

    std::vector<CaloSpecific> specificCalo_;
    std::vector<JPTSpecific> specificJPT_;
    std::vector<PFSpecific> specificPF_;

    // ---- id functions ----
    reco::JetID jetID_;

  private:
    // ---- helper functions ----

    void tryImportSpecific(const reco::Jet& source);

    template <typename T>
    const T* tagInfoByType() const {
      // First check the factorized PAT version
      for (size_t i = 0, n = tagInfosFwdPtr_.size(); i < n; ++i) {
        TagInfoFwdPtrCollection::value_type const& val = tagInfosFwdPtr_[i];
        reco::BaseTagInfo const* baseTagInfo = val.get();
        if (typeid(*baseTagInfo) == typeid(T)) {
          return static_cast<const T*>(baseTagInfo);
        }
      }
      // Then check compatibility version
      for (size_t i = 0, n = tagInfos_.size(); i < n; ++i) {
        edm::OwnVector<reco::BaseTagInfo>::value_type const& val = tagInfos_[i];
        reco::BaseTagInfo const* baseTagInfo = &val;
        if (typeid(*baseTagInfo) == typeid(T)) {
          return static_cast<const T*>(baseTagInfo);
        }
      }
      return nullptr;
    }

    template <typename T>
    const T* tagInfoByTypeOrLabel(const std::string& label = "") const {
      return (label.empty() ? tagInfoByType<T>() : dynamic_cast<const T*>(tagInfo(label)));
    }

    /// return the jet correction factors of a different set, for systematic studies
    const JetCorrFactors* corrFactors_(const std::string& set) const;
    /// return the correction factor for this jet. Throws if they're not available.
    const JetCorrFactors* corrFactors_() const;

    /// cache calo towers
    void cacheCaloTowers() const;
    void cachePFCandidates() const;
    void cacheDaughters() const;
  };
}  // namespace pat

inline float pat::Jet::chargedHadronEnergy() const {
  if (isPFJet()) {
    return pfSpecific().mChargedHadronEnergy;
  } else if (isJPTJet()) {
    return jptSpecific().mChargedHadronEnergy;
  } else {
    throw cms::Exception("Type Mismatch") << "This PAT jet was not made from a JPTJet nor from PFJet.\n";
  }
}

inline float pat::Jet::neutralHadronEnergy() const {
  if (isPFJet()) {
    return pfSpecific().mNeutralHadronEnergy;
  } else {
    throw cms::Exception("Type Mismatch") << "This PAT jet was not made from a JPTJet nor from PFJet.\n";
  }
}

inline float pat::Jet::chargedEmEnergy() const {
  if (isPFJet()) {
    return pfSpecific().mChargedEmEnergy;
  } else if (isJPTJet()) {
    return jptSpecific().mChargedEmEnergy;
  } else {
    throw cms::Exception("Type Mismatch") << "This PAT jet was not made from a JPTJet nor from PFJet.\n";
  }
}

inline float pat::Jet::neutralEmEnergy() const {
  if (isPFJet()) {
    return pfSpecific().mNeutralEmEnergy;
  } else {
    throw cms::Exception("Type Mismatch") << "This PAT jet was not made from a JPTJet nor from PFJet.\n";
  }
}

inline int pat::Jet::muonMultiplicity() const {
  if (isPFJet()) {
    return pfSpecific().mMuonMultiplicity;
  } else if (isJPTJet()) {
    return jptSpecific().muonsInVertexInCalo.size() + jptSpecific().muonsInVertexOutCalo.size();
  } else {
    throw cms::Exception("Type Mismatch") << "This PAT jet was not made from a JPTJet nor from PFJet.\n";
  }
}

inline int pat::Jet::chargedMultiplicity() const {
  if (isPFJet()) {
    return pfSpecific().mChargedMultiplicity;
  } else if (isJPTJet()) {
    return jptSpecific().muonsInVertexInCalo.size() + jptSpecific().muonsInVertexOutCalo.size() +
           jptSpecific().pionsInVertexInCalo.size() + jptSpecific().pionsInVertexOutCalo.size() +
           jptSpecific().elecsInVertexInCalo.size() + jptSpecific().elecsInVertexOutCalo.size();
  } else {
    throw cms::Exception("Type Mismatch") << "This PAT jet was not made from a JPTJet nor from PFJet.\n";
  }
}

#endif
