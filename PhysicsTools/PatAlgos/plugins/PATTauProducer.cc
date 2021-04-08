#include "PhysicsTools/PatAlgos/plugins/PATTauProducer.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "DataFormats/Common/interface/View.h"
#include "DataFormats/Common/interface/Association.h"
#include "DataFormats/Common/interface/Ref.h"

#include "DataFormats/TauReco/interface/PFTau.h"
#include "DataFormats/TauReco/interface/PFTauDiscriminator.h"
#include "DataFormats/TauReco/interface/TauDiscriminatorContainer.h"
#include "DataFormats/HepMCCandidate/interface/GenParticleFwd.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/JetReco/interface/GenJetCollection.h"

#include "DataFormats/PatCandidates/interface/TauJetCorrFactors.h"
#include "DataFormats/TauReco/interface/PFTauTransverseImpactParameterAssociation.h"
#include "DataFormats/TauReco/interface/PFTauTransverseImpactParameter.h"
#include "DataFormats/PatCandidates/interface/TauPFSpecific.h"
#include "DataFormats/PatCandidates/interface/PackedCandidate.h"

#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include <vector>
#include <memory>

using namespace pat;

PATTauProducer::PATTauProducer(const edm::ParameterSet& iConfig)
    : isolator_(iConfig.exists("userIsolation") ? iConfig.getParameter<edm::ParameterSet>("userIsolation")
                                                : edm::ParameterSet(),
                consumesCollector(),
                false),
      useUserData_(iConfig.exists("userData")),
      posAtECalEntranceComputer_(consumesCollector()) {
  firstOccurence_ = true;
  // initialize the configurables
  baseTauToken_ = consumes<edm::View<reco::BaseTau>>(iConfig.getParameter<edm::InputTag>("tauSource"));
  tauTransverseImpactParameterSrc_ = iConfig.getParameter<edm::InputTag>("tauTransverseImpactParameterSource");
  tauTransverseImpactParameterToken_ = consumes<PFTauTIPAssociationByRef>(tauTransverseImpactParameterSrc_);
  pfTauToken_ = consumes<reco::PFTauCollection>(iConfig.getParameter<edm::InputTag>("tauSource"));
  embedIsolationTracks_ = iConfig.getParameter<bool>("embedIsolationTracks");
  embedLeadTrack_ = iConfig.getParameter<bool>("embedLeadTrack");
  embedSignalTracks_ = iConfig.getParameter<bool>("embedSignalTracks");
  embedLeadPFCand_ = iConfig.getParameter<bool>("embedLeadPFCand");
  embedLeadPFChargedHadrCand_ = iConfig.getParameter<bool>("embedLeadPFChargedHadrCand");
  embedLeadPFNeutralCand_ = iConfig.getParameter<bool>("embedLeadPFNeutralCand");
  embedSignalPFCands_ = iConfig.getParameter<bool>("embedSignalPFCands");
  embedSignalPFChargedHadrCands_ = iConfig.getParameter<bool>("embedSignalPFChargedHadrCands");
  embedSignalPFNeutralHadrCands_ = iConfig.getParameter<bool>("embedSignalPFNeutralHadrCands");
  embedSignalPFGammaCands_ = iConfig.getParameter<bool>("embedSignalPFGammaCands");
  embedIsolationPFCands_ = iConfig.getParameter<bool>("embedIsolationPFCands");
  embedIsolationPFChargedHadrCands_ = iConfig.getParameter<bool>("embedIsolationPFChargedHadrCands");
  embedIsolationPFNeutralHadrCands_ = iConfig.getParameter<bool>("embedIsolationPFNeutralHadrCands");
  embedIsolationPFGammaCands_ = iConfig.getParameter<bool>("embedIsolationPFGammaCands");
  addGenMatch_ = iConfig.getParameter<bool>("addGenMatch");
  if (addGenMatch_) {
    embedGenMatch_ = iConfig.getParameter<bool>("embedGenMatch");
    if (iConfig.existsAs<edm::InputTag>("genParticleMatch")) {
      genMatchTokens_.push_back(consumes<edm::Association<reco::GenParticleCollection>>(
          iConfig.getParameter<edm::InputTag>("genParticleMatch")));
    } else {
      genMatchTokens_ = edm::vector_transform(
          iConfig.getParameter<std::vector<edm::InputTag>>("genParticleMatch"),
          [this](edm::InputTag const& tag) { return consumes<edm::Association<reco::GenParticleCollection>>(tag); });
    }
  }
  addGenJetMatch_ = iConfig.getParameter<bool>("addGenJetMatch");
  if (addGenJetMatch_) {
    embedGenJetMatch_ = iConfig.getParameter<bool>("embedGenJetMatch");
    genJetMatchToken_ =
        consumes<edm::Association<reco::GenJetCollection>>(iConfig.getParameter<edm::InputTag>("genJetMatch"));
  }
  addTauJetCorrFactors_ = iConfig.getParameter<bool>("addTauJetCorrFactors");
  tauJetCorrFactorsTokens_ = edm::vector_transform(
      iConfig.getParameter<std::vector<edm::InputTag>>("tauJetCorrFactorsSource"),
      [this](edm::InputTag const& tag) { return mayConsume<edm::ValueMap<TauJetCorrFactors>>(tag); });
  // tau ID configurables
  addTauID_ = iConfig.getParameter<bool>("addTauID");
  if (addTauID_) {
    // read the different tau ID names
    edm::ParameterSet idps = iConfig.getParameter<edm::ParameterSet>("tauIDSources");
    std::vector<std::string> names = idps.getParameterNamesForType<edm::ParameterSet>();
    std::map<std::string, IDContainerData> idContainerMap;
    for (auto const& name : names) {
      auto const& idp = idps.getParameter<edm::ParameterSet>(name);
      std::string prov_cfg_label = idp.getParameter<std::string>("provenanceConfigLabel");
      std::string prov_ID_label = idp.getParameter<std::string>("idLabel");
      edm::InputTag tag = idp.getParameter<edm::InputTag>("inputTag");
      if (prov_cfg_label.empty()) {
        tauIDSrcs_.push_back(NameTag(name, tag));
      } else {
        if (prov_cfg_label != "rawValues" && prov_cfg_label != "workingPoints" && prov_cfg_label != "IDdefinitions" &&
            prov_cfg_label != "IDWPdefinitions" && prov_cfg_label != "direct_rawValues" &&
            prov_cfg_label != "direct_workingPoints")
          throw cms::Exception("Configuration")
              << "PATTauProducer: Parameter 'provenanceConfigLabel' does only accept 'rawValues', 'workingPoints', "
                 "'IDdefinitions', 'IDWPdefinitions', 'direct_rawValues', 'direct_workingPoints'\n";
        std::map<std::string, IDContainerData>::iterator it;
        it = idContainerMap.insert({tag.label() + tag.instance(), {tag, std::vector<NameWPIdx>()}}).first;
        it->second.second.push_back(NameWPIdx(name, WPIdx(WPCfg(prov_cfg_label, prov_ID_label), -99)));
      }
    }
    // but in any case at least once
    if (tauIDSrcs_.empty() && idContainerMap.empty())
      throw cms::Exception("Configuration") << "PATTauProducer: id addTauID is true, you must specify either:\n"
                                            << "\tPSet tauIDSources = { \n"
                                            << "\t\tInputTag <someName> = <someTag>   // as many as you want \n "
                                            << "\t}\n";

    for (auto const& mapEntry : idContainerMap) {
      tauIDSrcContainers_.push_back(mapEntry.second.second);
      pfTauIDContainerTokens_.push_back(mayConsume<reco::TauDiscriminatorContainer>(mapEntry.second.first));
    }
  }
  pfTauIDTokens_ = edm::vector_transform(
      tauIDSrcs_, [this](NameTag const& tag) { return mayConsume<reco::PFTauDiscriminator>(tag.second); });
  skipMissingTauID_ = iConfig.getParameter<bool>("skipMissingTauID");
  // IsoDeposit configurables
  if (iConfig.exists("isoDeposits")) {
    edm::ParameterSet depconf = iConfig.getParameter<edm::ParameterSet>("isoDeposits");
    if (depconf.exists("tracker"))
      isoDepositLabels_.push_back(std::make_pair(pat::TrackIso, depconf.getParameter<edm::InputTag>("tracker")));
    if (depconf.exists("ecal"))
      isoDepositLabels_.push_back(std::make_pair(pat::EcalIso, depconf.getParameter<edm::InputTag>("ecal")));
    if (depconf.exists("hcal"))
      isoDepositLabels_.push_back(std::make_pair(pat::HcalIso, depconf.getParameter<edm::InputTag>("hcal")));
    if (depconf.exists("pfAllParticles"))
      isoDepositLabels_.push_back(
          std::make_pair(pat::PfAllParticleIso, depconf.getParameter<edm::InputTag>("pfAllParticles")));
    if (depconf.exists("pfChargedHadron"))
      isoDepositLabels_.push_back(
          std::make_pair(pat::PfChargedHadronIso, depconf.getParameter<edm::InputTag>("pfChargedHadron")));
    if (depconf.exists("pfNeutralHadron"))
      isoDepositLabels_.push_back(
          std::make_pair(pat::PfNeutralHadronIso, depconf.getParameter<edm::InputTag>("pfNeutralHadron")));
    if (depconf.exists("pfGamma"))
      isoDepositLabels_.push_back(std::make_pair(pat::PfGammaIso, depconf.getParameter<edm::InputTag>("pfGamma")));

    if (depconf.exists("user")) {
      std::vector<edm::InputTag> userdeps = depconf.getParameter<std::vector<edm::InputTag>>("user");
      std::vector<edm::InputTag>::const_iterator it = userdeps.begin(), ed = userdeps.end();
      int key = UserBaseIso;
      for (; it != ed; ++it, ++key) {
        isoDepositLabels_.push_back(std::make_pair(IsolationKeys(key), *it));
      }
    }
  }
  isoDepositTokens_ =
      edm::vector_transform(isoDepositLabels_, [this](std::pair<IsolationKeys, edm::InputTag> const& label) {
        return consumes<edm::ValueMap<IsoDeposit>>(label.second);
      });
  // Efficiency configurables
  addEfficiencies_ = iConfig.getParameter<bool>("addEfficiencies");
  if (addEfficiencies_) {
    efficiencyLoader_ =
        pat::helper::EfficiencyLoader(iConfig.getParameter<edm::ParameterSet>("efficiencies"), consumesCollector());
  }
  // Resolution configurables
  addResolutions_ = iConfig.getParameter<bool>("addResolutions");
  if (addResolutions_) {
    resolutionLoader_ = pat::helper::KinResolutionsLoader(iConfig.getParameter<edm::ParameterSet>("resolutions"));
  }
  // Check to see if the user wants to add user data
  if (useUserData_) {
    userDataHelper_ = PATUserDataHelper<Tau>(iConfig.getParameter<edm::ParameterSet>("userData"), consumesCollector());
  }
  // produces vector of taus
  produces<std::vector<Tau>>();
}

PATTauProducer::~PATTauProducer() {}

void PATTauProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  // switch off embedding (in unschedules mode)
  if (iEvent.isRealData()) {
    addGenMatch_ = false;
    embedGenMatch_ = false;
    addGenJetMatch_ = false;
  }

  // Get the collection of taus from the event
  edm::Handle<edm::View<reco::BaseTau>> anyTaus;
  try {
    iEvent.getByToken(baseTauToken_, anyTaus);
  } catch (const edm::Exception& e) {
    edm::LogWarning("DataSource") << "WARNING! No Tau collection found. This missing input will not block the job. "
                                     "Instead, an empty tau collection is being be produced.";
    auto patTaus = std::make_unique<std::vector<Tau>>();
    iEvent.put(std::move(patTaus));
    return;
  }

  posAtECalEntranceComputer_.beginEvent(iSetup);

  if (isolator_.enabled())
    isolator_.beginEvent(iEvent, iSetup);

  if (efficiencyLoader_.enabled())
    efficiencyLoader_.newEvent(iEvent);
  if (resolutionLoader_.enabled())
    resolutionLoader_.newEvent(iEvent, iSetup);

  std::vector<edm::Handle<edm::ValueMap<IsoDeposit>>> deposits(isoDepositTokens_.size());
  for (size_t j = 0, nd = deposits.size(); j < nd; ++j) {
    iEvent.getByToken(isoDepositTokens_[j], deposits[j]);
  }

  // prepare the MC matching
  std::vector<edm::Handle<edm::Association<reco::GenParticleCollection>>> genMatches(genMatchTokens_.size());
  if (addGenMatch_) {
    for (size_t j = 0, nd = genMatchTokens_.size(); j < nd; ++j) {
      iEvent.getByToken(genMatchTokens_[j], genMatches[j]);
    }
  }

  edm::Handle<edm::Association<reco::GenJetCollection>> genJetMatch;
  if (addGenJetMatch_)
    iEvent.getByToken(genJetMatchToken_, genJetMatch);

  // read in the jet correction factors ValueMap
  std::vector<edm::ValueMap<TauJetCorrFactors>> tauJetCorrs;
  if (addTauJetCorrFactors_) {
    for (size_t i = 0; i < tauJetCorrFactorsTokens_.size(); ++i) {
      edm::Handle<edm::ValueMap<TauJetCorrFactors>> tauJetCorr;
      iEvent.getByToken(tauJetCorrFactorsTokens_[i], tauJetCorr);
      tauJetCorrs.push_back(*tauJetCorr);
    }
  }

  auto patTaus = std::make_unique<std::vector<Tau>>();

  bool first = true;  // this is introduced to issue warnings only for the first tau-jet
  for (size_t idx = 0, ntaus = anyTaus->size(); idx < ntaus; ++idx) {
    edm::RefToBase<reco::BaseTau> tausRef = anyTaus->refAt(idx);
    edm::Ptr<reco::BaseTau> tausPtr = anyTaus->ptrAt(idx);

    Tau aTau(tausRef);
    if (embedLeadTrack_)
      aTau.embedLeadTrack();
    if (embedSignalTracks_)
      aTau.embedSignalTracks();
    if (embedIsolationTracks_)
      aTau.embedIsolationTracks();
    if (embedLeadPFCand_) {
      if (aTau.isPFTau())
        aTau.embedLeadPFCand();
      else
        edm::LogWarning("Type Error") << "Embedding a PFTau-specific information into a pat::Tau which wasn't made "
                                         "from a reco::PFTau is impossible.\n";
    }
    if (embedLeadPFChargedHadrCand_) {
      if (aTau.isPFTau())
        aTau.embedLeadPFChargedHadrCand();
      else
        edm::LogWarning("Type Error") << "Embedding a PFTau-specific information into a pat::Tau which wasn't made "
                                         "from a reco::PFTau is impossible.\n";
    }
    if (embedLeadPFNeutralCand_) {
      if (aTau.isPFTau())
        aTau.embedLeadPFNeutralCand();
      else
        edm::LogWarning("Type Error") << "Embedding a PFTau-specific information into a pat::Tau which wasn't made "
                                         "from a reco::PFTau is impossible.\n";
    }
    if (embedSignalPFCands_) {
      if (aTau.isPFTau())
        aTau.embedSignalPFCands();
      else
        edm::LogWarning("Type Error") << "Embedding a PFTau-specific information into a pat::Tau which wasn't made "
                                         "from a reco::PFTau is impossible.\n";
    }
    if (embedSignalPFChargedHadrCands_) {
      if (aTau.isPFTau())
        aTau.embedSignalPFChargedHadrCands();
      else
        edm::LogWarning("Type Error") << "Embedding a PFTau-specific information into a pat::Tau which wasn't made "
                                         "from a reco::PFTau is impossible.\n";
    }
    if (embedSignalPFNeutralHadrCands_) {
      if (aTau.isPFTau())
        aTau.embedSignalPFNeutralHadrCands();
      else
        edm::LogWarning("Type Error") << "Embedding a PFTau-specific information into a pat::Tau which wasn't made "
                                         "from a reco::PFTau is impossible.\n";
    }
    if (embedSignalPFGammaCands_) {
      if (aTau.isPFTau())
        aTau.embedSignalPFGammaCands();
      else
        edm::LogWarning("Type Error") << "Embedding a PFTau-specific information into a pat::Tau which wasn't made "
                                         "from a reco::PFTau is impossible.\n";
    }
    if (embedIsolationPFCands_) {
      if (aTau.isPFTau())
        aTau.embedIsolationPFCands();
      else
        edm::LogWarning("Type Error") << "Embedding a PFTau-specific information into a pat::Tau which wasn't made "
                                         "from a reco::PFTau is impossible.\n";
    }
    if (embedIsolationPFChargedHadrCands_) {
      if (aTau.isPFTau())
        aTau.embedIsolationPFChargedHadrCands();
      else
        edm::LogWarning("Type Error") << "Embedding a PFTau-specific information into a pat::Tau which wasn't made "
                                         "from a reco::PFTau is impossible.\n";
    }
    if (embedIsolationPFNeutralHadrCands_) {
      if (aTau.isPFTau())
        aTau.embedIsolationPFNeutralHadrCands();
      else
        edm::LogWarning("Type Error") << "Embedding a PFTau-specific information into a pat::Tau which wasn't made "
                                         "from a reco::PFTau is impossible.\n";
    }
    if (embedIsolationPFGammaCands_) {
      if (aTau.isPFTau())
        aTau.embedIsolationPFGammaCands();
      else
        edm::LogWarning("Type Error") << "Embedding a PFTau-specific information into a pat::Tau which wasn't made "
                                         "from a reco::PFTau is impossible.\n";
    }

    if (addTauJetCorrFactors_) {
      // add additional JetCorrs to the jet
      for (unsigned int i = 0; i < tauJetCorrs.size(); ++i) {
        const TauJetCorrFactors& tauJetCorr = tauJetCorrs[i][tausRef];
        // uncomment for debugging
        // tauJetCorr.print();
        aTau.addJECFactors(tauJetCorr);
      }
      std::vector<std::string> levels = tauJetCorrs[0][tausRef].correctionLabels();
      if (std::find(levels.begin(), levels.end(), "L2L3Residual") != levels.end()) {
        aTau.initializeJEC(tauJetCorrs[0][tausRef].jecLevel("L2L3Residual"));
      } else if (std::find(levels.begin(), levels.end(), "L3Absolute") != levels.end()) {
        aTau.initializeJEC(tauJetCorrs[0][tausRef].jecLevel("L3Absolute"));
      } else {
        aTau.initializeJEC(tauJetCorrs[0][tausRef].jecLevel("Uncorrected"));
        if (first) {
          edm::LogWarning("L3Absolute not found")
              << "L2L3Residual and L3Absolute are not part of the correction applied jetCorrFactors \n"
              << "of module " << tauJetCorrs[0][tausRef].jecSet() << " jets will remain"
              << " uncorrected.";
          first = false;
        }
      }
    }

    // store the match to the generated final state muons
    if (addGenMatch_) {
      for (size_t i = 0, n = genMatches.size(); i < n; ++i) {
        reco::GenParticleRef genTau = (*genMatches[i])[tausRef];
        aTau.addGenParticleRef(genTau);
      }
      if (embedGenMatch_)
        aTau.embedGenParticle();
    }

    // store the match to the visible part of the generated tau
    if (addGenJetMatch_) {
      reco::GenJetRef genJetTau = (*genJetMatch)[tausRef];
      if (genJetTau.isNonnull() && genJetTau.isAvailable()) {
        aTau.setGenJet(genJetTau);
      }  // leave empty if no match found
    }

    // prepare ID extraction
    if (addTauID_) {
      size_t numberPlainTauIds = tauIDSrcs_.size();
      size_t numberTauIds = numberPlainTauIds;
      for (auto const& it : tauIDSrcContainers_) {
        numberTauIds += it.size();
      }
      // if ID containers exist, product incices need to be retrieved from provenanceConfigLabel.
      // This is done if config history changes, in particular for the first event.
      if (numberPlainTauIds != numberTauIds && phID_ != iEvent.processHistoryID()) {
        phID_ = iEvent.processHistoryID();
        for (size_t idx = 0; idx < tauIDSrcContainers_.size(); ++idx) {
          auto pfTauIdDiscr = iEvent.getHandle(pfTauIDContainerTokens_[idx]);
          if (!pfTauIdDiscr.isValid())
            continue;  // missing IDs will be skipped lateron or crash there depending on skipMissingTauID_
          const edm::Provenance* prov = pfTauIdDiscr.provenance();
          for (NameWPIdx& idcfg : tauIDSrcContainers_[idx]) {
            std::string prov_cfg_label = idcfg.second.first.first;
            std::string prov_ID_label = idcfg.second.first.second;
            bool found = false;
            if (prov_cfg_label == "rawValues" || prov_cfg_label == "workingPoints") {
              const std::vector<std::string> psetsFromProvenance =
                  edm::parameterSet(prov->stable(), iEvent.processHistory())
                      .getParameter<std::vector<std::string>>(prov_cfg_label);
              for (size_t i = 0; i < psetsFromProvenance.size(); ++i) {
                if (psetsFromProvenance[i] == prov_ID_label) {
                  // using negative indices for raw values
                  if (prov_cfg_label == "rawValues")
                    idcfg.second.second = -1 - i;
                  else
                    idcfg.second.second = i;
                  found = true;
                }
              }
            } else if (prov_cfg_label == "IDdefinitions" || prov_cfg_label == "IDWPdefinitions") {
              const std::vector<edm::ParameterSet> psetsFromProvenance =
                  edm::parameterSet(prov->stable(), iEvent.processHistory())
                      .getParameter<std::vector<edm::ParameterSet>>(prov_cfg_label);
              for (size_t i = 0; i < psetsFromProvenance.size(); ++i) {
                if (psetsFromProvenance[i].getParameter<std::string>("IDname") == prov_ID_label) {
                  // using negative indices for raw values
                  if (prov_cfg_label == "IDdefinitions")
                    idcfg.second.second = -1 - i;
                  else
                    idcfg.second.second = i;
                  found = true;
                }
              }
            } else {
              // checked prov_cfg_label before, so it must be a direct access via indices
              try {
                int i = std::stoi(prov_ID_label);
                if (prov_cfg_label == "direct_rawValues")
                  idcfg.second.second = -1 - i;
                else
                  idcfg.second.second = i;
                found = true;
              } catch (std::invalid_argument const& e) {
                throw cms::Exception("Configuration") << "PATTauProducer: Direct access to ID container requested, so "
                                                         "argument of 'idLabel' must be convertable to int!\n";
              }
            }
            if (!found) {
              throw cms::Exception("Configuration") << "PATTauProducer: Requested working point '" << prov_ID_label
                                                    << "' for ID '" << idcfg.first << "' not found!\n";
            }
          }
        }
      }
      std::string missingDiscriminators;
      std::vector<pat::Tau::IdPair> ids(numberTauIds);
      auto const& tausDeref = *tausRef;
      if (typeid(tausDeref) == typeid(reco::PFTau)) {
        edm::Handle<reco::PFTauCollection> pfTauCollection;
        iEvent.getByToken(pfTauToken_, pfTauCollection);
        for (size_t i = 0; i < numberPlainTauIds; ++i) {
          //std::cout << "filling PFTauDiscriminator '" << tauIDSrcs_[i].first << "' into pat::Tau object..." << std::endl;

          auto pfTauIdDiscr = iEvent.getHandle(pfTauIDTokens_[i]);

          if (skipMissingTauID_ && !pfTauIdDiscr.isValid()) {
            if (!missingDiscriminators.empty()) {
              missingDiscriminators += ", ";
            }
            missingDiscriminators += tauIDSrcs_[i].first;
            continue;
          }
          ids[i].first = tauIDSrcs_[i].first;
          ids[i].second = getTauIdDiscriminator(pfTauCollection, idx, pfTauIdDiscr);
        }
        for (size_t i = 0; i < tauIDSrcContainers_.size(); ++i) {
          auto pfTauIdDiscr = iEvent.getHandle(pfTauIDContainerTokens_[i]);
          if (skipMissingTauID_ && !pfTauIdDiscr.isValid()) {
            for (auto const& it : tauIDSrcContainers_[i]) {
              if (!missingDiscriminators.empty()) {
                missingDiscriminators += ", ";
              }
              missingDiscriminators += it.first;
            }
            continue;
          }
          for (size_t j = 0; j < tauIDSrcContainers_[i].size(); ++j) {
            ids[numberPlainTauIds + j].first = tauIDSrcContainers_[i][j].first;
            ids[numberPlainTauIds + j].second = getTauIdDiscriminatorFromContainer(
                pfTauCollection, idx, pfTauIdDiscr, tauIDSrcContainers_[i][j].second.second);
          }
          numberPlainTauIds += tauIDSrcContainers_[i].size();
        }
      } else {
        throw cms::Exception("Type Mismatch")
            << "PATTauProducer: unsupported datatype '" << typeid(tausDeref).name() << "' for tauSource\n";
      }
      if (!missingDiscriminators.empty() && firstOccurence_) {
        edm::LogWarning("DataSource") << "The following tau discriminators have not been found in the event:\n"
                                      << missingDiscriminators << "\n"
                                      << "They will not be embedded into the pat::Tau object.\n"
                                      << "Note: this message will be printed only at first occurence.";
        firstOccurence_ = false;
      }
      aTau.setTauIDs(ids);
    }

    // extraction of reconstructed tau decay mode
    // (only available for PFTaus)
    if (aTau.isPFTau()) {
      edm::Handle<reco::PFTauCollection> pfTaus;
      iEvent.getByToken(pfTauToken_, pfTaus);
      reco::PFTauRef pfTauRef(pfTaus, idx);

      aTau.setDecayMode(pfTauRef->decayMode());
    }

    // extraction of variables needed to rerun MVA isolation and anti-electron discriminator on MiniAOD
    if (!aTau.pfEssential_.empty()) {
      edm::Handle<reco::PFTauCollection> pfTaus;
      iEvent.getByToken(pfTauToken_, pfTaus);
      reco::PFTauRef pfTauRef(pfTaus, idx);
      pat::tau::TauPFEssential& aTauPFEssential = aTau.pfEssential_[0];
      float ecalEnergy = 0;
      float hcalEnergy = 0;
      float sumPhiTimesEnergy = 0.;
      float sumEtaTimesEnergy = 0.;
      float sumEnergy = 0.;
      float leadChargedCandPt = -99;
      float leadChargedCandEtaAtEcalEntrance = -99;
      const std::vector<reco::CandidatePtr>& signalCands = pfTauRef->signalCands();
      for (const auto& it : signalCands) {
        const reco::PFCandidate* ipfcand = dynamic_cast<const reco::PFCandidate*>(it.get());
        if (ipfcand != nullptr) {
          ecalEnergy += ipfcand->ecalEnergy();
          hcalEnergy += ipfcand->hcalEnergy();
          sumPhiTimesEnergy += ipfcand->positionAtECALEntrance().phi() * ipfcand->energy();
          sumEtaTimesEnergy += ipfcand->positionAtECALEntrance().eta() * ipfcand->energy();
          sumEnergy += ipfcand->energy();
          const reco::Track* track = nullptr;
          if (ipfcand->trackRef().isNonnull())
            track = ipfcand->trackRef().get();
          else if (ipfcand->muonRef().isNonnull() && ipfcand->muonRef()->innerTrack().isNonnull())
            track = ipfcand->muonRef()->innerTrack().get();
          else if (ipfcand->muonRef().isNonnull() && ipfcand->muonRef()->globalTrack().isNonnull())
            track = ipfcand->muonRef()->globalTrack().get();
          else if (ipfcand->muonRef().isNonnull() && ipfcand->muonRef()->outerTrack().isNonnull())
            track = ipfcand->muonRef()->outerTrack().get();
          else if (ipfcand->gsfTrackRef().isNonnull())
            track = ipfcand->gsfTrackRef().get();
          if (track) {
            if (track->pt() > leadChargedCandPt) {
              leadChargedCandEtaAtEcalEntrance = ipfcand->positionAtECALEntrance().eta();
              leadChargedCandPt = track->pt();
            }
          }
        } else {
          // TauReco@MiniAOD: individual ECAL and HCAL energies recovered from fractions,
          // and position at ECAL entrance computed on-the-fly
          const pat::PackedCandidate* ipatcand = dynamic_cast<const pat::PackedCandidate*>(it.get());
          if (ipatcand != nullptr) {
            ecalEnergy += ipatcand->caloFraction() * ipatcand->energy() * (1. - ipatcand->hcalFraction());
            hcalEnergy += ipatcand->caloFraction() * ipatcand->energy() * ipatcand->hcalFraction();
            double posAtECal_phi = ipatcand->phi();
            double posAtECal_eta = ipatcand->eta();
            bool success = false;
            reco::Candidate::Point posAtECalEntrance = posAtECalEntranceComputer_(ipatcand, success);
            if (success) {
              posAtECal_phi = posAtECalEntrance.phi();
              posAtECal_eta = posAtECalEntrance.eta();
            }
            sumPhiTimesEnergy += posAtECal_phi * ipatcand->energy();
            sumEtaTimesEnergy += posAtECal_eta * ipatcand->energy();
            sumEnergy += ipatcand->energy();
            const reco::Track* track = ipatcand->bestTrack();
            if (track != nullptr) {
              if (track->pt() > leadChargedCandPt) {
                leadChargedCandEtaAtEcalEntrance = posAtECal_eta;
                leadChargedCandPt = track->pt();
              }
            }
          }
        }
      }
      aTauPFEssential.ecalEnergy_ = ecalEnergy;
      aTauPFEssential.hcalEnergy_ = hcalEnergy;
      aTauPFEssential.ptLeadChargedCand_ = leadChargedCandPt;
      aTauPFEssential.etaAtEcalEntranceLeadChargedCand_ = leadChargedCandEtaAtEcalEntrance;
      if (sumEnergy != 0.) {
        aTauPFEssential.phiAtEcalEntrance_ = sumPhiTimesEnergy / sumEnergy;
        aTauPFEssential.etaAtEcalEntrance_ = sumEtaTimesEnergy / sumEnergy;
      } else {
        aTauPFEssential.phiAtEcalEntrance_ = -99.;
        aTauPFEssential.etaAtEcalEntrance_ = -99.;
      }
      float leadingTrackNormChi2 = 0;
      float ecalEnergyLeadChargedHadrCand = -99.;
      float hcalEnergyLeadChargedHadrCand = -99.;
      float emFraction = -1.;
      float myHCALenergy = 0.;
      float myECALenergy = 0.;
      const reco::CandidatePtr& leadingPFCharged = pfTauRef->leadChargedHadrCand();
      if (leadingPFCharged.isNonnull()) {
        const reco::PFCandidate* pfCandPtr = dynamic_cast<const reco::PFCandidate*>(leadingPFCharged.get());
        if (pfCandPtr != nullptr) {  // PFTau made from PFCandidates
          ecalEnergyLeadChargedHadrCand = pfCandPtr->ecalEnergy();
          hcalEnergyLeadChargedHadrCand = pfCandPtr->hcalEnergy();
          reco::TrackRef trackRef = pfCandPtr->trackRef();
          if (trackRef.isNonnull()) {
            leadingTrackNormChi2 = trackRef->normalizedChi2();
            for (const auto& isoPFCand : pfTauRef->isolationPFCands()) {
              myHCALenergy += isoPFCand->hcalEnergy();
              myECALenergy += isoPFCand->ecalEnergy();
            }
            for (const auto& signalPFCand : pfTauRef->signalPFCands()) {
              myHCALenergy += signalPFCand->hcalEnergy();
              myECALenergy += signalPFCand->ecalEnergy();
            }
            if (myHCALenergy + myECALenergy != 0.) {
              emFraction = myECALenergy / (myHCALenergy + myECALenergy);
            }
          }
        } else {
          const pat::PackedCandidate* packedCandPtr = dynamic_cast<const pat::PackedCandidate*>(leadingPFCharged.get());
          if (packedCandPtr != nullptr) {
            // TauReco@MiniAOD: individual ECAL and HCAL energies recovered from fractions,
            // and position at ECAL entrance computed on-the-fly
            ecalEnergyLeadChargedHadrCand =
                packedCandPtr->caloFraction() * packedCandPtr->energy() * (1. - packedCandPtr->hcalFraction());
            hcalEnergyLeadChargedHadrCand =
                packedCandPtr->caloFraction() * packedCandPtr->energy() * packedCandPtr->hcalFraction();
            const reco::Track* track = packedCandPtr->bestTrack();
            if (track != nullptr) {
              leadingTrackNormChi2 = track->normalizedChi2();
              for (const auto& isoCand : pfTauRef->isolationCands()) {
                //can safely use static_cast as it is ensured that this PFTau is
                //built with packedCands as its leadingCanidate
                const pat::PackedCandidate* isoPackedCand = static_cast<const pat::PackedCandidate*>(isoCand.get());
                myHCALenergy += isoPackedCand->caloFraction() * isoPackedCand->energy() * isoPackedCand->hcalFraction();
                myECALenergy +=
                    isoPackedCand->caloFraction() * isoPackedCand->energy() * (1. - isoPackedCand->hcalFraction());
              }
              for (const auto& signalCand : pfTauRef->signalCands()) {
                //can safely use static_cast as it is ensured that this PFTau is
                //built with packedCands as its leadingCanidate
                const pat::PackedCandidate* sigPackedCand = static_cast<const pat::PackedCandidate*>(signalCand.get());
                myHCALenergy += sigPackedCand->caloFraction() * sigPackedCand->energy() * sigPackedCand->hcalFraction();
                myECALenergy +=
                    sigPackedCand->caloFraction() * sigPackedCand->energy() * (1. - sigPackedCand->hcalFraction());
              }
              if (myHCALenergy + myECALenergy != 0.) {
                emFraction = myECALenergy / (myHCALenergy + myECALenergy);
              }
            }
          }
        }
      }

      aTauPFEssential.emFraction_ = emFraction;
      aTauPFEssential.leadingTrackNormChi2_ = leadingTrackNormChi2;
      aTauPFEssential.ecalEnergyLeadChargedHadrCand_ = ecalEnergyLeadChargedHadrCand;
      aTauPFEssential.hcalEnergyLeadChargedHadrCand_ = hcalEnergyLeadChargedHadrCand;
      // extraction of tau lifetime information
      if (!tauTransverseImpactParameterSrc_.label().empty()) {
        edm::Handle<PFTauTIPAssociationByRef> tauLifetimeInfos;
        iEvent.getByToken(tauTransverseImpactParameterToken_, tauLifetimeInfos);
        const reco::PFTauTransverseImpactParameter& tauLifetimeInfo = *(*tauLifetimeInfos)[pfTauRef];
        pat::tau::TauPFEssential& aTauPFEssential = aTau.pfEssential_[0];
        aTauPFEssential.dxy_PCA_ = tauLifetimeInfo.dxy_PCA();
        aTauPFEssential.dxy_ = tauLifetimeInfo.dxy();
        aTauPFEssential.dxy_error_ = tauLifetimeInfo.dxy_error();
        aTauPFEssential.hasSV_ = tauLifetimeInfo.hasSecondaryVertex();
        aTauPFEssential.flightLength_ = tauLifetimeInfo.flightLength();
        aTauPFEssential.flightLengthSig_ = tauLifetimeInfo.flightLengthSig();
        aTauPFEssential.ip3d_ = tauLifetimeInfo.ip3d();
        aTauPFEssential.ip3d_error_ = tauLifetimeInfo.ip3d_error();
      }
    }

    // Isolation
    if (isolator_.enabled()) {
      isolator_.fill(*anyTaus, idx, isolatorTmpStorage_);
      typedef pat::helper::MultiIsolator::IsolationValuePairs IsolationValuePairs;
      // better to loop backwards, so the vector is resized less times
      for (IsolationValuePairs::const_reverse_iterator it = isolatorTmpStorage_.rbegin(),
                                                       ed = isolatorTmpStorage_.rend();
           it != ed;
           ++it) {
        aTau.setIsolation(it->first, it->second);
      }
    }

    for (size_t j = 0, nd = deposits.size(); j < nd; ++j) {
      aTau.setIsoDeposit(isoDepositLabels_[j].first, (*deposits[j])[tausRef]);
    }

    if (efficiencyLoader_.enabled()) {
      efficiencyLoader_.setEfficiencies(aTau, tausRef);
    }

    if (resolutionLoader_.enabled()) {
      resolutionLoader_.setResolutions(aTau);
    }

    if (useUserData_) {
      userDataHelper_.add(aTau, iEvent, iSetup);
    }

    patTaus->push_back(aTau);
  }

  // sort taus in pT
  std::sort(patTaus->begin(), patTaus->end(), pTTauComparator_);

  // put genEvt object in Event
  iEvent.put(std::move(patTaus));

  // clean up
  if (isolator_.enabled())
    isolator_.endEvent();
}

template <typename TauCollectionType, typename TauDiscrType>
float PATTauProducer::getTauIdDiscriminator(const edm::Handle<TauCollectionType>& tauCollection,
                                            size_t tauIdx,
                                            const edm::Handle<TauDiscrType>& tauIdDiscr) {
  edm::Ref<TauCollectionType> tauRef(tauCollection, tauIdx);
  return (*tauIdDiscr)[tauRef];
}
float PATTauProducer::getTauIdDiscriminatorFromContainer(const edm::Handle<reco::PFTauCollection>& tauCollection,
                                                         size_t tauIdx,
                                                         const edm::Handle<reco::TauDiscriminatorContainer>& tauIdDiscr,
                                                         int wpIdx) {
  edm::Ref<reco::PFTauCollection> tauRef(tauCollection, tauIdx);
  if (wpIdx < 0) {
    //Only 0th component filled with default value if prediscriminor in RecoTauDiscriminator failed.
    if ((*tauIdDiscr)[tauRef].rawValues.size() == 1)
      return (*tauIdDiscr)[tauRef].rawValues.at(0);
    //uses negative indices to access rawValues. In most cases only one rawValue at wpIdx=-1 exists.
    return (*tauIdDiscr)[tauRef].rawValues.at(-1 - wpIdx);
  } else {
    //WP vector not filled if prediscriminor in RecoTauDiscriminator failed. Set PAT output to false in this case
    if ((*tauIdDiscr)[tauRef].workingPoints.empty())
      return 0.0;
    return (*tauIdDiscr)[tauRef].workingPoints.at(wpIdx);
  }
}

// ParameterSet description for module
void PATTauProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription iDesc;
  iDesc.setComment("PAT tau producer module");

  // input source
  iDesc.add<edm::InputTag>("tauSource", edm::InputTag())->setComment("input collection");

  // embedding
  iDesc.add<bool>("embedIsolationTracks", false)->setComment("embed external isolation tracks");
  iDesc.add<bool>("embedLeadTrack", false)->setComment("embed external leading track");
  iDesc.add<bool>("embedLeadTracks", false)->setComment("embed external signal tracks");

  // MC matching configurables
  iDesc.add<bool>("addGenMatch", true)->setComment("add MC matching");
  iDesc.add<bool>("embedGenMatch", false)->setComment("embed MC matched MC information");
  std::vector<edm::InputTag> emptySourceVector;
  iDesc
      .addNode(edm::ParameterDescription<edm::InputTag>("genParticleMatch", edm::InputTag(), true) xor
               edm::ParameterDescription<std::vector<edm::InputTag>>("genParticleMatch", emptySourceVector, true))
      ->setComment("input with MC match information");

  // MC jet matching variables
  iDesc.add<bool>("addGenJetMatch", true)->setComment("add MC jet matching");
  iDesc.add<bool>("embedGenJetMatch", false)->setComment("embed MC jet matched jet information");
  iDesc.add<edm::InputTag>("genJetMatch", edm::InputTag("tauGenJetMatch"));

  pat::helper::KinResolutionsLoader::fillDescription(iDesc);

  // tau ID configurables
  iDesc.add<bool>("addTauID", true)->setComment("add tau ID variables");
  edm::ParameterSetDescription tauIDSourcesPSet;
  tauIDSourcesPSet.setAllowAnything();
  iDesc
      .addNode(edm::ParameterDescription<edm::InputTag>("tauIDSource", edm::InputTag(), true) xor
               edm::ParameterDescription<edm::ParameterSetDescription>("tauIDSources", tauIDSourcesPSet, true))
      ->setComment("input with tau ID variables");
  // (Dis)allow to skip missing tauId sources
  iDesc.add<bool>("skipMissingTauID", false)
      ->setComment("allow to skip a tau ID variable when not present in the event");

  // IsoDeposit configurables
  edm::ParameterSetDescription isoDepositsPSet;
  isoDepositsPSet.addOptional<edm::InputTag>("tracker");
  isoDepositsPSet.addOptional<edm::InputTag>("ecal");
  isoDepositsPSet.addOptional<edm::InputTag>("hcal");
  isoDepositsPSet.addOptional<edm::InputTag>("pfAllParticles");
  isoDepositsPSet.addOptional<edm::InputTag>("pfChargedHadron");
  isoDepositsPSet.addOptional<edm::InputTag>("pfNeutralHadron");
  isoDepositsPSet.addOptional<edm::InputTag>("pfGamma");
  isoDepositsPSet.addOptional<std::vector<edm::InputTag>>("user");
  iDesc.addOptional("isoDeposits", isoDepositsPSet);

  // Efficiency configurables
  edm::ParameterSetDescription efficienciesPSet;
  efficienciesPSet.setAllowAnything();  // TODO: the pat helper needs to implement a description.
  iDesc.add("efficiencies", efficienciesPSet);
  iDesc.add<bool>("addEfficiencies", false);

  // Check to see if the user wants to add user data
  edm::ParameterSetDescription userDataPSet;
  PATUserDataHelper<Tau>::fillDescription(userDataPSet);
  iDesc.addOptional("userData", userDataPSet);

  edm::ParameterSetDescription isolationPSet;
  isolationPSet.setAllowAnything();  // TODO: the pat helper needs to implement a description.
  iDesc.add("userIsolation", isolationPSet);
}

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(PATTauProducer);
