// Shared utilities for HLTP2GT*Filter plugins.
// Provides:
//  - parseObjectType()           string -> l1t::P2GTCandidate::ObjectType
//  - triggerTypeForP2GT()        ObjectType -> trigger::TriggerObjectType
//  - objectTypeName()            ObjectType -> human-readable string (for logging)
//  - CollectionSpec              per-collection kinematic requirements
//  - PairCuts / TripleCuts       inter-object requirements
//  - makeCollectionDescription() helper to build a sub-PSet description

#ifndef HLTrigger_HLTfilters_HLTP2GTUtilities_h
#define HLTrigger_HLTfilters_HLTP2GTUtilities_h

#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"
#include "DataFormats/HLTReco/interface/TriggerTypeDefs.h"
#include "DataFormats/HLTReco/interface/TriggerTypeDefs.h"
#include "DataFormats/L1Trigger/interface/P2GTCandidate.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/Exception.h"

#include <cmath>
#include <limits>
#include <map>
#include <string>

namespace hltp2gt {

  // ---------------------------------------------------------------------------
  // String -> enum
  // ---------------------------------------------------------------------------
  inline l1t::P2GTCandidate::ObjectType parseObjectType(const std::string& name) {
    using OT = l1t::P2GTCandidate::ObjectType;
    static const std::map<std::string, OT> kTable = {
        {"GCTNonIsoEg", OT::GCTNonIsoEg},
        {"GCTIsoEg", OT::GCTIsoEg},
        {"GCTJets", OT::GCTJets},
        {"GCTTaus", OT::GCTTaus},
        {"GCTHtSum", OT::GCTHtSum},
        {"GCTEtSum", OT::GCTEtSum},
        {"GMTSaPromptMuons", OT::GMTSaPromptMuons},
        {"GMTSaDisplacedMuons", OT::GMTSaDisplacedMuons},
        {"GMTTkMuons", OT::GMTTkMuons},
        {"GMTTopo", OT::GMTTopo},
        {"GTTPromptJets", OT::GTTPromptJets},
        {"GTTDisplacedJets", OT::GTTDisplacedJets},
        {"GTTPhiCandidates", OT::GTTPhiCandidates},
        {"GTTRhoCandidates", OT::GTTRhoCandidates},
        {"GTTBsCandidates", OT::GTTBsCandidates},
        {"GTTHadronicTaus", OT::GTTHadronicTaus},
        {"GTTPromptTracks", OT::GTTPromptTracks},
        {"GTTDisplacedTracks", OT::GTTDisplacedTracks},
        {"GTTPrimaryVert", OT::GTTPrimaryVert},
        {"GTTPromptHtSum", OT::GTTPromptHtSum},
        {"GTTDisplacedHtSum", OT::GTTDisplacedHtSum},
        {"GTTEtSum", OT::GTTEtSum},
        {"CL2JetsSC4", OT::CL2JetsSC4},
        {"CL2JetsSC8", OT::CL2JetsSC8},
        {"CL2Taus", OT::CL2Taus},
        {"CL2Electrons", OT::CL2Electrons},
        {"CL2Photons", OT::CL2Photons},
        {"CL2HtSum", OT::CL2HtSum},
        {"CL2EtSum", OT::CL2EtSum},
    };
    auto it = kTable.find(name);
    if (it == kTable.end())
      throw cms::Exception("Configuration") << "HLTP2GTFilter: unknown P2GT object type \"" << name << "\"";
    return it->second;
  }

  // ---------------------------------------------------------------------------
  // Enum -> TriggerObjectType
  // ---------------------------------------------------------------------------
  inline trigger::TriggerObjectType triggerTypeForP2GT(l1t::P2GTCandidate::ObjectType ot) {
    using OT = l1t::P2GTCandidate::ObjectType;
    switch (ot) {
      case OT::GCTNonIsoEg:
        return trigger::TriggerL1NoIsoEG;
      case OT::GCTIsoEg:
        return trigger::TriggerL1IsoEG;
      case OT::GCTJets:
        return trigger::TriggerL1Jet;
      case OT::GCTTaus:
        return trigger::TriggerL1TauJet;
      case OT::GCTHtSum:
        return trigger::TriggerL1HTT;
      case OT::GCTEtSum:
        return trigger::TriggerL1ETT;
      case OT::GMTSaPromptMuons:
        return trigger::TriggerL1Mu;
      case OT::GMTSaDisplacedMuons:
        return trigger::TriggerL1Mu;
      case OT::GMTTkMuons:
        return trigger::TriggerL1Mu;
      case OT::GMTTopo:
        return trigger::TriggerL1Mu;
      case OT::GTTPromptJets:
        return trigger::TriggerL1Jet;
      case OT::GTTDisplacedJets:
        return trigger::TriggerL1Jet;
      case OT::GTTPhiCandidates:
        return trigger::TriggerL1Jet;
      case OT::GTTRhoCandidates:
        return trigger::TriggerL1Jet;
      case OT::GTTBsCandidates:
        return trigger::TriggerL1Jet;
      case OT::GTTHadronicTaus:
        return trigger::TriggerL1TauJet;
      case OT::GTTPromptTracks:
        return trigger::TriggerTrack;
      case OT::GTTDisplacedTracks:
        return trigger::TriggerTrack;
      case OT::GTTPrimaryVert:
        return trigger::TriggerL1Vertex;
      case OT::GTTPromptHtSum:
        return trigger::TriggerL1HTT;
      case OT::GTTDisplacedHtSum:
        return trigger::TriggerL1HTT;
      case OT::GTTEtSum:
        return trigger::TriggerL1ETT;
      case OT::CL2JetsSC4:
        return trigger::TriggerL1Jet;
      case OT::CL2JetsSC8:
        return trigger::TriggerL1Jet;
      case OT::CL2Taus:
        return trigger::TriggerL1Tau;
      case OT::CL2Electrons:
        return trigger::TriggerL1EG;
      case OT::CL2Photons:
        return trigger::TriggerL1EG;
      case OT::CL2HtSum:
        return trigger::TriggerL1HTT;
      case OT::CL2EtSum:
        return trigger::TriggerL1ETT;
      default:
        return trigger::TriggerCluster;
    }
  }

  // ---------------------------------------------------------------------------
  // Enum -> name string (for LogDebug)
  // ---------------------------------------------------------------------------
  inline const char* objectTypeName(l1t::P2GTCandidate::ObjectType ot) {
    using OT = l1t::P2GTCandidate::ObjectType;
    switch (ot) {
      case OT::GCTNonIsoEg:
        return "GCTNonIsoEg";
      case OT::GCTIsoEg:
        return "GCTIsoEg";
      case OT::GCTJets:
        return "GCTJets";
      case OT::GCTTaus:
        return "GCTTaus";
      case OT::GCTHtSum:
        return "GCTHtSum";
      case OT::GCTEtSum:
        return "GCTEtSum";
      case OT::GMTSaPromptMuons:
        return "GMTSaPromptMuons";
      case OT::GMTSaDisplacedMuons:
        return "GMTSaDisplacedMuons";
      case OT::GMTTkMuons:
        return "GMTTkMuons";
      case OT::GMTTopo:
        return "GMTTopo";
      case OT::GTTPromptJets:
        return "GTTPromptJets";
      case OT::GTTDisplacedJets:
        return "GTTDisplacedJets";
      case OT::GTTPhiCandidates:
        return "GTTPhiCandidates";
      case OT::GTTRhoCandidates:
        return "GTTRhoCandidates";
      case OT::GTTBsCandidates:
        return "GTTBsCandidates";
      case OT::GTTHadronicTaus:
        return "GTTHadronicTaus";
      case OT::GTTPromptTracks:
        return "GTTPromptTracks";
      case OT::GTTDisplacedTracks:
        return "GTTDisplacedTracks";
      case OT::GTTPrimaryVert:
        return "GTTPrimaryVert";
      case OT::GTTPromptHtSum:
        return "GTTPromptHtSum";
      case OT::GTTDisplacedHtSum:
        return "GTTDisplacedHtSum";
      case OT::GTTEtSum:
        return "GTTEtSum";
      case OT::CL2JetsSC4:
        return "CL2JetsSC4";
      case OT::CL2JetsSC8:
        return "CL2JetsSC8";
      case OT::CL2Taus:
        return "CL2Taus";
      case OT::CL2Electrons:
        return "CL2Electrons";
      case OT::CL2Photons:
        return "CL2Photons";
      case OT::CL2HtSum:
        return "CL2HtSum";
      case OT::CL2EtSum:
        return "CL2EtSum";
      default:
        return "Unknown";
    }
  }

  // ---------------------------------------------------------------------------
  // Per-collection kinematic specification
  // ---------------------------------------------------------------------------
  struct CollectionSpec {
    l1t::P2GTCandidate::ObjectType objectType;
    double minPt;
    double maxAbsEta;

    CollectionSpec(const edm::ParameterSet& ps)
        : objectType(parseObjectType(ps.getParameter<std::string>("objectType"))),
          minPt(ps.getParameter<double>("minPt")),
          maxAbsEta(ps.getParameter<double>("maxAbsEta")) {}

    bool accepts(const l1t::P2GTCandidate& c) const {
      return c.objectType() == objectType && c.pt() >= minPt && std::abs(c.eta()) <= maxAbsEta;
    }

    // Same logic as accepts() but emits one edm::LogPrint line per candidate
    // describing exactly which cut failed (or that it passed).  Activated by
    // the debugAccepts parameter on HLTP2GTSingleObjectFilter.
    bool acceptsWithDebug(const l1t::P2GTCandidate& c, const std::string& algoName) const {
      const char* typeName = objectTypeName(objectType);

      if (c.objectType() != objectType) {
        std::cout << "HLTP2GTUtilities"
                  << "[" << algoName << "] REJECT objectType mismatch:"
                  << " candidate=" << objectTypeName(c.objectType()) << " expected=" << typeName << std::endl;
        return false;
      } else {
        std::cout << "HLTP2GTUtilities"
                  << "[" << algoName << "] ACCEPT objectType:"
                  << " candidate=" << objectTypeName(c.objectType()) << " expected=" << typeName << std::endl;
      }

      const double absEta = std::abs(c.eta());
      if (absEta > maxAbsEta) {
        std::cout << "HLTP2GTUtilities"
                  << "[" << algoName << "] REJECT " << typeName << " |eta|=" << absEta << " > maxAbsEta=" << maxAbsEta
                  << std::endl;
        return false;
      } else {
        std::cout << "HLTP2GTUtilities"
                  << "[" << algoName << "] ACCEPT " << typeName << " |eta|=" << absEta << " < maxAbsEta=" << maxAbsEta
                  << std::endl;
      }

      const double value = c.pt();
      const auto& hWpT = c.hwPT();

      if (value < minPt) {
        std::cout << "HLTP2GTUtilities"
                  << "[" << algoName << "] REJECT " << typeName << " pT =" << value << " < minPt=" << minPt
                  << "(hW pT:)" << hWpT << std::endl;
        return false;
      } else {
        std::cout << "HLTP2GTUtilities"
                  << "[" << algoName << "] ACCEPT " << typeName << " pT =" << value << " > minPt=" << minPt
                  << "(hW pT:)" << hWpT << std::endl;
      }

      std::cout << "HLTP2GTUtilities" << "[" << algoName << "] ACCEPT " << typeName << "pT=" << value
                << " >= minPt=" << minPt << std::endl;
      return true;
    }

    bool operator==(const CollectionSpec& o) const {
      return objectType == o.objectType && minPt == o.minPt && maxAbsEta == o.maxAbsEta;
    }

    static void fillDescription(edm::ParameterSetDescription& desc, const std::string& defaultType = "CL2Taus") {
      desc.add<std::string>("objectType", defaultType);
      desc.add<double>("minPt", 0.0);
      desc.add<double>("maxAbsEta", 1e9);
    }
  };

  // ---------------------------------------------------------------------------
  // Inter-object cuts (used by double and triple filters)
  // Negative minDR / minDEta / minDPhi / minInvMass means "no cut".
  // ---------------------------------------------------------------------------
  struct PairCuts {
    double minDR2;  ///< squared, for speed
    double maxDR2;
    double minDEta;      ///< absolute value cut; negative = disabled
    double minDPhi;      ///< absolute value cut; negative = disabled
    double minInvMass2;  ///< squared
    double maxInvMass2;

    PairCuts(const edm::ParameterSet& ps)
        : minDR2(std::pow(ps.getParameter<double>("minDR"), 2)),
          maxDR2(std::pow(ps.getParameter<double>("maxDR"), 2)),
          minDEta(ps.getParameter<double>("minDEta")),
          minDPhi(ps.getParameter<double>("minDPhi")),
          minInvMass2(std::pow(ps.getParameter<double>("minInvMass"), 2)),
          maxInvMass2(std::pow(ps.getParameter<double>("maxInvMass"), 2)) {}

    bool accepts(const l1t::P2GTCandidate& a, const l1t::P2GTCandidate& b) const {
      const double deta = a.eta() - b.eta();
      const double dphi = reco::deltaPhi(a.phi(), b.phi());
      const double dr2 = deta * deta + dphi * dphi;
      if (dr2 < minDR2 || dr2 > maxDR2)
        return false;
      if (minDEta >= 0 && std::abs(deta) < minDEta)
        return false;
      if (minDPhi >= 0 && std::abs(dphi) < minDPhi)
        return false;
      // Invariant mass (massless approximation: m² ≈ 2·pT·pT·(cosh(Δη)−cos(Δφ)))
      const double m2 = 2.0 * a.pt() * b.pt() * (std::cosh(deta) - std::cos(dphi));
      if (m2 < minInvMass2 || m2 > maxInvMass2)
        return false;
      return true;
    }

    static void fillDescription(edm::ParameterSetDescription& desc) {
      desc.add<double>("minDR", 0.0);
      desc.add<double>("maxDR", 1e9);
      desc.add<double>("minDEta", -1.0);  // disabled
      desc.add<double>("minDPhi", -1.0);  // disabled
      desc.add<double>("minInvMass", 0.0);
      desc.add<double>("maxInvMass", 1e9);
    }
  };

  // ---------------------------------------------------------------------------
  // Register a collection tag in the filter product (deduplicating by ProductID)
  // ---------------------------------------------------------------------------
  inline void addCollectionTagOnce(const l1t::P2GTCandidateRef& ref,
                                   edm::Event& iEvent,
                                   trigger::TriggerFilterObjectWithRefs& fp,
                                   edm::InputTag& lastTag) {
    const auto& prov = iEvent.getStableProvenance(ref.id());
    edm::InputTag tag(prov.moduleLabel(), prov.productInstanceName(), prov.processName());
    if (tag.encode() != lastTag.encode()) {
      fp.addCollectionTag(tag);
      lastTag = tag;
    }
  }

}  // namespace hltp2gt

// deltaPhi is in DataFormats/Math
#include "DataFormats/Math/interface/deltaPhi.h"

#endif  // HLTrigger_HLTfilters_HLTP2GTUtilities_h
