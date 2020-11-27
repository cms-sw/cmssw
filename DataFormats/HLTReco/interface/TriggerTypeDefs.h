#ifndef HLTReco_TriggerTypeDefs_h
#define HLTReco_TriggerTypeDefs_h

/** \class trigger::TriggerTypeDefs
 *
 *  Misc. common simple typedefs
 *
 *
 *  \author Martin Grunewald
 *
 */

#include <vector>
#include <cstdint>

namespace trigger {

  typedef uint16_t size_type;
  typedef std::vector<size_type> Keys;

  typedef std::vector<int> Vids;

  enum TriggerObjectType {

    /// enum start value shifted to 81 so as to avoid clashes with PDG codes

    /// L1 - using cases as defined in enum L1GtObject, file:
    /// DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutSetupFwd.h"

    TriggerL1Mu = -81,
    TriggerL1NoIsoEG = -82,  // legacy and stage1
    TriggerL1IsoEG = -83,    // legacy and stage1
    TriggerL1CenJet = -84,   // legacy and stage1
    TriggerL1ForJet = -85,   // legacy and stage1
    TriggerL1TauJet = -86,   // legacy and stage1
    TriggerL1ETM = -87,
    TriggerL1ETT = -88,
    TriggerL1HTT = -89,
    TriggerL1HTM = -90,
    TriggerL1JetCounts = -91,     // legacy and stage1
    TriggerL1HfBitCounts = -92,   // legacy and stage1
    TriggerL1HfRingEtSums = -93,  // legacy and stage1
    TriggerL1TechTrig = -94,
    TriggerL1Castor = -95,
    TriggerL1BPTX = -96,
    TriggerL1GtExternal = -97,
    TriggerL1EG = -98,    // stage2
    TriggerL1Jet = -99,   // stage2
    TriggerL1Tau = -100,  // stage2
    TriggerL1ETMHF = -101,
    TriggerL1Centrality = -102,
    TriggerL1MinBiasHFP0 = -103,
    TriggerL1MinBiasHFM0 = -104,
    TriggerL1MinBiasHFP1 = -105,
    TriggerL1MinBiasHFM1 = -106,
    TriggerL1TotalEtEm = -107,
    TriggerL1MissingHtHF = -108,
    TriggerL1TowerCount = -109,
    TriggerL1AsymEt = -110,
    TriggerL1AsymHt = -111,
    TriggerL1AsymEtHF = -112,
    TriggerL1AsymHtHF = -113,
    /// This has all to be decided for Phase-2. Here is Thiago's proposal.
    TriggerL1TkMu = -114,
    TriggerL1TkEle = -115,
    TriggerL1PFJet = -116,
    TriggerL1PFTau = -117,
    TriggerL1TkEm = -118,  // used for photons
    TriggerL1PFMET = -119,
    TriggerL1PFETT = -120,
    TriggerL1PFHT = -121,
    TriggerL1PFMHT = -122,
    TriggerL1PFTrack = -123,
    TriggerL1Vertex = -124,

    /// HLT
    TriggerPhoton = +81,
    TriggerElectron = +82,
    TriggerMuon = +83,
    TriggerTau = +84,
    TriggerJet = +85,
    TriggerBJet = +86,
    TriggerMET = +87,
    TriggerTET = +88,
    TriggerTHT = +89,
    TriggerMHT = +90,
    TriggerTrack = +91,
    TriggerCluster = +92,
    TriggerMETSig = +93,
    TriggerELongit = +94,
    TriggerMHTSig = +95,
    TriggerHLongit = +96

  };

}  // namespace trigger

#endif
