import FWCore.ParameterSet.Config as cms

# HLT photon trigger
import HLTrigger.HLTfilters.hltHighLevel_cfi
hltPhotonHI = HLTrigger.HLTfilters.hltHighLevel_cfi.hltHighLevel.clone()
hltPhotonHI.HLTPaths = ["HLT_HIPhoton15"]
hltPhotonHI.throw = False
hltPhotonHI.andOr = True

# photon selection
goodPhotons = cms.EDFilter("PhotonSelector",
    src = cms.InputTag("photons"),
    cut = cms.string('et > 20 && hadronicOverEm < 0.1 && r9 > 0.8')
)

# select pairs around Z mass
photonCombiner = cms.EDFilter("CandViewShallowCloneCombiner",
  checkCharge = cms.bool(False),
  cut = cms.string('60 < mass < 120'),
  decay = cms.string('goodPhotons goodPhotons')
)

photonPairCounter = cms.EDFilter("CandViewCountFilter",
  src = cms.InputTag("photonCombiner"),
  minNumber = cms.uint32(1)
)

# Z->ee skim sequence
zEESkimSequence = cms.Sequence(
    hltPhotonHI *
    goodPhotons *
    photonCombiner *
    photonPairCounter
    )
