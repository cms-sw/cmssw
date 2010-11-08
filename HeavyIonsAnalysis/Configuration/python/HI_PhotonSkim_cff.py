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

# ECAL spike cleaning filter ??
# noSpikePhotons = cms.EDFilter()

# leading photon E_T filter
photonFilter = cms.EDFilter("EtMinPhotonCountFilter",
    src = cms.InputTag("goodPhotons"),
    etMin = cms.double(40.0),
    minNumber = cms.uint32(1)
)

# photon skim sequence
photonSkimSequence = cms.Sequence(hltPhotonHI
                                   * goodPhotons
                                   #* noSpikePhotons
                                   * photonFilter
                                  )

# select pairs around Z mass
photonCombiner = cms.EDProducer("CandViewShallowCloneCombiner",
  checkCharge = cms.bool(False),
  cut = cms.string('60 < mass < 120'),
  decay = cms.string('goodPhotons goodPhotons')
)

photonPairCounter = cms.EDFilter("CandViewCountFilter",
  src = cms.InputTag("photonCombiner"),
  minNumber = cms.uint32(1)
)

# Z->ee skim sequence
zEESkimSequence = cms.Sequence(hltPhotonHI
                               * goodPhotons
                               #* noSpikePhotons
                               * photonCombiner
                               * photonPairCounter
                               )

