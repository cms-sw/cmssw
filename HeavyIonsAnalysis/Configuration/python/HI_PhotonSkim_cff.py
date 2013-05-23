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

# leading photon E_T filter
photonFilter = cms.EDFilter("EtMinPhotonCountFilter",
    src = cms.InputTag("goodPhotons"),
    etMin = cms.double(40.0),
    minNumber = cms.uint32(1)
)

# supercluster cleaning sequence (output = cleanPhotons)
from RecoHI.HiEgammaAlgos.HiEgamma_cff import *

# photon selection of spike-cleaned collection
goodCleanPhotons = goodPhotons.clone(src=cms.InputTag("cleanPhotons"))

# leading E_T filter on cleaned collection
cleanPhotonFilter = photonFilter.clone(src=cms.InputTag("goodCleanPhotons"))

# photon skim sequence
photonSkimSequence = cms.Sequence(hltPhotonHI
                                  * goodPhotons
                                  * photonFilter
                                  * hiPhotonCleaningSequence
                                  * goodCleanPhotons
                                  * cleanPhotonFilter
                                  )

# two-photon E_T filter
twoPhotonFilter = cms.EDFilter("EtMinPhotonCountFilter",
    src = cms.InputTag("goodPhotons"),
    etMin = cms.double(20.0),
    minNumber = cms.uint32(2)
)

# select pairs around Z mass
photonCombiner = cms.EDProducer("CandViewShallowCloneCombiner",
  checkCharge = cms.bool(False),
  cut = cms.string('60 < mass < 120'),
  decay = cms.string('goodCleanPhotons goodCleanPhotons')
)

photonPairCounter = cms.EDFilter("CandViewCountFilter",
  src = cms.InputTag("photonCombiner"),
  minNumber = cms.uint32(1)
)

# Z->ee skim sequence
zEESkimSequence = cms.Sequence(hltPhotonHI
                               * goodPhotons
                               * twoPhotonFilter
                               * hiPhotonCleaningSequence
                               * goodCleanPhotons
                               * photonCombiner
                               * photonPairCounter
                               )

