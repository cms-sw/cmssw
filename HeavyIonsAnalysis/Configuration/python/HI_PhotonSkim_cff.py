import FWCore.ParameterSet.Config as cms

# HLT photon trigger
import HLTrigger.HLTfilters.hltHighLevel_cfi
hltPhotonHI = HLTrigger.HLTfilters.hltHighLevel_cfi.hltHighLevel.clone()
hltPhotonHI.HLTPaths = ["HLT_HISinglePhoton50_Eta1p5_v*"]
hltPhotonHI.throw = False
hltPhotonHI.andOr = True

# photon selection
goodPhotons = cms.EDFilter("PhotonSelector",
    src = cms.InputTag("photons"),
    cut = cms.string('et > 20 && hadronicOverEm < 0.1 && r9 > 0.8 && sigmaIetaIeta > 0.002')
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

