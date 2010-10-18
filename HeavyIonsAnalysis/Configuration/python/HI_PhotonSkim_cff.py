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
    cut = cms.string('et > 40.0 && hadronicOverEm < 0.1 && r9 > 0.8')
)

# ECAL spike cleaning filter ??
# ecalSpikeFilter = cms.EDFilter()

# leading photon E_T filter
photonFilter = cms.EDFilter("PhotonCountFilter",
    src = cms.InputTag("goodPhotons"),
    minNumber = cms.uint32(1)
)

# photon skim sequence
photonSkimSequence = cms.Sequence(hltPhotonHI
                                  * goodPhotons
                                  # * ecalSpikeFilter
                                  * photonFilter
                                  )
