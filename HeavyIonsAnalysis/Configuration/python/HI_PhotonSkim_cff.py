import FWCore.ParameterSet.Config as cms

# HLT photon trigger
import HLTrigger.HLTfilters.hltHighLevel_cfi
hltPhotonHI = HLTrigger.HLTfilters.hltHighLevel_cfi.hltHighLevel.clone()
hltPhotonHI.HLTPaths = ["HLT_HIPhoton15"]
hltPhotonHI.throw = False
hltPhotonHI.andOr = True

# leading photon E_T filter
singlePhoton40 = cms.EDFilter("PhotonSelector",
    src = cms.InputTag("photons"),
    cut = cms.string('et > 40.0')
)

# ECAL spike cleaning filter ??
# ecalSpikeFilter = cms.EDFilter()

# photon skim sequence
photonSkimSequence = cms.Sequence(hltPhotonHI
                                  * singlePhoton40
                                  # * ecalSpikeFilter
                                  )
