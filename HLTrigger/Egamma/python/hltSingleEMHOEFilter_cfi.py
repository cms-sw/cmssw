import FWCore.ParameterSet.Config as cms

#
# producer for hltSingleEMHOEFilster
#
hltSinlgeEMHOEFilter = cms.EDFilter("HLTEgammaHOEFilter",
    doIsolated = cms.bool(False),
    nonIsoTag = cms.InputTag("l1NonIsolatedElectronHcalIsol"),
    hcalisolbarrelcut = cms.double(0.05),
    hcalisolendcapcut = cms.double(0.05),
    ncandcut = cms.int32(1),
    isoTag = cms.InputTag("l1IsolatedElectronHcalIsol"),
    candTag = cms.InputTag("hltL1NonIsoSingleEMHighEtEcalIsolFilter")
)


