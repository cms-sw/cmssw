import FWCore.ParameterSet.Config as cms

#
# producer for hltSingleEMHcalDoubleConeFilter
#
hltSingleEMHcalDoubleConeFilter = cms.EDFilter("HLTEgammaHcalDBCFilter",
    doIsolated = cms.bool(False),
    nonIsoTag = cms.InputTag("l1NonIsoEMHcalDoubleCone"),
    hcalisolbarrelcut = cms.double(8.0),
    hcalisolendcapcut = cms.double(8.0),
    ncandcut = cms.int32(1),
    isoTag = cms.InputTag("hltHcalDoubleCone"),
    candTag = cms.InputTag("hltL1NonIsoSingleEMHighEtHOEFilter")
)


