import FWCore.ParameterSet.Config as cms

#
# Egamma specific data, currently empty
#
egammaSkimEventContent = cms.PSet(
    outputCommands = cms.untracked.vstring()
)
#
# Loose Z selection (originally from EWK group) for electron validation
#
egammaLooseZEventSelection = cms.PSet(
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('egammaLooseZCluster', 
            'egammaLooseZTrack')
    )
)
#
# keep only very high pt clusters for control studies
#
egammaVeryHighEtEventSelection = cms.PSet(
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('egammaVeryHighEt')
    )
)
#
# W + EM object or Jet for fake rate studies
#
egammaWPlusEMOrJetEventSelection = cms.PSet(
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('electronFilterWPath', 
            'muonFilterWPath')
    )
)
#
# Z + EM object or Jet for fake rate studies
#
egammaZPlusEMOrJetEventSelection = cms.PSet(
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('electronFilterZPath', 
            'muonFilterZPath')
    )
)

