import FWCore.ParameterSet.Config as cms

# AnaSiStripClusters
modAnaSiStripClusters = cms.EDFilter("AnaSiStripClusters",
    oLabelSiStripCluster = cms.untracked.string('siStripClusters'),
    oOutputFileName = cms.untracked.string('AnaSiStripClusters_out.root'),
    oLabelSiStripDigi = cms.untracked.string('modMTCCAmplifyDigis'),
    oProdInstNameDigi = cms.untracked.string(''),
    bMTCCMode = cms.untracked.bool(True),
    oProdInstNameCluster = cms.untracked.string('')
)


