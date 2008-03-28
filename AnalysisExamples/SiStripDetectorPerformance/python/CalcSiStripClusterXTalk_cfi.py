import FWCore.ParameterSet.Config as cms

# CalcSiStripClusterXTalk
modCalcSiStripClusterXTalk = cms.EDFilter("CalcSiStripClusterXTalk",
    oProdInstName = cms.untracked.string('ZeroSuppressed'),
    oLabelSiStripCluster = cms.untracked.string('siStripClusters'),
    oOutputFileName = cms.untracked.string('CalcSiStripClusterXTalk_out.root'),
    oLabelSiStripDigi = cms.untracked.string('SiStripDigis'),
    oLabelSiStripClusterInfo = cms.untracked.string('siStripClusterInfoProducer')
)


