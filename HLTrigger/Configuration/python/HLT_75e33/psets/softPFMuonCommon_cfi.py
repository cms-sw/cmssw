import FWCore.ParameterSet.Config as cms

softPFMuonCommon = cms.PSet(
    gbrForestLabel = cms.string('btag_SoftPFMuon_BDT'),
    useAdaBoost = cms.bool(True),
    useCondDB = cms.bool(True),
    useGBRForest = cms.bool(True),
    weightFile = cms.FileInPath('RecoBTag/SoftLepton/data/SoftPFMuon_BDT.weights.xml.gz')
)