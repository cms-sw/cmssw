import FWCore.ParameterSet.Config as cms

# parameters for MuonMillepedeeAlgorithm
MuonMillepedeAlgorithm = cms.PSet(
    algoName = cms.string('MuonMillepedeAlgorithm'),
    CollectionFile = cms.string('Resultado.root'), 
    isCollectionJob = cms.bool(False),
    collectionPath = cms.string("./job"),
    collectionNumber = cms.int32(2),
    outputCollName = cms.string("FinalResult.root"),
    ptCut = cms.double(10.0),
    chi2nCut = cms.double(5.0)
)

