import FWCore.ParameterSet.Config as cms
from GeneratorInterface.Core.genWeights_cfi import genWeights
from GeneratorInterface.Core.lheWeights_cfi import lheWeights

genWeightsNano = genWeights.clone()
genWeightsNano.weightProductLabels = ["genWeights"]

lheWeightsNano = lheWeights.clone()
lheWeightsNano.weightProductLabels = ["lheWeights"]

genWeightsTable = cms.EDProducer("GenWeightsTableProducer",
    lheWeightPrecision = cms.int32(14),
    lheWeights = cms.VInputTag(["lheWeights", "lheWeightsNano"]),
    genWeights = cms.VInputTag(["genWeights", "genWeightsNano"]),
    genLumiInfoHeader = cms.InputTag("generator"),
    # Warning: you can use a full string, but only the first character is read.                                           
    # Note also that the capitalization is important! For example, 'parton shower'                                        
    # must be lower case and 'PDF' must be capital                                                                        
    weightgroups = cms.vstring(['scale', 'PDF', 'matrix element', 'unknown', 'parton shower']),
    # Max number of groups to store for each type above, -1 ==> store all found                                           
    maxGroupsPerType = cms.vint32([1, 1, -1, 2, -1]),
    # If empty or not specified, no criteria are applied to filter on LHAPDF IDs                                          
    # pdfIds = cms.untracked.vint32([91400, 306000, 260000]),                                                              
    unknownOnlyIfEmpty = cms.vstring(['scale', 'PDF']),
    keepAllPSWeights = cms.bool(False),
    # ignoreGenGroups = cms.untracked.bool(True),
    # nStoreUngroupedGen = cms.untracked.int32(40),
    # ignoreLheGroups = cms.untracked.bool(False),
    # nStoreUngroupedLhe = cms.untracked.int32(100),
)

genWeightsTableTask = cms.Task(lheWeightsNano, genWeightsNano, genWeightsTable)
