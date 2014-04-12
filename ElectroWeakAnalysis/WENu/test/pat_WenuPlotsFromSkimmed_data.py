import FWCore.ParameterSet.Config as cms

process = cms.Process("WENUPLOTS")


process.MessageLogger = cms.Service(
        "MessageLogger",
            categories = cms.untracked.vstring('info', 'debug','cout')
            )

process.options = cms.untracked.PSet(
    Rethrow = cms.untracked.vstring('ProductNotFound')
)


# source
process.source = cms.Source("PoolSource", 
     fileNames = cms.untracked.vstring(
    'file:wenuCandidates.root'
    )
)
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )


## Load additional processes
process.load("Configuration.StandardSequences.Geometry_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
## global tags:
process.GlobalTag.globaltag = cms.string('GR_R_35X_V8B::All')
process.load("Configuration.StandardSequences.MagneticField_cff")


####################################################################################
##
## the W selection that you prefer
from ElectroWeakAnalysis.WENu.simpleCutBasedSpring10SelectionBlocks_cfi import *

selection_inverse = cms.PSet (
    deta_EB_inv = cms.untracked.bool(True),
    deta_EE_inv = cms.untracked.bool(True)
    )
# dummy selection for debugging
selection_dummy = cms.PSet (
    trackIso_EB = cms.untracked.double(100.),
    ecalIso_EB =  cms.untracked.double(100.),
    hcalIso_EB =  cms.untracked.double(100.),
    sihih_EB =    cms.untracked.double(0.1 ),
    dphi_EB =     cms.untracked.double(0.1 ),
    deta_EB =     cms.untracked.double(0.1 ),
    hoe_EB =      cms.untracked.double(0.1 ),
    cIso_EB =     cms.untracked.double(100.),
    
    trackIso_EE = cms.untracked.double(100.),
    ecalIso_EE =  cms.untracked.double(100.),
    hcalIso_EE =  cms.untracked.double(100.),
    sihih_EE =    cms.untracked.double(0.1 ),
    dphi_EE =     cms.untracked.double(0.1 ),
    deta_EE =     cms.untracked.double(0.1 ),
    hoe_EE =      cms.untracked.double(0.1 ),
    cIso_EE =     cms.untracked.double(100.),
    useConversionRejection = cms.untracked.bool(False),
    useExpectedMissingHits = cms.untracked.bool(False),
    maxNumberOfExpectedMissingHits = cms.untracked.int32(99),
    )


####################################################################################
##
## and the plot creator
process.plotter = cms.EDAnalyzer('WenuPlots',
                                 # selection in use
                                 selection_80relIso,
                                 selection_inverse,
                                 # if usePrecalcID the precalculated ID will be used only
                                 usePrecalcID = cms.untracked.bool(False),
                                 usePrecalcIDType = cms.untracked.string('simpleEleId95cIso'),
                                 usePrecalcIDSign = cms.untracked.string('='),
                                 usePrecalcIDValue = cms.untracked.double(7),
                                 #
                                 wenuCollectionTag = cms.untracked.InputTag(
                                                   "wenuFilter","selectedWenuCandidates","PAT")
                                 )



process.p = cms.Path(process.plotter)


