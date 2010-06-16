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
process.GlobalTag.globaltag = cms.string('START3X_V26A::All')
process.load("Configuration.StandardSequences.MagneticField_cff")


####################################################################################
##
## the W selection that you prefer
from ElectroWeakAnalysis.WENu.simpleCutBasedSpring10SelectionBlocks_cfi import *

selection_inverse = cms.PSet (
    deta_EB_inv = cms.untracked.bool(True),
    deta_EE_inv = cms.untracked.bool(True)
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


