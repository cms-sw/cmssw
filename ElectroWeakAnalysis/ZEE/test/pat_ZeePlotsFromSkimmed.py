import FWCore.ParameterSet.Config as cms

process = cms.Process("ZEEPLOTS")


process.MessageLogger = cms.Service("MessageLogger")

process.options = cms.untracked.PSet(
    Rethrow = cms.untracked.vstring('ProductNotFound')
)


# source
process.source = cms.Source("PoolSource", 
     fileNames = cms.untracked.vstring(
    'file:zeeCandidates.root'
    )
)
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )

## the Z selection that you prefer
from ElectroWeakAnalysis.ZEE.simpleCutBasedSpring10SelectionBlocks_cfi import *

selection_inverse = cms.PSet (
    deta_EB_inv = cms.untracked.bool(True),
    deta_EE_inv = cms.untracked.bool(True)
    )


selection_secondLeg = cms.PSet (
    ## set this to true if you want to switch on diff 2nd leg selection
    useDifferentSecondLegSelection = cms.untracked.bool(False),
    ## preselection criteria are independent of useDifferentSecondLegSelection
    #  set them to False if you don't want them
    useConversionRejection2 = cms.untracked.bool(False),
    useValidFirstPXBHit2 = cms.untracked.bool(False),
    useExpectedMissingHits2 =cms.untracked.bool(False),
    maxNumberOfExpectedMissingHits2 = cms.untracked.int32(1),    
    ##
    usePrecalcID2 = cms.untracked.bool(False),
    usePrecalcIDType2 = cms.untracked.string('simpleEleId95cIso'),
    usePrecalcIDSign2 = cms.untracked.string('='),
    usePrecalcIDValue2 = cms.untracked.double(7),    
    )

####################################################################################
##
## and the plot creator
process.plotter = cms.EDAnalyzer('ZeePlots',
                                 selection_95relIso,
                                 selection_secondLeg,
                                 usePrecalcID = cms.untracked.bool(False),
                                 usePrecalcIDType = cms.untracked.string('simpleEleId95cIso'),
                                 usePrecalcIDSign = cms.untracked.string('='),
                                 usePrecalcIDValue = cms.untracked.double(7),
                                 zeeCollectionTag = cms.untracked.InputTag("zeeFilter","selectedZeeCandidates","PAT")
                                 )






process.p = cms.Path(process.plotter)


