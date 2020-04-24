import FWCore.ParameterSet.Config as cms
import FWCore.ParameterSet.VarParsing as VarParsing

process = cms.Process("MultiplicityMonitor")

#prepare options

options = VarParsing.VarParsing("analysis")

options.register ('globalTag',
                  "DONOTEXIST",
                  VarParsing.VarParsing.multiplicity.singleton, # singleton or list
                  VarParsing.VarParsing.varType.string,          # string, int, or float
                  "GlobalTag")
options.register ('fromRAW',
                  "0",
                  VarParsing.VarParsing.multiplicity.singleton, # singleton or list
                  VarParsing.VarParsing.varType.int,          # string, int, or float
                  "=1 if from RAW")
options.register ('testTag',
                  "0",
                  VarParsing.VarParsing.multiplicity.singleton, # singleton or list
                  VarParsing.VarParsing.varType.int,          # string, int, or float
                  "=1 if test tag to be used")

options.parseArguments()

#

process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool(True),
    fileMode = cms.untracked.string("FULLMERGE")
    )

process.load("FWCore.MessageService.MessageLogger_cfi")

process.MessageLogger.cout.placeholder = cms.untracked.bool(False)
process.MessageLogger.cout.threshold = cms.untracked.string("INFO")
process.MessageLogger.cout.default = cms.untracked.PSet(
    limit = cms.untracked.int32(10000000)
    )
process.MessageLogger.cout.FwkReport = cms.untracked.PSet(
    reportEvery = cms.untracked.int32(10000)
    )

process.MessageLogger.cerr.placeholder = cms.untracked.bool(False)
process.MessageLogger.cerr.threshold = cms.untracked.string("WARNING")
process.MessageLogger.cerr.default = cms.untracked.PSet(
    limit = cms.untracked.int32(10000000)
    )
process.MessageLogger.cerr.FwkReport = cms.untracked.PSet(
    reportEvery = cms.untracked.int32(100000)
    )

#------------------------------------------------------------------

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(options.maxEvents) )

process.source = cms.Source("PoolSource",
                    fileNames = cms.untracked.vstring(options.inputFiles),
#                    skipBadFiles = cms.untracked.bool(True),
                    inputCommands = cms.untracked.vstring("keep *", "drop *_MEtoEDMConverter_*_*")
                    )

#--------------------------------------
process.seqRECO = cms.Sequence()

if options.fromRAW == 1:
    process.load("Configuration.StandardSequences.RawToDigi_Data_cff")
    process.load("Configuration.StandardSequences.MagneticField_AutoFromDBCurrent_cff")
    process.load("Configuration.StandardSequences.GeometryDB_cff")
    process.load("Configuration.StandardSequences.Reconstruction_cff")
    process.load("Configuration.StandardSequences.L1Reco_cff")
    process.seqRECO = cms.Sequence(process.gtEvmDigis + process.L1Reco
                                   + process.siStripDigis + process.siStripZeroSuppression + process.siStripClusters
                                   + process.siPixelDigis + process.siPixelClusters )


#

process.load("DPGAnalysis.SiStripTools.sipixelclustermultiplicityprod_cfi")
process.load("DPGAnalysis.SiStripTools.sistripclustermultiplicityprod_cfi")

#process.spclustermultprod.wantedSubDets.append = cms.VPSet(    
#    cms.PSet(detSelection = cms.uint32(101),detLabel = cms.string("BPIX"),selection=cms.untracked.vstring("0x1e000000-0x12000000")),
#    cms.PSet(detSelection = cms.uint32(102),detLabel = cms.string("FPIX"),selection=cms.untracked.vstring("0x1e000000-0x14000000"))
#)

#process.ssclustermultprod.wantedSubDets.append = cms.VPSet(    
#    cms.PSet(detSelection = cms.uint32(103),detLabel = cms.string("TIB"),selection=cms.untracked.vstring("0x1e000000-0x16000000")),
#    cms.PSet(detSelection = cms.uint32(105),detLabel = cms.string("TOB"),selection=cms.untracked.vstring("0x1e000000-0x1a000000")),
#    cms.PSet(detSelection = cms.uint32(114),detLabel = cms.string("TIDm"),selection=cms.untracked.vstring("0x1e006000-0x18002000")),
#    cms.PSet(detSelection = cms.uint32(124),detLabel = cms.string("TIDp"),selection=cms.untracked.vstring("0x1e006000-0x18004000")),
#    cms.PSet(detSelection = cms.uint32(116),detLabel = cms.string("TECm"),selection=cms.untracked.vstring("0x1e0c0000-0x1c040000")),
#    cms.PSet(detSelection = cms.uint32(126),detLabel = cms.string("TECp"),selection=cms.untracked.vstring("0x1e0c0000-0x1c080000"))
#)



process.seqMultProd = cms.Sequence(process.spclustermultprod+process.ssclustermultprod)

process.goodVertices = cms.EDFilter("VertexSelector",
   src = cms.InputTag("offlinePrimaryVertices"),
   cut = cms.string("!isFake && ndof > 4 && abs(z) <= 24 && position.Rho <= 2"),  
   filter = cms.bool(False),   # otherwise it won't filter the events, just produce an empty vertex collection.
)



process.seqProducers = cms.Sequence(process.seqRECO + process.seqMultProd)
                                    #+process.goodVertices)

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import *
process.hltSelection = triggerResultsFilter.clone(
                                          triggerConditions = cms.vstring("HLT_ZeroBias_*"),
                                          hltResults = cms.InputTag( "TriggerResults", "", "HLT" ),
                                          l1tResults = cms.InputTag( "" ),
                                          throw = cms.bool(False)
                                          )

process.manypixelclus = cms.EDFilter('BySiPixelClusterVsSiStripClusterMultiplicityEventFilter',
                                     multiplicityConfig = cms.PSet(
    firstMultiplicityConfig = cms.PSet(
    collectionName = cms.InputTag("siPixelClusters"),
    moduleThreshold = cms.untracked.int32(-1),
    useQuality = cms.untracked.bool(False),
    qualityLabel = cms.untracked.string("")
    ),
    secondMultiplicityConfig = cms.PSet(
    collectionName = cms.InputTag("siStripClusters"),
    moduleThreshold = cms.untracked.int32(-1),
    useQuality = cms.untracked.bool(False),
    qualityLabel = cms.untracked.string("")
    )
    ),
                                     cut = cms.string("( mult1 > 5000+0.1*mult2)")
                                     )

process.load("DPGAnalysis.SiStripTools.ssclusmultinvestigator_cfi")

process.ssclusmultinvestigator.scaleFactor=cms.untracked.int32(1)

process.ssclusmultinvestigator.runHisto = cms.untracked.bool(True)
#process.ssclusmultinvestigator.wantedSubDets.append = cms.untracked.VPSet(    
#    cms.PSet(detSelection = cms.uint32(103),detLabel = cms.string("TIB"), binMax = cms.int32(1787904/32)),
#    cms.PSet(detSelection = cms.uint32(105),detLabel = cms.string("TOB"), binMax = cms.int32(1787904/32)),
#    cms.PSet(detSelection = cms.uint32(114),detLabel = cms.string("TIDm"), binMax = cms.int32(565248/32)),
#    cms.PSet(detSelection = cms.uint32(124),detLabel = cms.string("TIDp"), binMax = cms.int32(565248/32)),
#    cms.PSet(detSelection = cms.uint32(116),detLabel = cms.string("TECm"), binMax = cms.int32(3866624/64)),
#    cms.PSet(detSelection = cms.uint32(126),detLabel = cms.string("TECp"), binMax = cms.int32(3866624/64))
#    )

process.load("DPGAnalysis.SiStripTools.spclusmultinvestigator_cfi")
process.spclusmultinvestigator.scaleFactor=cms.untracked.int32(5)

#process.spclusmultinvestigator.wantedSubDets.append = cms.untracked.VPSet(    
#    cms.PSet(detSelection = cms.uint32(101),detLabel = cms.string("BPIX"), binMax = cms.int32(100000)),
#    cms.PSet(detSelection = cms.uint32(102),detLabel = cms.string("FPIX"), binMax = cms.int32(100000))
#    )

process.load("DPGAnalysis.SiStripTools.multiplicitycorr_cfi")
process.multiplicitycorr.correlationConfigurations = cms.VPSet(
   cms.PSet(xMultiplicityMap = cms.InputTag("spclustermultprod"),
            xDetSelection = cms.uint32(0), xDetLabel = cms.string("Pixel"), xBins = cms.uint32(2000), xMax=cms.double(50000), 
            yMultiplicityMap = cms.InputTag("ssclustermultprod"),
            yDetSelection = cms.uint32(0), yDetLabel = cms.string("Tracker"), yBins = cms.uint32(2000), yMax=cms.double(500000),
            rBins = cms.uint32(200), scaleFactor = cms.untracked.double(.25),
            runHisto=cms.bool(False),runHistoBXProfile=cms.bool(False),runHistoBX=cms.bool(False),runHisto2D=cms.bool(False)),
   cms.PSet(xMultiplicityMap = cms.InputTag("spclustermultprod"),
            xDetSelection = cms.uint32(0), xDetLabel = cms.string("Pixel"), xBins = cms.uint32(2000), xMax=cms.double(50000), 
            yMultiplicityMap = cms.InputTag("ssclustermultprod"),
            yDetSelection = cms.uint32(3), yDetLabel = cms.string("TIB"), yBins = cms.uint32(2000), yMax=cms.double(500000),
            rBins = cms.uint32(200), scaleFactor = cms.untracked.double(.25),
            runHisto=cms.bool(False),runHistoBXProfile=cms.bool(False),runHistoBX=cms.bool(False),runHisto2D=cms.bool(False)),
   cms.PSet(xMultiplicityMap = cms.InputTag("spclustermultprod"),
            xDetSelection = cms.uint32(0), xDetLabel = cms.string("Pixel"), xBins = cms.uint32(2000), xMax=cms.double(50000), 
            yMultiplicityMap = cms.InputTag("ssclustermultprod"),
            yDetSelection = cms.uint32(5), yDetLabel = cms.string("TOB"), yBins = cms.uint32(2000), yMax=cms.double(500000),
            rBins = cms.uint32(200), scaleFactor = cms.untracked.double(.25),
            runHisto=cms.bool(False),runHistoBXProfile=cms.bool(False),runHistoBX=cms.bool(False),runHisto2D=cms.bool(False)),
   )

process.multcorrmanypixel = process.multiplicitycorr.clone()

process.seqClusMultInvest = cms.Sequence(process.ssclusmultinvestigator +process.spclusmultinvestigator + process.multiplicitycorr)




process.p0 = cms.Path(
    #    process.hltSelection +
    process.seqProducers +
    process.seqClusMultInvest + 
    process.manypixelclus +
    process.multcorrmanypixel)

#----GlobalTag ------------------------

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, options.globalTag, '')


process.TFileService = cms.Service('TFileService',
                                   fileName = cms.string('MultiplicityMonitor.root')
                                   )

print process.dumpPython()
