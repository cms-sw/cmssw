import FWCore.ParameterSet.Config as cms
import FWCore.ParameterSet.VarParsing as VarParsing

process = cms.Process("ByMultiplicityFilterTest")

#prepare options

options = VarParsing.VarParsing("analysis")

options.register ('globalTag',
                  "DONOTEXIST::All",
                  VarParsing.VarParsing.multiplicity.singleton, # singleton or list
                  VarParsing.VarParsing.varType.string,          # string, int, or float
                  "GlobalTag")

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

#----Remove too verbose PrimaryVertexProducer

process.MessageLogger.suppressInfo.append("pixelVerticesAdaptive")
process.MessageLogger.suppressInfo.append("pixelVerticesAdaptiveNoBS")

#----Remove too verbose BeamSpotOnlineProducer

process.MessageLogger.suppressInfo.append("testBeamSpot")
process.MessageLogger.suppressInfo.append("onlineBeamSpot")
process.MessageLogger.suppressWarning.append("testBeamSpot")
process.MessageLogger.suppressWarning.append("onlineBeamSpot")

#----Remove too verbose TrackRefitter

process.MessageLogger.suppressInfo.append("newTracksFromV0")
process.MessageLogger.suppressInfo.append("newTracksFromOtobV0")

#------------------------------------------------------------------

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )

process.load("TrackingPFG.Configuration.poolSource_cff")
process.source.fileNames = cms.untracked.vstring(options.inputFiles)

#--------------------------------------

process.load("DPGAnalysis.SiStripTools.sipixelclustermultiplicityprod_cfi")
process.load("DPGAnalysis.SiStripTools.sistripclustermultiplicityprod_cfi")
process.seqMultProd = cms.Sequence(process.spclustermultprod+process.ssclustermultprod)

process.load("DPGAnalysis.SiStripTools.multiplicitycorr_cfi")
process.multiplicitycorr.correlationConfigurations = cms.VPSet(
   cms.PSet(xMultiplicityMap = cms.InputTag("ssclustermultprod"),
            xDetSelection = cms.uint32(0), xDetLabel = cms.string("TK"), xBins = cms.uint32(3000), xMax=cms.double(100000), 
            yMultiplicityMap = cms.InputTag("spclustermultprod"),
            yDetSelection = cms.uint32(0), yDetLabel = cms.string("Pixel"), yBins = cms.uint32(1000), yMax=cms.double(30000),
            rBins = cms.uint32(200), scaleFactor = cms.untracked.double(5.))
   )

process.multiplicitycorrtest1 = process.multiplicitycorr.clone()
process.multiplicitycorrtest2 = process.multiplicitycorr.clone()
process.multiplicitycorrtest1not = process.multiplicitycorr.clone()
process.multiplicitycorrtest2not = process.multiplicitycorr.clone()
process.multiplicitycorrstripconsistencytest1 = process.multiplicitycorr.clone()
process.multiplicitycorrstripconsistencytest2 = process.multiplicitycorr.clone()
process.multiplicitycorrpixelconsistencytest1 = process.multiplicitycorr.clone()
process.multiplicitycorrpixelconsistencytest2 = process.multiplicitycorr.clone()

process.seqClusMultInvest = cms.Sequence(process.multiplicitycorr) 
#--------------------------------------------------------------------

process.load("DPGAnalysis.SiStripTools.largesipixelclusterevents_cfi")
process.largeSiPixelClusterEvents.absoluteThreshold = 1000
process.largeSiPixelClusterEvents.moduleThreshold = -1

process.load("DPGAnalysis.SiStripTools.largesistripclusterevents_cfi")
process.largeSiStripClusterEvents.absoluteThreshold = 15000
process.largeSiStripClusterEvents.moduleThreshold = -1

process.load("DPGAnalysis.SiStripTools.bysipixelclustmulteventfilter_cfi")
process.bysipixelclustmulteventfilter.multiplicityConfig.moduleThreshold = -1
process.bysipixelclustmulteventfilter.cut = cms.string("mult > 1000")

process.load("DPGAnalysis.SiStripTools.bysistripclustmulteventfilter_cfi")
process.bysistripclustmulteventfilter.multiplicityConfig.moduleThreshold = -1
process.bysistripclustmulteventfilter.cut = cms.string("mult > 15000")

process.stripfiltertest1 = cms.Sequence(process.largeSiStripClusterEvents + ~process.bysistripclustmulteventfilter)
process.stripfiltertest2 = cms.Sequence(~process.largeSiStripClusterEvents + process.bysistripclustmulteventfilter)

process.pixelfiltertest1 = cms.Sequence(process.largeSiPixelClusterEvents + ~process.bysipixelclustmulteventfilter)
process.pixelfiltertest2 = cms.Sequence(~process.largeSiPixelClusterEvents + process.bysipixelclustmulteventfilter)

process.load("DPGAnalysis.SiStripTools.bysipixelvssistripclustmulteventfilter_cfi")
process.pixelvsstripfilter1 = process.bysipixelvssistripclustmulteventfilter.clone(cut=cms.string("(mult2 > 10000) && ( mult2 > 2000+7*mult1)"))
process.pixelvsstripfilter2 = process.bysipixelvssistripclustmulteventfilter.clone(cut=cms.string("(mult1 > 1000) && (mult2 <30000) && ( mult2 < -2000+7*mult1)"))
                                                                                 

#-------------------------------------------------------------------------------------------

process.seqProducers = cms.Sequence(process.seqMultProd)

process.pstripfiltertest1 = cms.Path(process.stripfiltertest1 + process.seqProducers + process.multiplicitycorrstripconsistencytest1)
process.pstripfiltertest2 = cms.Path(process.stripfiltertest2 + process.seqProducers + process.multiplicitycorrstripconsistencytest2)
process.ppixelfiltertest1 = cms.Path(process.pixelfiltertest1 + process.seqProducers + process.multiplicitycorrpixelconsistencytest1)
process.ppixelfiltertest2 = cms.Path(process.pixelfiltertest2 + process.seqProducers + process.multiplicitycorrpixelconsistencytest2)

process.p0 = cms.Path(
   process.seqProducers +
   process.seqClusMultInvest 
   )

process.pfiltertest1 = cms.Path(process.pixelvsstripfilter1 + process.seqProducers + process.multiplicitycorrtest1)
process.pfiltertest2 = cms.Path(process.pixelvsstripfilter2 + process.seqProducers + process.multiplicitycorrtest2)
process.pfiltertest1not = cms.Path(~process.pixelvsstripfilter1 + process.seqProducers + process.multiplicitycorrtest1not)
process.pfiltertest2not = cms.Path(~process.pixelvsstripfilter2 + process.seqProducers + process.multiplicitycorrtest2not)

#----GlobalTag ------------------------

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = options.globalTag


process.TFileService = cms.Service('TFileService',
                                   fileName = cms.string('ByMultiplicityFilterTest.root')
#                                   fileName = cms.string('TrackerLocal_rereco.root')
                                   )

#print process.dumpPython()
