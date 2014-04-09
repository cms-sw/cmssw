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

process.source = cms.Source("PoolSource",
                    fileNames = cms.untracked.vstring(options.inputFiles),
#                    skipBadFiles = cms.untracked.bool(True),
                    inputCommands = cms.untracked.vstring("keep *", "drop *_MEtoEDMConverter_*_*")
                    )

#--------------------------------------

process.load("DPGAnalysis.SiStripTools.sipixelclustermultiplicityprod_cfi")
process.load("DPGAnalysis.SiStripTools.sistripclustermultiplicityprod_cfi")
process.load("DPGAnalysis.SiStripTools.clustersummarymultiplicityprod_cfi")
process.seqMultProd = cms.Sequence(process.spclustermultprod+process.ssclustermultprod + process.clustsummmultprod)
#process.seqMultProd = cms.Sequence(process.clustsummmultprod)

process.load("DPGAnalysis.SiStripTools.multiplicitycorr_cfi")
process.multiplicitycorr.correlationConfigurations = cms.VPSet(
   cms.PSet(xMultiplicityMap = cms.InputTag("ssclustermultprod"),
            xDetSelection = cms.uint32(0), xDetLabel = cms.string("TK"), xBins = cms.uint32(3000), xMax=cms.double(100000), 
            yMultiplicityMap = cms.InputTag("spclustermultprod"),
            yDetSelection = cms.uint32(0), yDetLabel = cms.string("Pixel"), yBins = cms.uint32(1000), yMax=cms.double(30000),
            rBins = cms.uint32(200), scaleFactor = cms.untracked.double(5.),
            runHisto=cms.bool(False),runHistoBXProfile=cms.bool(False),runHistoBX=cms.bool(False),runHisto2D=cms.bool(False))
   )

process.clustsummmultiplicitycorr = process.multiplicitycorr.clone()
process.clustsummmultiplicitycorr.correlationConfigurations = cms.VPSet(
   cms.PSet(xMultiplicityMap = cms.InputTag("clustsummmultprod"),
            xDetSelection = cms.uint32(0), xDetLabel = cms.string("TK"), xBins = cms.uint32(3000), xMax=cms.double(100000), 
            yMultiplicityMap = cms.InputTag("clustsummmultprod"),
            yDetSelection = cms.uint32(1005), yDetLabel = cms.string("Pixel"), yBins = cms.uint32(1000), yMax=cms.double(30000),
            rBins = cms.uint32(200), scaleFactor = cms.untracked.double(5.),
            runHisto=cms.bool(False),runHistoBXProfile=cms.bool(False),runHistoBX=cms.bool(False),runHisto2D=cms.bool(False))
   )

process.multiplicitycorrtest1 = process.multiplicitycorr.clone()
process.multiplicitycorrtest2 = process.multiplicitycorr.clone()
process.multiplicitycorrtest1not = process.multiplicitycorr.clone()
process.multiplicitycorrtest2not = process.multiplicitycorr.clone()
process.clustsummmultiplicitycorrtest1 = process.multiplicitycorr.clone()
process.clustsummmultiplicitycorrtest2 = process.multiplicitycorr.clone()
process.clustsummmultiplicitycorrtest1not = process.multiplicitycorr.clone()
process.clustsummmultiplicitycorrtest2not = process.multiplicitycorr.clone()
process.multiplicitycorrstripconsistencytest1 = process.multiplicitycorr.clone()
process.multiplicitycorrstripconsistencytest2 = process.multiplicitycorr.clone()
process.multiplicitycorrpixelconsistencytest1 = process.multiplicitycorr.clone()
process.multiplicitycorrpixelconsistencytest2 = process.multiplicitycorr.clone()
process.multiplicitycorrstripconsistencytestnew1 = process.multiplicitycorr.clone()
process.multiplicitycorrstripconsistencytestnew2 = process.multiplicitycorr.clone()
process.multiplicitycorrpixelconsistencytestnew1 = process.multiplicitycorr.clone()
process.multiplicitycorrpixelconsistencytestnew2 = process.multiplicitycorr.clone()
process.multiplicitycorrpixelstripconsistencytest1 = process.multiplicitycorr.clone()
process.multiplicitycorrpixelstripconsistencytestnot1 = process.multiplicitycorr.clone()
process.multiplicitycorrpixelstripconsistencytest2 = process.multiplicitycorr.clone()
process.multiplicitycorrpixelstripconsistencytestnot2 = process.multiplicitycorr.clone()

process.seqClusMultInvest = cms.Sequence(process.multiplicitycorr + process.clustsummmultiplicitycorr) 
#--------------------------------------------------------------------

process.load("DPGAnalysis.SiStripTools.largesipixelclusterevents_cfi")
process.largeSiPixelClusterEvents.absoluteThreshold = 3000
process.largeSiPixelClusterEvents.moduleThreshold = -1

process.load("DPGAnalysis.SiStripTools.largesistripclusterevents_cfi")
process.largeSiStripClusterEvents.absoluteThreshold = 20000
process.largeSiStripClusterEvents.moduleThreshold = -1

process.load("DPGAnalysis.SiStripTools.bysipixelclustmulteventfilter_cfi")
process.bysipixelclustmulteventfilter.multiplicityConfig.moduleThreshold = -1
process.bysipixelclustmulteventfilter.cut = cms.string("mult > 3000")

process.load("DPGAnalysis.SiStripTools.bysistripclustmulteventfilter_cfi")
process.bysistripclustmulteventfilter.multiplicityConfig.moduleThreshold = -1
process.bysistripclustmulteventfilter.cut = cms.string("mult > 20000")

process.load("DPGAnalysis.SiStripTools.byclustsummsipixelmulteventfilter_cfi")
process.byclustsummsipixelmulteventfilter.cut = cms.string("mult > 3000")

process.load("DPGAnalysis.SiStripTools.byclustsummsistripmulteventfilter_cfi")
process.byclustsummsistripmulteventfilter.cut = cms.string("mult > 20000")

process.stripfiltertest1 = cms.Sequence(process.largeSiStripClusterEvents + ~process.bysistripclustmulteventfilter)
process.stripfiltertest2 = cms.Sequence(~process.largeSiStripClusterEvents + process.bysistripclustmulteventfilter)

process.pixelfiltertest1 = cms.Sequence(process.largeSiPixelClusterEvents + ~process.bysipixelclustmulteventfilter)
process.pixelfiltertest2 = cms.Sequence(~process.largeSiPixelClusterEvents + process.bysipixelclustmulteventfilter)

process.stripfiltertestnew1 = cms.Sequence(process.byclustsummsistripmulteventfilter + ~process.bysistripclustmulteventfilter)
process.stripfiltertestnew2 = cms.Sequence(~process.byclustsummsistripmulteventfilter + process.bysistripclustmulteventfilter)

process.pixelfiltertestnew1 = cms.Sequence(process.byclustsummsipixelmulteventfilter + ~process.bysipixelclustmulteventfilter)
process.pixelfiltertestnew2 = cms.Sequence(~process.byclustsummsipixelmulteventfilter + process.bysipixelclustmulteventfilter)

process.load("DPGAnalysis.SiStripTools.bysipixelvssistripclustmulteventfilter_cfi")
process.pixelvsstripfilter1 = process.bysipixelvssistripclustmulteventfilter.clone(cut=cms.string("(mult2 > 10000) && ( mult2 > 2000+7*mult1)"))
process.pixelvsstripfilter2 = process.bysipixelvssistripclustmulteventfilter.clone(cut=cms.string("(mult1 > 1000) && (mult2 <30000) && ( mult2 < -2000+7*mult1)"))

process.load("DPGAnalysis.SiStripTools.byclustsummsipixelvssistripmulteventfilter_cfi")
process.clustsummpixelvsstripfilter1 = process.byclustsummsipixelvssistripmulteventfilter.clone(cut=cms.string("(mult2 > 10000) && ( mult2 > 2000+7*mult1)"))
process.clustsummpixelvsstripfilter2 = process.byclustsummsipixelvssistripmulteventfilter.clone(cut=cms.string("(mult1 > 1000) && (mult2 <30000) && ( mult2 < -2000+7*mult1)"))
                                                                                 
process.pixelvsstripfiltertest1 = cms.Sequence(process.clustsummpixelvsstripfilter1 + ~process.pixelvsstripfilter1)
process.pixelvsstripfiltertestnot1 = cms.Sequence(~process.clustsummpixelvsstripfilter1 + process.pixelvsstripfilter1)
process.pixelvsstripfiltertest2 = cms.Sequence(process.clustsummpixelvsstripfilter2 + ~process.pixelvsstripfilter2)
process.pixelvsstripfiltertestnot2 = cms.Sequence(~process.clustsummpixelvsstripfilter2 + process.pixelvsstripfilter2)



#-------------------------------------------------------------------------------------------

process.seqProducers = cms.Sequence(process.seqMultProd)

process.pstripfiltertest1 = cms.Path(process.seqProducers + process.stripfiltertest1 + process.multiplicitycorrstripconsistencytest1)
process.pstripfiltertest2 = cms.Path(process.seqProducers + process.stripfiltertest2 + process.multiplicitycorrstripconsistencytest2)
process.ppixelfiltertest1 = cms.Path(process.seqProducers + process.pixelfiltertest1 + process.multiplicitycorrpixelconsistencytest1)
process.ppixelfiltertest2 = cms.Path(process.seqProducers + process.pixelfiltertest2 + process.multiplicitycorrpixelconsistencytest2)

process.pstripfiltertestnew1 = cms.Path(process.seqProducers + process.stripfiltertestnew1 + process.multiplicitycorrstripconsistencytestnew1)
process.pstripfiltertestnew2 = cms.Path(process.seqProducers + process.stripfiltertestnew2 + process.multiplicitycorrstripconsistencytestnew2)
process.ppixelfiltertestnew1 = cms.Path(process.seqProducers + process.pixelfiltertestnew1 + process.multiplicitycorrpixelconsistencytestnew1)
process.ppixelfiltertestnew2 = cms.Path(process.seqProducers + process.pixelfiltertestnew2 + process.multiplicitycorrpixelconsistencytestnew2)

process.ppixelstripfiltertest1 = cms.Path(process.seqProducers + process.pixelvsstripfiltertest1 + process.multiplicitycorrpixelstripconsistencytest1)
process.ppixelstripfiltertestnot1 = cms.Path(process.seqProducers + process.pixelvsstripfiltertestnot1 + process.multiplicitycorrpixelstripconsistencytestnot1)
process.ppixelstripfiltertest2 = cms.Path(process.seqProducers + process.pixelvsstripfiltertest2 + process.multiplicitycorrpixelstripconsistencytest2)
process.ppixelstripfiltertestnot2 = cms.Path(process.seqProducers + process.pixelvsstripfiltertestnot2 + process.multiplicitycorrpixelstripconsistencytestnot2)

process.p0 = cms.Path(
   process.seqProducers +
   process.seqClusMultInvest 
   ) 
process.pfiltertest1 = cms.Path(process.seqProducers + process.pixelvsstripfilter1 + process.multiplicitycorrtest1)
process.pfiltertest2 = cms.Path(process.seqProducers + process.pixelvsstripfilter2 + process.multiplicitycorrtest2)
process.pfiltertest1not = cms.Path(process.seqProducers + ~process.pixelvsstripfilter1 + process.multiplicitycorrtest1not)
process.pfiltertest2not = cms.Path(process.seqProducers + ~process.pixelvsstripfilter2 + process.multiplicitycorrtest2not)

process.pclustsummfiltertest1 = cms.Path(process.seqProducers + process.clustsummpixelvsstripfilter1 + process.clustsummmultiplicitycorrtest1)
process.pclustsummfiltertest2 = cms.Path(process.seqProducers + process.clustsummpixelvsstripfilter2 + process.clustsummmultiplicitycorrtest2)
process.pclustsummfiltertest1not = cms.Path(process.seqProducers + ~process.clustsummpixelvsstripfilter1 + process.clustsummmultiplicitycorrtest1not)
process.pclustsummfiltertest2not = cms.Path(process.seqProducers + ~process.clustsummpixelvsstripfilter2 + process.clustsummmultiplicitycorrtest2not)

#----GlobalTag ------------------------

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = options.globalTag


process.TFileService = cms.Service('TFileService',
                                   fileName = cms.string('ByMultiplicityFilterTest.root')
                                   )

#print process.dumpPython()
