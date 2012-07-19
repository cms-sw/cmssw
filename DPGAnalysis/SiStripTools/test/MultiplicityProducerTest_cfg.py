import FWCore.ParameterSet.Config as cms
import FWCore.ParameterSet.VarParsing as VarParsing

process = cms.Process("MultiplicityProducerTest")

#prepare options

options = VarParsing.VarParsing("analysis")

options.register ('globalTag',
                  "DONOTEXIST::All",
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
process.MessageLogger.cout.threshold = cms.untracked.string("WARNING")
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

process.spclustermultprodnew = process.spclustermultprod.clone(wantedSubDets = cms.VPSet(    
    cms.PSet(detSelection = cms.uint32(101),detLabel = cms.string("BPIX"),selection=cms.untracked.vstring("0x1e000000-0x12000000")),
    cms.PSet(detSelection = cms.uint32(102),detLabel = cms.string("FPIX"),selection=cms.untracked.vstring("0x1e000000-0x14000000")),
    cms.PSet(detSelection = cms.uint32(11),detLabel = cms.string("BPIXL1"),selection=cms.untracked.vstring("0x1e0f0000-0x12010000")),
    cms.PSet(detSelection = cms.uint32(12),detLabel = cms.string("BPIXL2"),selection=cms.untracked.vstring("0x1e0f0000-0x12020000")),
    cms.PSet(detSelection = cms.uint32(13),detLabel = cms.string("BPIXL3"),selection=cms.untracked.vstring("0x1e0f0000-0x12030000")),
    cms.PSet(detSelection = cms.uint32(21),detLabel = cms.string("FPIXm"),selection=cms.untracked.vstring("0x1f800000-0x14800000")),
    cms.PSet(detSelection = cms.uint32(22),detLabel = cms.string("FPIXp"),selection=cms.untracked.vstring("0x1f800000-0x15000000"))
    )
)
process.ssclustermultprodnew = process.ssclustermultprod.clone(wantedSubDets = cms.VPSet(    
    cms.PSet(detSelection = cms.uint32(103),detLabel = cms.string("TIB"),selection=cms.untracked.vstring("0x1e000000-0x16000000")),
    cms.PSet(detSelection = cms.uint32(114),detLabel = cms.string("TIDm"),selection=cms.untracked.vstring("0x1e006000-0x18002000")),
    cms.PSet(detSelection = cms.uint32(124),detLabel = cms.string("TIDp"),selection=cms.untracked.vstring("0x1e006000-0x18004000")),
    cms.PSet(detSelection = cms.uint32(116),detLabel = cms.string("TECm"),selection=cms.untracked.vstring("0x1e0c0000-0x1c040000")),
    cms.PSet(detSelection = cms.uint32(126),detLabel = cms.string("TECp"),selection=cms.untracked.vstring("0x1e0c0000-0x1c080000"))
    )
)

process.seqMultProd = cms.Sequence(process.spclustermultprod+process.ssclustermultprod+process.spclustermultprodnew+process.ssclustermultprodnew)

process.goodVertices = cms.EDFilter("VertexSelector",
   src = cms.InputTag("offlinePrimaryVertices"),
   cut = cms.string("!isFake && ndof > 4 && abs(z) <= 24 && position.Rho <= 2"),  
   filter = cms.bool(False),   # otherwise it won't filter the events, just produce an empty vertex collection.
)


process.seqProducers = cms.Sequence(process.seqRECO + process.seqMultProd +process.goodVertices)

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import *
process.hltSelection = triggerResultsFilter.clone(
                                          triggerConditions = cms.vstring("HLT_ZeroBias*"),
                                          hltResults = cms.InputTag( "TriggerResults", "", "HLT" ),
                                          l1tResults = cms.InputTag( "" ),
                                          throw = cms.bool(False)
                                          )


process.load("DPGAnalysis.SiStripTools.ssclusmultinvestigator_cfi")
process.ssclusmultinvestigator.vertexCollection = cms.InputTag("goodVertices")
process.ssclusmultinvestigator.wantInvestHist = cms.bool(True)
process.ssclusmultinvestigator.wantVtxCorrHist = cms.bool(True)
process.ssclusmultinvestigator.digiVtxCorrConfig = cms.PSet(
    wantedSubDets = process.ssclusmultinvestigator.wantedSubDets,
    hitName = cms.untracked.string("cluster"),
    numberOfBins = cms.untracked.int32(100),   
    scaleFactor = cms.untracked.int32(1),
    maxNvtx = cms.untracked.int32(100)
    )
process.ssclusmultinvestigator.wantLumiCorrHist = cms.bool(True)
process.ssclusmultinvestigator.digiLumiCorrConfig = cms.PSet(
    lumiProducer = cms.InputTag("lumiProducer"),
    wantedSubDets = process.ssclusmultinvestigator.wantedSubDets,
    hitName = cms.untracked.string("cluster"),
    numberOfBins = cms.untracked.int32(100),   
    scaleFactor = cms.untracked.int32(1),
    maxLumi = cms.untracked.double(25)
    )


#process.ssclusmultinvestigator.runHisto = cms.untracked.bool(True)
process.ssclusmultinvestigator.scaleFactor=cms.untracked.int32(1)

process.ssclusmultinvestigatornew = process.ssclusmultinvestigator.clone()
process.ssclusmultinvestigatornew.wantedSubDets = cms.untracked.VPSet(    
    cms.PSet(detSelection = cms.uint32(103),detLabel = cms.string("TIB"), binMax = cms.int32(1787904/32)),
    cms.PSet(detSelection = cms.uint32(114),detLabel = cms.string("TIDm"), binMax = cms.int32(565248/32)),
    cms.PSet(detSelection = cms.uint32(124),detLabel = cms.string("TIDp"), binMax = cms.int32(565248/32)),
    cms.PSet(detSelection = cms.uint32(116),detLabel = cms.string("TECm"), binMax = cms.int32(3866624/64)),
    cms.PSet(detSelection = cms.uint32(126),detLabel = cms.string("TECp"), binMax = cms.int32(3866624/64))
    )
process.ssclusmultinvestigatornew.digiVtxCorrConfig.wantedSubDets = cms.untracked.VPSet(    
    cms.PSet(detSelection = cms.uint32(103),detLabel = cms.string("TIB"), binMax = cms.int32(1787904/32)),
    cms.PSet(detSelection = cms.uint32(114),detLabel = cms.string("TIDm"), binMax = cms.int32(565248/32)),
    cms.PSet(detSelection = cms.uint32(124),detLabel = cms.string("TIDp"), binMax = cms.int32(565248/32)),
    cms.PSet(detSelection = cms.uint32(116),detLabel = cms.string("TECm"), binMax = cms.int32(3866624/64)),
    cms.PSet(detSelection = cms.uint32(126),detLabel = cms.string("TECp"), binMax = cms.int32(3866624/64))
    )
process.ssclusmultinvestigatornew.digiLumiCorrConfig.wantedSubDets = cms.untracked.VPSet(    
    cms.PSet(detSelection = cms.uint32(103),detLabel = cms.string("TIB"), binMax = cms.int32(1787904/32)),
    cms.PSet(detSelection = cms.uint32(114),detLabel = cms.string("TIDm"), binMax = cms.int32(565248/32)),
    cms.PSet(detSelection = cms.uint32(124),detLabel = cms.string("TIDp"), binMax = cms.int32(565248/32)),
    cms.PSet(detSelection = cms.uint32(116),detLabel = cms.string("TECm"), binMax = cms.int32(3866624/64)),
    cms.PSet(detSelection = cms.uint32(126),detLabel = cms.string("TECp"), binMax = cms.int32(3866624/64))
    )
process.ssclusmultinvestigatornew.multiplicityMap = cms.InputTag("ssclustermultprodnew")


process.load("DPGAnalysis.SiStripTools.spclusmultinvestigator_cfi")
process.spclusmultinvestigator.vertexCollection = cms.InputTag("goodVertices")
process.spclusmultinvestigator.wantInvestHist = cms.bool(True)
process.spclusmultinvestigator.wantVtxCorrHist = cms.bool(True)
process.spclusmultinvestigator.digiVtxCorrConfig = cms.PSet(
    wantedSubDets = process.spclusmultinvestigator.wantedSubDets,
    hitName = cms.untracked.string("cluster"),
    numberOfBins = cms.untracked.int32(100),   
    scaleFactor = cms.untracked.int32(5)
    )
process.spclusmultinvestigator.wantLumiCorrHist = cms.bool(True)
process.spclusmultinvestigator.digiLumiCorrConfig = cms.PSet(
    lumiProducer = cms.InputTag("lumiProducer"),
    wantedSubDets = process.spclusmultinvestigator.wantedSubDets,
    hitName = cms.untracked.string("cluster"),
    numberOfBins = cms.untracked.int32(100),   
    scaleFactor = cms.untracked.int32(5)
    )
#process.spclusmultinvestigator.runHisto = cms.untracked.bool(True)
process.spclusmultinvestigator.scaleFactor=cms.untracked.int32(5)

process.spclusmultinvestigatornew = process.spclusmultinvestigator.clone()
process.spclusmultinvestigatornew.wantedSubDets = cms.untracked.VPSet(    
    cms.PSet(detSelection = cms.uint32(101),detLabel = cms.string("BPIX"), binMax = cms.int32(100000)),
    cms.PSet(detSelection = cms.uint32(11),detLabel = cms.string("BPIXL1"), binMax = cms.int32(100000)),
    cms.PSet(detSelection = cms.uint32(12),detLabel = cms.string("BPIXL2"), binMax = cms.int32(100000)),
    cms.PSet(detSelection = cms.uint32(13),detLabel = cms.string("BPIXL3"), binMax = cms.int32(100000)),
    cms.PSet(detSelection = cms.uint32(102),detLabel = cms.string("FPIX"), binMax = cms.int32(100000)),
    cms.PSet(detSelection = cms.uint32(21),detLabel = cms.string("FPIXm"), binMax = cms.int32(100000)),
    cms.PSet(detSelection = cms.uint32(22),detLabel = cms.string("FPIXp"), binMax = cms.int32(100000))
    )
process.spclusmultinvestigatornew.digiVtxCorrConfig.wantedSubDets = cms.untracked.VPSet(    
    cms.PSet(detSelection = cms.uint32(101),detLabel = cms.string("BPIX"), binMax = cms.int32(100000)),
    cms.PSet(detSelection = cms.uint32(11),detLabel = cms.string("BPIXL1"), binMax = cms.int32(100000)),
    cms.PSet(detSelection = cms.uint32(12),detLabel = cms.string("BPIXL2"), binMax = cms.int32(100000)),
    cms.PSet(detSelection = cms.uint32(13),detLabel = cms.string("BPIXL3"), binMax = cms.int32(100000)),
    cms.PSet(detSelection = cms.uint32(102),detLabel = cms.string("FPIX"), binMax = cms.int32(100000)),
    cms.PSet(detSelection = cms.uint32(21),detLabel = cms.string("FPIXm"), binMax = cms.int32(100000)),
    cms.PSet(detSelection = cms.uint32(22),detLabel = cms.string("FPIXp"), binMax = cms.int32(100000))
    )
process.spclusmultinvestigatornew.digiLumiCorrConfig.wantedSubDets = cms.untracked.VPSet(    
    cms.PSet(detSelection = cms.uint32(101),detLabel = cms.string("BPIX"), binMax = cms.int32(100000)),
    cms.PSet(detSelection = cms.uint32(11),detLabel = cms.string("BPIXL1"), binMax = cms.int32(100000)),
    cms.PSet(detSelection = cms.uint32(12),detLabel = cms.string("BPIXL2"), binMax = cms.int32(100000)),
    cms.PSet(detSelection = cms.uint32(13),detLabel = cms.string("BPIXL3"), binMax = cms.int32(100000)),
    cms.PSet(detSelection = cms.uint32(102),detLabel = cms.string("FPIX"), binMax = cms.int32(100000)),
    cms.PSet(detSelection = cms.uint32(21),detLabel = cms.string("FPIXm"), binMax = cms.int32(100000)),
    cms.PSet(detSelection = cms.uint32(22),detLabel = cms.string("FPIXp"), binMax = cms.int32(100000))
    )
process.spclusmultinvestigatornew.multiplicityMap = cms.InputTag("spclustermultprodnew")

process.load("DPGAnalysis.SiStripTools.multiplicitycorr_cfi")
process.multiplicitycorr.correlationConfigurations = cms.VPSet(
   cms.PSet(xMultiplicityMap = cms.InputTag("spclustermultprodnew"),
            xDetSelection = cms.uint32(12), xDetLabel = cms.string("BPixL2"), xBins = cms.uint32(1000), xMax=cms.double(10000), 
            yMultiplicityMap = cms.InputTag("ssclustermultprodnew"),
            yDetSelection = cms.uint32(103), yDetLabel = cms.string("TIB"), yBins = cms.uint32(1000), yMax=cms.double(50000),
            rBins = cms.uint32(200), scaleFactor = cms.untracked.double(.25),
            runHisto=cms.bool(False),runHistoBXProfile=cms.bool(False),runHistoBX=cms.bool(False),runHisto2D=cms.bool(False)),
   cms.PSet(xMultiplicityMap = cms.InputTag("spclustermultprodnew"),
            xDetSelection = cms.uint32(12), xDetLabel = cms.string("BPixL2"), xBins = cms.uint32(1000), xMax=cms.double(10000), 
            yMultiplicityMap = cms.InputTag("ssclustermultprodnew"),
            yDetSelection = cms.uint32(114), yDetLabel = cms.string("TIDm"), yBins = cms.uint32(1000), yMax=cms.double(20000),
            rBins = cms.uint32(200), scaleFactor = cms.untracked.double(.5),
            runHisto=cms.bool(False),runHistoBXProfile=cms.bool(False),runHistoBX=cms.bool(False),runHisto2D=cms.bool(False)),
   cms.PSet(xMultiplicityMap = cms.InputTag("spclustermultprodnew"),
            xDetSelection = cms.uint32(12), xDetLabel = cms.string("BPixL2"), xBins = cms.uint32(1000), xMax=cms.double(10000), 
            yMultiplicityMap = cms.InputTag("ssclustermultprodnew"),
            yDetSelection = cms.uint32(124), yDetLabel = cms.string("TIDp"), yBins = cms.uint32(1000), yMax=cms.double(20000),
            rBins = cms.uint32(200), scaleFactor = cms.untracked.double(.5),
            runHisto=cms.bool(False),runHistoBXProfile=cms.bool(False),runHistoBX=cms.bool(False),runHisto2D=cms.bool(False)),
   cms.PSet(xMultiplicityMap = cms.InputTag("spclustermultprodnew"),
            xDetSelection = cms.uint32(12), xDetLabel = cms.string("BPixL2"), xBins = cms.uint32(1000), xMax=cms.double(10000), 
            yMultiplicityMap = cms.InputTag("spclustermultprodnew"),
            yDetSelection = cms.uint32(11), yDetLabel = cms.string("BPixL1"), yBins = cms.uint32(1000), yMax=cms.double(10000),
            rBins = cms.uint32(200), scaleFactor = cms.untracked.double(1.),
            runHisto=cms.bool(False),runHistoBXProfile=cms.bool(False),runHistoBX=cms.bool(False),runHisto2D=cms.bool(False))
   )

process.seqClusMultInvest = cms.Sequence(process.spclusmultinvestigator + process.ssclusmultinvestigator +
                                         process.spclusmultinvestigatornew + process.ssclusmultinvestigatornew
                                         + process.multiplicitycorr) 



process.p0 = cms.Path(
    process.hltSelection +
    process.seqProducers +
    process.seqClusMultInvest)

#----GlobalTag ------------------------

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = options.globalTag

if options.testTag==1:
    process.stripConditions = cms.ESSource(
        "PoolDBESSource",
        process.CondDBSetup,
        connect = cms.string('frontier://FrontierProd/CMS_COND_31X_STRIP'),
        toGet = cms.VPSet(
        cms.PSet(
        record = cms.string('SiStripNoisesRcd'), tag = cms.string('SiStripNoise_GR10_v2_offline')
        ),
        ),
        )

    process.es_prefer_strips = cms.ESPrefer("PoolDBESSource","stripConditions")

process.TFileService = cms.Service('TFileService',
                                   fileName = cms.string('MultiplicityProducerTest.root')
                                   )

#print process.dumpPython()
