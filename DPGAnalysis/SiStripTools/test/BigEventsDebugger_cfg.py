import FWCore.ParameterSet.Config as cms
import FWCore.ParameterSet.VarParsing as VarParsing

process = cms.Process("BigEventsDebugger")

#prepare options

options = VarParsing.VarParsing("analysis")

options.register ('globalTag',
                  "DONOTEXIST",
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
process.load("Configuration.StandardSequences.RawToDigi_Data_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("Configuration.StandardSequences.GeometryRecoDB_cff")
process.load("Configuration.StandardSequences.Reconstruction_cff")

process.load("DPGAnalysis.SiStripTools.sipixelclustermultiplicityprod_cfi")
process.load("DPGAnalysis.SiStripTools.sistripclustermultiplicityprod_cfi")
process.seqMultProd = cms.Sequence(process.spclustermultprod+process.ssclustermultprod)

process.load("DPGAnalysis.SiStripTools.ssclusmultinvestigator_cfi")
process.ssclusmultinvestigator.runHisto = cms.untracked.bool(False)
process.ssclusmultinvestigator.scaleFactor=cms.untracked.int32(2)

process.load("DPGAnalysis.SiStripTools.spclusmultinvestigator_cfi")
process.spclusmultinvestigator.runHisto = cms.untracked.bool(False)
process.spclusmultinvestigator.scaleFactor=cms.untracked.int32(5)

process.load("DPGAnalysis.SiStripTools.multiplicitycorr_cfi")
process.multiplicitycorr.correlationConfigurations = cms.VPSet(
   cms.PSet(xMultiplicityMap = cms.InputTag("ssclustermultprod"),
            xDetSelection = cms.uint32(0), xDetLabel = cms.string("TK"), xBins = cms.uint32(3000), xMax=cms.double(100000), 
            yMultiplicityMap = cms.InputTag("spclustermultprod"),
            yDetSelection = cms.uint32(0), yDetLabel = cms.string("Pixel"), yBins = cms.uint32(1000), yMax=cms.double(30000),
            rBins = cms.uint32(200), scaleFactor = cms.untracked.double(5.),
            runHisto=cms.bool(False),runHistoBXProfile=cms.bool(False),runHistoBX=cms.bool(False),runHisto2D=cms.bool(False))
   )

process.seqClusMultInvest = cms.Sequence(process.spclusmultinvestigator + process.ssclusmultinvestigator + process.multiplicitycorr) 

process.froml1abcHEs = cms.EDProducer("EventWithHistoryProducerFromL1ABC",
                                      l1ABCCollection=cms.InputTag("scalersRawToDigi")
                                      )
process.load("DPGAnalysis.SiStripTools.apvcyclephaseproducerfroml1ts2011_cfi")
process.load("DPGAnalysis.SiStripTools.eventtimedistribution_cfi")
process.eventtimedistribution.historyProduct = cms.InputTag("froml1abcHEs")

process.seqEventHistoryReco = cms.Sequence(process.froml1abcHEs + process.APVPhases)
process.seqEventHistory = cms.Sequence(process.eventtimedistribution)

process.load("DPGAnalysis.SiStripTools.bigeventsdebugger_cfi")


process.seqProducers = cms.Sequence(process.seqMultProd
                                    + process.seqEventHistoryReco
)

process.p0 = cms.Path(
   process.siPixelDigis +  process.siStripDigis +
   process.trackerlocalreco +
   process.scalersRawToDigi +
   process.seqProducers +
   process.seqEventHistory +
   process.seqClusMultInvest +
   process.bigeventsdebugger
   )

#----GlobalTag ------------------------

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, options.globalTag, '')

#process.GlobalTag.toGet = cms.VPSet(
#   cms.PSet(record = cms.string("SiPixelTemplateDBObjectRcd"),
##            tag = cms.string("SiPixelTemplateDBObject_38T_v3_mc"),
#            tag = cms.string("SiPixelTemplateDBObject38Tv2_mc"),
#            connect = cms.untracked.string("frontier://FrontierProd/CMS_COND_31X_PIXEL")
#            )
#   )


process.TFileService = cms.Service('TFileService',
                                   fileName = cms.string('BigEventsDebugger.root')
#                                   fileName = cms.string('TrackerLocal_rereco.root')
                                   )

#print process.dumpPython()
