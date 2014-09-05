import FWCore.ParameterSet.Config as cms

process = cms.Process("HLTHARVEST")

from PhysicsTools.PatAlgos.tools.coreTools import *	


process.load('FWCore.MessageService.MessageLogger_cfi')
process.load("Configuration.StandardSequences.Geometry_cff")
process.load('Configuration.StandardSequences.Services_cff')
process.load("DQMServices.Components.EDMtoMEConverter_cff")
process.load("L1TriggerConfig.L1GtConfigProducers.L1GtConfig_cff")
process.load("HLTriggerOffline.Btag.hltJetMCTools_cff")

# read them from my.ini

#process.load("HLTriggerOffline.Btag.Validation.readConfig")
from HLTriggerOffline.Btag.readConfig import *
#ReadFile("my.ini")

fileini = fileINI("config.ini")
fileini.read()

#def readConfig(fileName)
print
print  "Reading ", fileini.fileName
print
#print  "FastPrimaryVertex	", fileini.TagInfos
print  "maxEvents		=	",fileini.maxEvents
print  "CMSSWVER		=	",fileini.CMSSWVER
print  "processname		=	",fileini.processname
print  "jets (for matching)	=	",fileini.jets
print  "files			=	",fileini.files
print  "btag_modules		",fileini.btag_modules
print  "btag_pathes		",fileini.btag_pathes
print  "vertex_modules		",fileini.vertex_modules
print  "vertex_pathes		",fileini.vertex_pathes
print

process.hltJetsbyRef.jets = cms.InputTag(fileini.jets)
process.hltPartons.src = cms.InputTag("genParticles")

process.VertexValidationVertices= cms.EDAnalyzer("HLTVertexPerformanceAnalyzer",
   TriggerResults  = cms.InputTag('TriggerResults','',fileini.processname),
   HLTPathNames     = cms.vstring(fileini.vertex_pathes),
   Vertex        = fileini.vertex_modules,
)

process.bTagValidation     = cms.EDAnalyzer("HLTBTagPerformanceAnalyzer",
   TriggerResults  = cms.InputTag('TriggerResults','',fileini.processname),
   HLTPathNames     = cms.vstring(fileini.btag_pathes),
   JetTag        = fileini.btag_modules,
   MinJetPT        = cms.double(20),
   mcFlavours = cms.PSet(
      light = cms.vuint32(1, 2, 3, 21),   # udsg
      c = cms.vuint32(4),
      b = cms.vuint32(5),
      g = cms.vuint32(21),
      uds = cms.vuint32(1, 2, 3)
    ),
   mcPartons = cms.InputTag("hltJetsbyValAlgo")
)

process.bTagPostValidation = cms.EDAnalyzer("HLTBTagHarvestingAnalyzer",
   HLTPathNames     = fileini.btag_pathes,
   histoName		= fileini.btag_modules_string,
   minTag			= cms.double(0.6),
  # MC stuff
   mcFlavours = cms.PSet(
      light = cms.vuint32(1, 2, 3, 21),   # udsg
      c = cms.vuint32(4),
      b = cms.vuint32(5),
      g = cms.vuint32(21),
      uds = cms.vuint32(1, 2, 3)
    )
)

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(fileini.files)
)

process.DQM_BTag = cms.Path(
	process.hltJetMCTools
+	process.VertexValidationVertices
+	process.bTagValidation
+	process.bTagPostValidation
+	process.EDMtoMEConverter
+	process.dqmSaver
)	

process.dqmSaver.convention = 'Offline'
#Settings equivalent to 'RelVal' convention:
process.dqmSaver.saveByRun = cms.untracked.int32(-1)
process.dqmSaver.saveAtJobEnd = cms.untracked.bool(True)
process.dqmSaver.forceRunNumber = cms.untracked.int32(1)
process.dqmSaver.workflow = "/" + fileini.CMSSWVER + "/RelVal/TrigVal"

process.DQMStore.collateHistograms = False
process.DQMStore.verbose=0

process.options = cms.untracked.PSet(
    fileMode = cms.untracked.string('FULLMERGE'),
    SkipEvent = cms.untracked.vstring('ProductNotFound')
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(int(fileini.maxEvents))
)
