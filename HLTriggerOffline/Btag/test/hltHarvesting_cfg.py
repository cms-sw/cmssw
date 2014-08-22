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

filexml = fileXML("my.ini")
filexml.read()

#def readConfig(fileName)
print
print  "Reading ", filexml.fileName
print
print  "l3TagInfos		=	",  filexml.l3TagInfos
print  "l25JetTags		=	", filexml.l25JetTags
print  "l3JetTags		=	", filexml.l3JetTags
print  "minTags			=	",filexml.mintags
print  "maxEvents		=	",filexml.maxEvents
print  "CMSSWVER		=	",filexml.CMSSWVER
print  "processname		=	",filexml.processname
print  "jets (for matching)	=	",filexml.jets
print "genParticlesProcess	=	", filexml.genParticlesProcess
print  "HLTPathNames		=	",filexml.HLTPathNames
print  "BTagAlgorithms		=	",filexml.BTagAlgorithms
print  "files			=	",filexml.files
print
print

process.hltJetsbyRef.jets = cms.InputTag(filexml.jets,"",filexml.processname)
process.hltPartons.src = cms.InputTag("genParticles","",filexml.genParticlesProcess)

process.bTagValidation     = cms.EDAnalyzer("HLTBTagPerformanceAnalyzer",
   TriggerResults  = cms.InputTag('TriggerResults','',filexml.processname),
   HLTPathNames     = cms.vstring(filexml.HLTPathNames),
   L25IPTagInfo    = cms.VInputTag(cms.InputTag('hltBLifetimeL25TagInfosbbPhiL1FastJetFastPV')),
   L25JetTag       = filexml.l25JetTags,
   L3IPTagInfo     = filexml.l3TagInfos,
   L3JetTag        = filexml.l3JetTags,
   TrackIPTagInfo  = cms.InputTag('NULL'),
   OfflineJetTag   = cms.InputTag('NULL'),
   MinJetPT        = cms.double(20),
   BTagAlgorithms   = cms.vstring(filexml.BTagAlgorithms),
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
   HLTPathNames     = cms.vstring(filexml.HLTPathNames),
   minTags=cms.vdouble(filexml.mintags), # TCHP , 6 -- TCH6
   maxTag=cms.double(100.),
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
    fileNames = cms.untracked.vstring(filexml.files)
)

process.DQM_BTag = cms.Path(
	process.hltJetMCTools
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
process.dqmSaver.workflow = "/" + filexml.CMSSWVER + "/RelVal/TrigVal"

process.DQMStore.collateHistograms = False
process.DQMStore.verbose=0

process.options = cms.untracked.PSet(
    fileMode = cms.untracked.string('FULLMERGE'),
    SkipEvent = cms.untracked.vstring('ProductNotFound')
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(int(filexml.maxEvents))
)
