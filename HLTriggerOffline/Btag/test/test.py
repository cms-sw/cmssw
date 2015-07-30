import FWCore.ParameterSet.Config as cms
process = cms.Process("HLTBTAG")

from PhysicsTools.PatAlgos.tools.coreTools import *	
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load("Configuration.StandardSequences.GeometryRecoDB_cff")
process.load('Configuration.StandardSequences.Services_cff')
process.load("DQMServices.Components.EDMtoMEConverter_cff")
process.load("L1TriggerConfig.L1GtConfigProducers.L1GtConfig_cff")

#load hltJetMCTools sequence for the jet/partons matching
process.load("HLTriggerOffline.Btag.hltBtagJetMCTools_cff")

#read config.ini
from HLTriggerOffline.Btag.readConfig import *
fileini = fileINI("config.ini")
fileini.read()

#print read variables
print
print "Reading ", fileini.fileName
print
print "maxEvents = ",fileini.maxEvents
print "CMSSWVER = ",fileini.CMSSWVER
print "processname = ",fileini.processname
print "jets (for matching) = ",fileini.jets
print "files = ",fileini.files
print "btag_modules ",fileini.btag_modules
print "btag_pathes ",fileini.btag_pathes
print "vertex_modules ",fileini.vertex_modules
print "vertex_pathes ",fileini.vertex_pathes
print

triggerFilter = []
triggerFilter.extend(fileini.vertex_pathes)
triggerFilter.extend(fileini.btag_pathes)
triggerFilter = list(set(triggerFilter))
triggerString = ""

for i in range(len(triggerFilter)):
	if i is not 0:
		triggerString += " OR "
	
	triggerString +=  triggerFilter[i] + "*"

print "triggerString : ",triggerString

#denominator trigger
process.hltBtagTriggerSelection = cms.EDFilter( "TriggerResultsFilter",
    triggerConditions = cms.vstring(
      triggerString),
    hltResults = cms.InputTag( "TriggerResults", "", fileini.processname ),
#    l1tResults = cms.InputTag( "gtDigis" ),
#    l1tIgnoreMask = cms.bool( False ),
#    l1techIgnorePrescales = cms.bool( False ),
#    daqPartitions = cms.uint32( 1 ),
    throw = cms.bool( True )
)

#correct the jet used for the matching
process.hltBtagJetsbyRef.jets = cms.InputTag(fileini.jets)

#define VertexValidationVertices for the vertex DQM validation
process.VertexValidationVertices= cms.EDAnalyzer("HLTVertexPerformanceAnalyzer",
	TriggerResults = cms.InputTag('TriggerResults','',fileini.processname),
	HLTPathNames = cms.vstring(fileini.vertex_pathes),
	Vertex = fileini.vertex_modules,
	SimVertexCollection = cms.InputTag("g4SimHits"),
)

#define bTagValidation for the b-tag DQM validation (distribution plot)
process.bTagValidation = cms.EDAnalyzer("HLTBTagPerformanceAnalyzer",
	TriggerResults = cms.InputTag('TriggerResults','',fileini.processname),
	HLTPathNames = cms.vstring(fileini.btag_pathes),
	JetTag = fileini.btag_modules,
	MinJetPT = cms.double(20),
	mcFlavours = cms.PSet(
	light = cms.vuint32(1, 2, 3, 21), # udsg
	c = cms.vuint32(4),
	b = cms.vuint32(5),
	g = cms.vuint32(21),
	uds = cms.vuint32(1, 2, 3)
	),
	mcPartons = cms.InputTag("hltBtagJetsbyValAlgo")
)

#define bTagPostValidation for the b-tag DQM validation (efficiency and mistagrate plot)
process.bTagPostValidation = cms.EDAnalyzer("HLTBTagHarvestingAnalyzer",
	HLTPathNames = fileini.btag_pathes,
	histoName	= fileini.btag_modules_string,
	minTag	= cms.double(0.6),
	# MC stuff
	mcFlavours = cms.PSet(
		light = cms.vuint32(1, 2, 3, 21), # udsg
		c = cms.vuint32(4),
		b = cms.vuint32(5),
		g = cms.vuint32(21),
		uds = cms.vuint32(1, 2, 3)
	)
)
#read input file
process.source = cms.Source("PoolSource",
	fileNames = cms.untracked.vstring(fileini.files)
)

#put all in a path
process.DQM_BTag = cms.Path(
process.hltBtagTriggerSelection
+	process.hltBtagJetMCTools
+	process.VertexValidationVertices
+	process.bTagValidation
+	process.bTagPostValidation
#+	process.EDMtoMEConverter
+	process.dqmSaver
)	

#Settings equivalent to 'RelVal' convention:
process.dqmSaver.convention = 'Offline'
process.dqmSaver.saveByRun = cms.untracked.int32(-1)
process.dqmSaver.saveAtJobEnd = cms.untracked.bool(True)
process.dqmSaver.forceRunNumber = cms.untracked.int32(1)
process.dqmSaver.workflow = "/" + fileini.CMSSWVER + "/RelVal/TrigVal"
process.DQMStore.collateHistograms = False
process.DQMStore.verbose=0
process.options = cms.untracked.PSet(
	wantSummary	= cms.untracked.bool( True ),
	fileMode	= cms.untracked.string('FULLMERGE'),
	SkipEvent	= cms.untracked.vstring('ProductNotFound')
)

#maxEvents
process.maxEvents = cms.untracked.PSet(
	input = cms.untracked.int32(int(fileini.maxEvents))
)
