import FWCore.ParameterSet.Config as cms
import FWCore.ParameterSet.VarParsing as VarParsing

process = cms.Process("OverlapProblemALCAZmumu")

#prepare options

options = VarParsing.VarParsing()

options.register ('globalTag',
                  "DONOTEXIST",
                  VarParsing.VarParsing.multiplicity.singleton, # singleton or list
                  VarParsing.VarParsing.varType.string,          # string, int, or float
                  "GlobalTag")
#options.globalTag = "DONOTEXIST::All"

options.parseArguments()

#
process.load("DPGAnalysis.SiStripTools.processOptions_cff")
process.load("DPGAnalysis.SiStripTools.MessageLogger_cff")

#process.MessageLogger.cout.threshold = cms.untracked.string("DEBUG")
#process.MessageLogger.debugModules = cms.untracked.vstring("overlapproblemanalyzer")

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )

process.load("DPGAnalysis.SiStripTools.poolSource_cff")

process.source.fileNames = cms.untracked.vstring(
#    "rfio:/castor/cern.ch/user/v/venturia/SingleMuPt15_tec5_GEN_SIM_RECODEBUG_default.root")
#    "rfio:/castor/cern.ch/user/v/venturia/SingleMuPt15_tec5_GEN_SIM_RECODEBUG_1500um.root")
    "rfio:/castor/cern.ch/cms/store/mc/Fall10/DYToMuMu_M-20_TuneZ2_7TeV-pythia6/ALCARECO/START38_V12_TkAlZMuMu-v1/0001/86BB9127-E5D9-DF11-A995-00215E2222E0.root")    

process.load("Configuration.StandardSequences.MagneticField_AutoFromDBCurrent_cff")
process.load("Configuration.StandardSequences.GeometryRecoDB_cff")
process.load("SimTracker.TrackAssociatorProducers.trackAssociatorByHits_cfi")

process.load("DPGAnalysis.SiStripTools.tkAlTrackRefitSequence_cff")
process.refittedTracks.src = cms.InputTag("ALCARECOTkAlZMuMu")


process.offlineBeamSpot = cms.EDProducer("BeamSpotProducer")

process.load("DPGAnalysis.SiStripTools.overlapproblemtsosanalyzer_cfi")
process.overlapproblemtsoshitfiltered = process.overlapproblemtsosanalyzer.clone(trajTrackAssoCollection = cms.InputTag("HitFilteredTracks"))
process.overlapproblemtsosats = process.overlapproblemtsosanalyzer.clone(trajTrackAssoCollection = cms.InputTag("refittedATSTracks"))

process.overlapproblemtsosall = process.overlapproblemtsosanalyzer.clone(onlyValidRecHit = cms.bool(False))
process.overlapproblemtsoshitfilteredall = process.overlapproblemtsoshitfiltered.clone(onlyValidRecHit = cms.bool(False))
process.overlapproblemtsosatsall = process.overlapproblemtsosats.clone(onlyValidRecHit = cms.bool(False))


process.p0 = cms.Path(process.offlineBeamSpot 
                      + process.seqTrackRefitting
                      + process.trackAssociatorByHits
                      + process.overlapproblemtsosanalyzer + process.overlapproblemtsoshitfiltered + process.overlapproblemtsosats
                      + process.overlapproblemtsosall + process.overlapproblemtsoshitfilteredall + process.overlapproblemtsosatsall 
                      )

#----GlobalTag ------------------------

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_condDBv2_cff")
from Configuration.AlCa.GlobalTag_condDBv2 import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, options.globalTag, '')


process.TFileService = cms.Service('TFileService',
                                   fileName = cms.string('OverlapProblem_ALCAZmumu_multi.root')
                                   )

