import FWCore.ParameterSet.Config as cms
import FWCore.ParameterSet.VarParsing as VarParsing


process = cms.Process("cosmicstracks")

#prepare options

options = VarParsing.VarParsing("analysis")

options.register ('globalTag',
                  "DONOTEXIST",
                  VarParsing.VarParsing.multiplicity.singleton, # singleton or list
                  VarParsing.VarParsing.varType.string,          # string, int, or float
                  "GlobalTag")

options.parseArguments()

process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool(True),
    fileMode = cms.untracked.string("FULLMERGE")
    )

process.load("FWCore.MessageService.MessageLogger_cfi")

#----------------------------------------------------------------

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

#process.MessageLogger.suppressWarning.append("consecutiveHEs")


#------------------------------------------------------------------

#process.MessageLogger.cout.threshold = cms.untracked.string("DEBUG")
#process.MessageLogger.debugModules = cms.untracked.vstring("eventtimedistribution")

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )

process.source = cms.Source("PoolSource",
                    fileNames = cms.untracked.vstring(options.inputFiles),
#                    skipBadFiles = cms.untracked.bool(True),
                    inputCommands = cms.untracked.vstring("keep *", "drop *_MEtoEDMConverter_*_*")
                    )




process.load("DPGAnalysis.SiStripTools.eventwithhistoryproducerfroml1abc_cfi")

import DPGAnalysis.SiStripTools.apvcyclephaseproducerfroml1tsDB_cfi 
process.APVPhases = DPGAnalysis.SiStripTools.apvcyclephaseproducerfroml1tsDB_cfi.APVPhases 
process.APVPhases.wantHistos = cms.untracked.bool(True)

#------- Filter for isolated pairs
process.retrigger = cms.EDFilter('EventWithHistoryEDFilter',
                                 commonConfiguration = cms.untracked.PSet(historyProduct= cms.untracked.InputTag("consecutiveHEs"),
                                                                          APVPhaseLabel= cms.untracked.string("APVPhases")
                                                                          ),
                                 filterConfigurations = cms.untracked.VPSet(cms.PSet(dbxRange = cms.untracked.vint32(0,6)))
                                 )
process.closeretrigger = cms.EDFilter('EventWithHistoryEDFilter',
                                      commonConfiguration = cms.untracked.PSet(historyProduct= cms.untracked.InputTag("consecutiveHEs"),
                                                                               APVPhaseLabel= cms.untracked.string("APVPhases")
                                                                               ),
                                      filterConfigurations = cms.untracked.VPSet(cms.PSet(dbxRange = cms.untracked.vint32(0,3)))
                                      )

process.load("DPGAnalysis.SiStripTools.apvcyclephasemonitor_cfi")

process.load("DPGAnalysis.SiStripTools.eventtimedistribution_cfi")
process.eventtimedistribution.wantEWHDepthHisto = cms.untracked.bool(True)
process.eventtimedistribution.wantDBXvsBX = cms.untracked.bool(True)

process.eventtimeretrigger = process.eventtimedistribution.clone()
process.eventtimecloseretrigger = process.eventtimedistribution.clone()

process.load("DPGAnalysis.SiStripTools.trackcount_cfi")
process.trackcount.trackCollection = cms.InputTag('ctfWithMaterialTracksP5')
process.trackcountretrigger = process.trackcount.clone()
process.trackcountcloseretrigger = process.trackcount.clone()

process.TFileService = cms.Service('TFileService',
                                   fileName = cms.string('cosmicstracks.root')
                                   )

#----GlobalTag ------------------------

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_condDBv2_cff")
from Configuration.AlCa.GlobalTag_condDBv2 import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, options.globalTag, '')

#

process.p0 = cms.Path(process.consecutiveHEs
                      + process.APVPhases
                      + process.apvcyclephasemonitor
                      + process.eventtimedistribution
                      + process.trackcount
                      )
process.pretrigger = cms.Path(process.consecutiveHEs
                              + process.APVPhases
                              + process.apvcyclephasemonitor
                              + process.retrigger
                              + process.eventtimeretrigger
                              + process.trackcountretrigger
                              )
process.pcloseretrigger = cms.Path(process.consecutiveHEs
                                   + process.APVPhases
                                   + process.apvcyclephasemonitor
                                   + process.closeretrigger
                                   + process.eventtimecloseretrigger
                                   + process.trackcountcloseretrigger
                              )


#process.schedule = cms.Schedule(process.p0)
