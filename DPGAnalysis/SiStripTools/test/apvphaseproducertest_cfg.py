import FWCore.ParameterSet.Config as cms

from PhysicsTools.PatAlgos.tools.helpers import cloneProcessingSnippet

process = cms.Process("apvphaseTest")

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

#process.MessageLogger.cout.threshold = cms.untracked.string("DEBUG")
#process.MessageLogger.debugModules = cms.untracked.vstring("APVPhases")

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )

process.source = cms.Source("PoolSource",
                    fileNames = cms.untracked.vstring(),
#                    skipBadFiles = cms.untracked.bool(True),
                    inputCommands = cms.untracked.vstring("keep *", "drop *_MEtoEDMConverter_*_*")
                    )



fileNames = cms.untracked.vstring('')

process.load("DPGAnalysis.SiStripTools.eventwithhistoryproducerfroml1abc_cfi")

#process.load("DPGAnalysis.SiStripTools.apvcyclephaseproducerfroml1ts_cfi")

import DPGAnalysis.SiStripTools.apvcyclephaseproducerfroml1ts2011_cfi 
process.APVPhases2011 = DPGAnalysis.SiStripTools.apvcyclephaseproducerfroml1ts2011_cfi.APVPhases 
import DPGAnalysis.SiStripTools.apvcyclephaseproducerfroml1ts2010_cfi 
process.APVPhases2010 = DPGAnalysis.SiStripTools.apvcyclephaseproducerfroml1ts2010_cfi.APVPhases 

process.APVPhases2011.wantHistos = cms.untracked.bool(True)
process.APVPhases2010.wantHistos = cms.untracked.bool(True)

process.APVPhasesEC02010 = process.APVPhases2010.clone(useEC0 = cms.untracked.bool(True),magicOffset = cms.untracked.int32(0))
process.APVPhasesEC02011 = process.APVPhases2011.clone(useEC0 = cms.untracked.bool(True),magicOffset = cms.untracked.int32(0))

process.load("DPGAnalysis.SiStripTools.apvcyclephasemonitor_cfi")
process.apvcyclephasemonitorEC02011 = process.apvcyclephasemonitor.clone(apvCyclePhaseCollection = cms.InputTag("APVPhasesEC02011"))
process.apvcyclephasemonitorEC02010 = process.apvcyclephasemonitor.clone(apvCyclePhaseCollection = cms.InputTag("APVPhasesEC02010"))
process.apvcyclephasemonitor2011 = process.apvcyclephasemonitor.clone(apvCyclePhaseCollection = cms.InputTag("APVPhases2011"))
process.apvcyclephasemonitor2010 = process.apvcyclephasemonitor.clone(apvCyclePhaseCollection = cms.InputTag("APVPhases2010"))

process.load("DPGAnalysis.SiStripTools.eventtimedistribution_cfi")
process.evttimedistrwithnophase = process.eventtimedistribution.clone(phasePartition = cms.untracked.string("None"))
process.evttimedistrwithphaseEC02011 = process.eventtimedistribution.clone(apvPhaseCollection = cms.InputTag("APVPhasesEC02011"))
process.evttimedistrwithphaseEC02010 = process.eventtimedistribution.clone(apvPhaseCollection = cms.InputTag("APVPhasesEC02010"))
process.evttimedistrwithphase2011 = process.eventtimedistribution.clone(apvPhaseCollection = cms.InputTag("APVPhases2011"))
process.evttimedistrwithphase2010 = process.eventtimedistribution.clone(apvPhaseCollection = cms.InputTag("APVPhases2010"))


# cluster analysis

process.load("DPGAnalysis.SiStripTools.sistripclustermultiplicityprod_cfi")

process.load("DPGAnalysis.SiStripTools.ssclusmultinvestigator_cfi")

process.load("DPGAnalysis.SiStripTools.ssclusmulttimecorrelations_cfi")
process.ssclusmulttimecorrelations.historyProduct = cms.InputTag("consecutiveHEs")
process.ssclusmulttimecorrphaseEC02011 = process.ssclusmulttimecorrelations.clone(apvPhaseCollection = cms.InputTag("APVPhasesEC02011"))
process.ssclusmulttimecorrphaseEC02010 = process.ssclusmulttimecorrelations.clone(apvPhaseCollection = cms.InputTag("APVPhasesEC02010"))
process.ssclusmulttimecorrphase2011 = process.ssclusmulttimecorrelations.clone(apvPhaseCollection = cms.InputTag("APVPhases2011"))
process.ssclusmulttimecorrphase2010 = process.ssclusmulttimecorrelations.clone(apvPhaseCollection = cms.InputTag("APVPhases2010"))
process.ssclusmulttimecorrnophase = process.ssclusmulttimecorrelations.clone()
process.ssclusmulttimecorrnophase.wantedSubDets = cms.untracked.VPSet(    
 cms.PSet(detSelection = cms.uint32(0),detLabel = cms.string("TK"),  binMax = cms.int32(9523712/64), phasePartition = cms.untracked.string("None")),
 cms.PSet(detSelection = cms.uint32(3),detLabel = cms.string("TIB"), binMax = cms.int32(1787904/64), phasePartition = cms.untracked.string("None")),
 cms.PSet(detSelection = cms.uint32(4),detLabel = cms.string("TID"), binMax = cms.int32( 565248/64), phasePartition = cms.untracked.string("None")),
 cms.PSet(detSelection = cms.uint32(5),detLabel = cms.string("TOB"), binMax = cms.int32(3303936/64), phasePartition = cms.untracked.string("None")),
 cms.PSet(detSelection = cms.uint32(6),detLabel = cms.string("TEC"), binMax = cms.int32(3866624/64), phasePartition = cms.untracked.string("None"))
 )

#

process.TFileService = cms.Service('TFileService',
                                   fileName = cms.string('apvcyclephaseTest.root')
                                   )

#

process.p0 = cms.Path(process.consecutiveHEs
                      + process.APVPhasesEC02011 + process.APVPhasesEC02010 + process.APVPhases2011 + process.APVPhases2010
                      + process.apvcyclephasemonitorEC02011 + process.apvcyclephasemonitorEC02010
                      + process.apvcyclephasemonitor2011 + process.apvcyclephasemonitor2010
                      + process.evttimedistrwithnophase
                      + process.evttimedistrwithphaseEC02011 + process.evttimedistrwithphaseEC02011
                      + process.evttimedistrwithphase2011 + process.evttimedistrwithphase2010
                      + process.ssclustermultprod + process.ssclusmultinvestigator
                      + process.ssclusmulttimecorrnophase
                      + process.ssclusmulttimecorrphaseEC02011 + process.ssclusmulttimecorrphaseEC02010
                      + process.ssclusmulttimecorrphase2011 + process.ssclusmulttimecorrphase2010)
