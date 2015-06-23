import FWCore.ParameterSet.Config as cms
import FWCore.ParameterSet.VarParsing as VarParsing

process = cms.Process("OverlapProblemALCAZmumu")

#prepare options

options = VarParsing.VarParsing("analysis")

options.register('globalTag',
                 "DONOTEXIST",
                 VarParsing.VarParsing.multiplicity.singleton, # singleton or list
                 VarParsing.VarParsing.varType.string,          # string, int, or float
                 "GlobalTag")
#options.globalTag = "DONOTEXIST::All"

options.parseArguments()

#
process.load("DPGAnalysis.SiStripTools.processOptions_cff")
process.load("DPGAnalysis.SiStripTools.MessageLogger_cff")

process.MessageLogger.cout.threshold = cms.untracked.string("WARNING")
#process.MessageLogger.debugModules = cms.untracked.vstring("overlapproblemanalyzer")

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )

process.source = cms.Source("PoolSource",
                            fileNames = cms.untracked.vstring(options.inputFiles),
                            #                    skipBadFiles = cms.untracked.bool(True),
                            inputCommands = cms.untracked.vstring("keep *", "drop *_MEtoEDMConverter_*_*")
                            )

process.load("Configuration.StandardSequences.MagneticField_AutoFromDBCurrent_cff")
#process.load("Configuration.StandardSequences.GeometryDB_cff")
process.load("Configuration.Geometry.GeometryExtendedPhaseIPixelReco_cff")
process.load("Configuration.Geometry.GeometryExtendedPhaseIPixel_cff")
process.load("SimTracker.TrackAssociatorProducers.trackAssociatorByHits_cfi")


process.load("DPGAnalysis.SiStripTools.tkAlTrackRefitSequence_cff")
process.refittedTracks.src = cms.InputTag("generalTracks")
process.refittedTracks.TTRHBuilder = cms.string('WithTrackAngle')

process.load('SLHCUpgradeSimulations.Geometry.fakeConditions_Phase1_R30F12_cff')

#process.KFFittingSmootherWithOutliersRejectionAndRK.LogPixelProbabilityCut = cms.double(-16.0)

process.PixelCPEGenericESProducer = cms.ESProducer("PixelCPEGenericESProducer",
                                                   EdgeClusterErrorX = cms.double(50.0),
                                                   DoCosmics = cms.bool(False),
                                                   LoadTemplatesFromDB = cms.bool(False),
                                                   UseErrorsFromTemplates = cms.bool(False),
                                                   eff_charge_cut_highX = cms.double(1.0),
                                                   TruncatePixelCharge = cms.bool(False),
                                                   size_cutY = cms.double(3.0),
                                                   size_cutX = cms.double(3.0),
                                                   inflate_all_errors_no_trk_angle = cms.bool(False),
                                                   IrradiationBiasCorrection = cms.bool(False),
                                                   TanLorentzAnglePerTesla = cms.double(0.106),
                                                   inflate_errors = cms.bool(False),
                                                   eff_charge_cut_lowX = cms.double(0.0),
                                                   eff_charge_cut_highY = cms.double(1.0),
                                                   ClusterProbComputationFlag = cms.int32(0),
                                                   EdgeClusterErrorY = cms.double(85.0),
                                                   ComponentName = cms.string('PixelCPEGeneric'),
                                                   eff_charge_cut_lowY = cms.double(0.0),
                                                   PixelErrorParametrization = cms.string('NOTcmsim'),
                                                   Alpha2Order = cms.bool(True),
                                                   Upgrade = cms.bool(True)
                                                   )

#process.siPixelRecHits.CPE = cms.string('PixelCPEGeneric')

process.load("DPGAnalysis.SiStripTools.overlapproblemtsosanalyzer_cfi")
process.overlapproblemtsoshitfiltered = process.overlapproblemtsosanalyzer.clone(trajTrackAssoCollection = cms.InputTag("HitFilteredTracks"))
process.overlapproblemtsosats = process.overlapproblemtsosanalyzer.clone(trajTrackAssoCollection = cms.InputTag("refittedATSTracks"))

process.overlapproblemtsosall = process.overlapproblemtsosanalyzer.clone(onlyValidRecHit = cms.bool(False))
process.overlapproblemtsoshitfilteredall = process.overlapproblemtsoshitfiltered.clone(onlyValidRecHit = cms.bool(False))
process.overlapproblemtsosatsall = process.overlapproblemtsosats.clone(onlyValidRecHit = cms.bool(False))

process.load("DPGAnalysis.SiStripTools.overlapproblemtpanalyzer_cfi")
#process.load("DebugTools.RecHits.tpanalyzer_cfi")



process.p0 = cms.Path( process.seqTrackRefitting
                       + process.trackAssociatorByHits
                       + process.overlapproblemtsosanalyzer # + process.overlapproblemtsoshitfiltered + process.overlapproblemtsosats
                       + process.overlapproblemtsosall # + process.overlapproblemtsoshitfilteredall + process.overlapproblemtsosatsall 
                       + process.overlapproblemtpanalyzer
#                       + process.tpanalyzer
                       )

#----GlobalTag ------------------------

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_condDBv2_cff")
from Configuration.AlCa.GlobalTag_condDBv2 import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, options.globalTag, '')


process.TFileService = cms.Service('TFileService',
                                   fileName = cms.string('OverlapProblem_tpanalyzer.root')
                                   )

#print process.dumpPython()
