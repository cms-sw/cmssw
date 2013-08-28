import FWCore.ParameterSet.Config as cms

process = cms.Process("DTVDriftWriter")

# the source
process.source = cms.Source("EmptySource")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
    )

process.load("Configuration.StandardSequences.Geometry_cff")

from CondCore.DBCommon.CondDBSetup_cfi import CondDBSetup
process.PoolDBOutputService = cms.Service("PoolDBOutputService",
                                          CondDBSetup,
                                          toPut = cms.VPSet(cms.PSet(
                                                            record = cms.string('DTMtimeRcd'),
                                                            tag = cms.string('vDrift')
                                                            )),
                                          connect = cms.string('sqlite_file:vDrift.db'),
                                          timetype = cms.untracked.string('runnumber'),
                                          )



process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = "CRAFT_V2P::All"
    
#Module to compute vdrift and hit resolution
process.dtvdriftwriter = cms.EDAnalyzer("DTVDriftWriter",
                                        # Switch on/off the verbosity
                                        debug = cms.untracked.bool(True),
                                        # Name of the ROOT file which contains the TMax histos
                                        rootFileName = cms.untracked.string("DTTMaxHistosFromData_partialSum.root"),
                                       	# Name of the txt file which will contain the calibrated v_drift
                                        vDriftFileName = cms.untracked.string("DTVDriftCalibration.txt"),
                                        #  Parameter set for  DTCalibrationMap constructor
                                        calibFileConfig = cms.untracked.PSet(
                                               # Chosen granularity (N.B. bySL is the only one implemented for calibration at the moment)
                                               calibConstGranularity = cms.untracked.string("bySL"))
                                        )


process.path = cms.Path(process.dtvdriftwriter)


