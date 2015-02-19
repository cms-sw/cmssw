import FWCore.ParameterSet.Config as cms

hcalBeamMonitor=cms.EDAnalyzer("HcalBeamMonitor",
                               # base class stuff
                               debug                  = cms.untracked.int32(0),
                               online                 = cms.untracked.bool(False),
                               AllowedCalibTypes      = cms.untracked.vint32(0),
                               mergeRuns              = cms.untracked.bool(False),
                               enableCleanup          = cms.untracked.bool(False),
                               subSystemFolder        = cms.untracked.string("Hcal/"),
                               TaskFolder             = cms.untracked.string("BeamMonitor_Hcal/"),
                               skipOutOfOrderLS       = cms.untracked.bool(True),
                               NLumiBlocks            = cms.untracked.int32(4000),
                               
                               # BeamMonitor-specific Info
                               
                               # Input collections
                               hbheRechitLabel        = cms.untracked.InputTag("hbhereco"),
                               hoRechitLabel          = cms.untracked.InputTag("horeco"),
                               hfRechitLabel          = cms.untracked.InputTag("hfreco"),
                               digiLabel              = cms.untracked.InputTag("hcalDigis"),

                               minEvents              = cms.untracked.int32(500), # changed from 500 to 250 when lumi section changed, but saw a lot of errors.  Re-evaluate at some point?
                               lumiqualitydir         = cms.untracked.string(""),
                               occupancyThresh        = cms.untracked.double(0.0625),
                               hotrate                = cms.untracked.double(0.25),
                               minBadCells            = cms.untracked.int32(10),
                               Overwrite              = cms.untracked.bool(False),
                               FEDRawDataCollection=cms.untracked.InputTag("rawDataCollector")
                               )
