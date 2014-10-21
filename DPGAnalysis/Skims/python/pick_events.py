import FWCore.ParameterSet.Config as cms

process = cms.Process("SKIMb")

process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.load("CondCore.DBCommon.CondDBSetup_cfi")

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
#              '/store/data/Commissioning09/Cosmics/RAW/v1/000/079/153/025F7681-391A-DE11-9556-0016177CA7A0.root'
              '/store/data/Commissioning08/Cosmics/RAW-RECO/CRAFT_ALL_V9_SuperPointing_225-v3/0005/B8FB3273-5DFF-DD11-BEAB-00304875A7B5.root',
              '/store/data/Commissioning08/Cosmics/RAW-RECO/CRAFT_ALL_V9_SuperPointing_225-v3/0005/C0E4F880-5CFF-DD11-B561-0030487624FD.root',
              '/store/data/Commissioning08/Cosmics/RAW-RECO/CRAFT_ALL_V9_SuperPointing_225-v3/0005/EAA9AE47-5FFF-DD11-A966-001A92810ADE.root'
      )
)
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

process.load("Configuration.EventContent.EventContentCosmics_cff")


#--------------------------------------------------
#   Pick a range of events
#     (includes the first and last ones specified)
#--------------------------------------------------
process.pickEvents = cms.EDFilter(
    "PickEvents",

    # the original format to input run/event -based selection is described in :
    # DPGAnalysis/Skims/data/listrunev
    # and kept as default, for historical reasons
    RunEventList = cms.untracked.string("DPGAnalysis/Skims/data/listrunev"),

    # run/lumiSection @json -based input of selection can be toggled (but not used in THIS example)
    IsRunLsBased  = cms.bool(False),

    # json is not used in this example -> list of LS left empty
    LuminositySectionsBlockRange = cms.untracked.VLuminosityBlockRange( () )

    )

process.PickEventsPath  = cms.Path( process.pickEvents )

process.out = cms.OutputModule("PoolOutputModule",
           outputCommands = cms.untracked.vstring('keep *','drop *_MEtoEDMConverter_*_*'),
           SelectEvents = cms.untracked.PSet(
                          SelectEvents = cms.vstring('PickEventsPath')
                          ),
           dataset = cms.untracked.PSet(
		     dataTier = cms.untracked.string('RAW-RECO'),
                     filterName = cms.untracked.string('PickEvents')),
  
          fileName = cms.untracked.string("PickEvents.root"),
)
process.this_is_the_end = cms.EndPath(process.out)
