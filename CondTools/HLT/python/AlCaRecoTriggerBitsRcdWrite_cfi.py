import FWCore.ParameterSet.Config as cms

# the module writing to DB
AlCaRecoTriggerBitsRcdWrite = cms.EDAnalyzer(
    "AlCaRecoTriggerBitsRcdWrite",
    firstRunIOV = cms.uint32(1),
    lastRunIOV = cms.int32(-1), # -1 means infinity (must be -1 if appending to existing tag)
    triggerLists = cms.VPSet(
      cms.PSet(listName = cms.string('test2'),
               hltPaths = cms.vstring('path_2')
               ),
      cms.PSet(listName = cms.string('test13'),
               hltPaths = cms.vstring('path_1', 'path_3')
               )
      )
    ) # ends analyzer
