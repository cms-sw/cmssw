import FWCore.ParameterSet.Config as cms

# module updating/writing to DB
AlCaRecoTriggerBitsRcdUpdate = cms.EDAnalyzer(
    "AlCaRecoTriggerBitsRcdUpdate",
    
    firstRunIOV = cms.uint32(1),
    lastRunIOV = cms.int32(-1), # -1 means infinity (must be -1 if appending to existing tag)

    # Start with empty list, not looking into DB:
    startEmpty = cms.bool(True),
    # If (startEmpty==False) take AlCaRecoTriggerBitsRcd from EventSetup and remove the
    # following keys entries:
    listNamesRemove = cms.vstring(),
   
    # New triggerLists to add
    # (for updating them first remove with 'listNamesRemove'):
    triggerListsAdd = cms.VPSet(
      cms.PSet(listName = cms.string('TkAlZMuMu'),
               hltPaths = cms.vstring('path_2')
               ),
      cms.PSet(listName = cms.string('TkAlMinBias'),
               hltPaths = cms.vstring('path_1', 'path_3')
               )
      )

    )
