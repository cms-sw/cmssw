import FWCore.ParameterSet.Config as cms

process = cms.Process('SKIM')

# import of standard configurations
process.load('Configuration/StandardSequences/Services_cff')
process.load('FWCore/MessageService/MessageLogger_cfi')
process.load('Configuration/EventContent/EventContent_cff')

# module to select events based on HLT Trigger bits
process.load('HLTrigger/HLTfilters/hltHighLevelDev_cfi')

# Loading "hltHighLevelDev_cfi" defines an EDFilter called hltHighLevelDev
# now we can configure it 

# All events from Zero PD
process.hltHighLevelDev.TriggerResultsTag = cms.InputTag("TriggerResults","","HLT")
process.hltHighLevelDev.HLTPaths = (
   'HLT_PhysicsDeclared',
    )

process.hltHighLevelDev.andOr = True # True = OR, False = AND
process.hltHighLevelDev.throw = False # throw exception on unknown path names
process.hltHighLevelDev.HLTPathsPrescales  = cms.vuint32(
   1,    #'HLT_PhysicsDeclared',
    )
process.hltHighLevelDev.HLTOverallPrescale = cms.uint32 (1)

# All events from Zero PD, prescaled by a factor 10
process.hltHighLevelDev2 = process.hltHighLevelDev.clone(andOr = True)
process.hltHighLevelDev2.HLTPaths = (
   'HLT_PhysicsDeclared',
    )

process.hltHighLevelDev2.andOr = True # True = OR, False = AND
process.hltHighLevelDev2.throw = False # throw exception on unknown path names
process.hltHighLevelDev2.HLTPathsPrescales  = cms.vuint32(
   1,    #'HLT_PhysicsDeclared',
    )
process.hltHighLevelDev2.HLTOverallPrescale = cms.uint32 (10)

process.configurationMetadata = cms.untracked.PSet(
    version = cms.untracked.string('$Revision: 1.2 $'),
    annotation = cms.untracked.string('skim'),
    name = cms.untracked.string('skim')
)
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(100)
)
process.options = cms.untracked.PSet(
    Rethrow = cms.untracked.vstring('ProductNotFound')
)
# Input source
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
'/store/data/BeamCommissioning09/MinimumBias/RAW/v1/000/123/060/16B938A8-7DDD-DE11-873D-003048D37514.root',
'/store/data/BeamCommissioning09/MinimumBias/RAW/v1/000/123/060/20334064-77DD-DE11-9592-001D09F2AF96.root',
'/store/data/BeamCommissioning09/MinimumBias/RAW/v1/000/123/060/56193FC8-7FDD-DE11-BAFD-003048D373AE.root',
'/store/data/BeamCommissioning09/MinimumBias/RAW/v1/000/123/060/68BC42A8-82DD-DE11-8AB2-001D09F282F5.root',
'/store/data/BeamCommissioning09/MinimumBias/RAW/v1/000/123/060/E62B7FEF-75DD-DE11-8BB3-00304879FA4A.root',
#        '/store/data/BeamCommissioning09/ZeroBias/RAW/v1/000/123/056/1A460C9A-6DDD-DE11-8FF9-001617C3B6DC.root',
#        '/store/data/BeamCommissioning09/MinimumBias/RAW/v1/000/123/140/F0B85C46-DCDD-DE11-8312-001D09F2423B.root',
#        '/store/data/BeamCommissioning09/MinimumBias/RAW/v1/000/123/140/F082C441-DCDD-DE11-8DC2-001D09F248F8.root',
#        '/store/data/BeamCommissioning09/MinimumBias/RAW/v1/000/123/140/DC7CCE42-DCDD-DE11-B2C6-001D09F2A465.root',
#        '/store/data/BeamCommissioning09/MinimumBias/RAW/v1/000/123/140/BC946243-DCDD-DE11-8B8D-001D09F28F11.root',
#        '/store/data/BeamCommissioning09/MinimumBias/RAW/v1/000/123/140/B200ED47-DCDD-DE11-B9F6-001D09F24FEC.root',
#        '/store/data/BeamCommissioning09/MinimumBias/RAW/v1/000/123/140/A439AA41-DCDD-DE11-A4A5-001D09F28D54.root',
#        '/store/data/BeamCommissioning09/MinimumBias/RAW/v1/000/123/140/A0A4F142-DCDD-DE11-AE97-0019B9F730D2.root',
#        '/store/data/BeamCommissioning09/MinimumBias/RAW/v1/000/123/140/90D99042-DCDD-DE11-A086-001D09F2462D.root',
#        '/store/data/BeamCommissioning09/MinimumBias/RAW/v1/000/123/140/8A566245-DCDD-DE11-B3E5-001D09F2841C.root',
#        '/store/data/BeamCommissioning09/MinimumBias/RAW/v1/000/123/140/88E4C942-DCDD-DE11-88CC-001D09F24FBA.root',
#        '/store/data/BeamCommissioning09/MinimumBias/RAW/v1/000/123/140/7A1D3F43-DCDD-DE11-97E2-001D09F295FB.root',
#        '/store/data/BeamCommissioning09/MinimumBias/RAW/v1/000/123/140/70323F42-DCDD-DE11-8411-001D09F2514F.root',
#        '/store/data/BeamCommissioning09/MinimumBias/RAW/v1/000/123/140/62B9CB46-DCDD-DE11-B517-001D09F241B9.root'
#    'file:/data/rossin/CMS/store/mc/Summer09/MinBias/GEN-SIM-RAW/STARTUP3X_V8D_900GeV-v1/outputA_4_PhysDecl.root',
#    'file:/data/rossin/CMS/store/mc/Summer09/MinBias/GEN-SIM-RAW/STARTUP3X_V8D_900GeV-v1/outputA_3.root',
#    'file:/data/rossin/CMS/store/mc/Summer09/MinBias/GEN-SIM-RAW/STARTUP3X_V8D_900GeV-v1/outputA_4.root',
#    'file:/data/rossin/CMS/store/mc/Summer09/MinBias/GEN-SIM-RAW/STARTUP3X_V8D_900GeV-v1/outputA_5.root',
#    'file:/data/rossin/CMS/store/mc/Summer09/MinBias/GEN-SIM-RAW/STARTUP3X_V8D_900GeV-v1/outputA_6.root'
    )
)

process.source.inputCommands = cms.untracked.vstring("keep *", "drop *_MEtoEDMConverter_*_*", "drop L1GlobalTriggerObjectMapRecord_hltL1GtObjectMap__HLT")

# All zerobias with PhysicsDeclared
process.output1 = cms.OutputModule("PoolOutputModule",
                                   splitLevel = cms.untracked.int32(0),
                                   outputCommands = cms.untracked.vstring('keep *','drop *_MEtoEDMConverter_*_*'),
                                   fileName = cms.untracked.string('SD_AllZeroBias_900GeV_PhysDecl.root'),
                                   dataset = cms.untracked.PSet(
                                      dataTier = cms.untracked.string('RAW-RECO'),
                                      filterName = cms.untracked.string('SD_AllZeroBias')
                                      ),
                                   SelectEvents = cms.untracked.PSet(
                                      SelectEvents = cms.vstring('skim1')
                                      )
)

# 10% of zerobias with PhysicsDeclared
process.output2 = cms.OutputModule("PoolOutputModule",
                                   splitLevel = cms.untracked.int32(0),
                                   outputCommands = cms.untracked.vstring('keep *','drop *_MEtoEDMConverter_*_*'),
                                   fileName = cms.untracked.string('SD_ZeroBias10_900GeV_PhysDecl.root'),
                                   dataset = cms.untracked.PSet(
                                      dataTier = cms.untracked.string('RAW-RECO'),
                                      filterName = cms.untracked.string('SD_ZeroBias10')
                                      ),
                                   SelectEvents = cms.untracked.PSet(
                                      SelectEvents = cms.vstring('skim2')
                                      )
)

# the usage of trigger bits for selection is explained here:
## https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideEDMPathsAndTriggerBits#Selecting_Pass_for_Some_Trigger 

process.skim1 = cms.Path(process.hltHighLevelDev)
process.skim2 = cms.Path(process.hltHighLevelDev2)


process.out_step = cms.EndPath(process.output1+process.output2)

process.schedule = cms.Schedule(process.skim1,process.skim2)
process.schedule.append(process.out_step)
