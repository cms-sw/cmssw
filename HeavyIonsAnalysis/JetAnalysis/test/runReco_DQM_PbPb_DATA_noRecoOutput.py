# Auto generated configuration file
# using: 
# Revision: 1.381.2.28 
# Source: /local/reps/CMSSW/CMSSW/Configuration/PyReleaseValidation/python/ConfigBuilder.py,v 
# with command line options: hiReco --conditions GR_R_53_LV2B::All -s RAW2DIGI,L1Reco,RECO --scenario HeavyIons --datatier RECO --eventcontent RECO -n1 --no_exec --repacked
import FWCore.ParameterSet.Config as cms

process = cms.Process('DQM')

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContentHeavyIons_cff')
process.load('SimGeneral.MixingModule.mixNoPU_cfi')
process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load('Configuration.StandardSequences.MagneticField_38T_cff')
process.load('Configuration.StandardSequences.RawToDigi_Data_cff')
process.load('Configuration.StandardSequences.L1Reco_cff')
process.load('Configuration.StandardSequences.ReconstructionHeavyIons_cff')
process.load('DQMOffline.Configuration.DQMOfflineHeavyIons_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

# Input source
process.source = cms.Source("PoolSource",
    secondaryFileNames = cms.untracked.vstring(),
    fileNames = 
cms.untracked.vstring('file:/mnt/hadoop/cms/store/user/yjlee/HIHighPt/HIHighPt_RAW_HLTJet55or65Skim_v4/001ed3dc956dd889d0fd27ce36fb998b/SD_HLTJet55Jet65_1000_2_eWj.root')
)

process.options = cms.untracked.PSet(

)

process.SimpleMemoryCheck=cms.Service("SimpleMemoryCheck",
                                      oncePerEventMode=cms.untracked.bool(False))

process.Timing=cms.Service("Timing",
                           useJobReport = cms.untracked.bool(True)
                           )


# Production Info
process.configurationMetadata = cms.untracked.PSet(
    version = cms.untracked.string('$Revision: 1.381.2.28 $'),
    annotation = cms.untracked.string('hiReco nevts:1'),
    name = cms.untracked.string('PyReleaseValidation')
)

# Output definition

# process.RECOoutput = cms.OutputModule("PoolOutputModule",
#     splitLevel = cms.untracked.int32(0),
#     eventAutoFlushCompressedSize = cms.untracked.int32(5242880),
#     outputCommands = process.RECOEventContent.outputCommands,
#     fileName = cms.untracked.string('hiReco_RAW2DIGI_L1Reco_RECO.root'),
#     dataset = cms.untracked.PSet(
#         filterName = cms.untracked.string(''),
#         dataTier = cms.untracked.string('RECO')
#     ),
# #                                      SelectEvents = cms.untracked.PSet(
# #    SelectEvents = cms.vstring('hltTrigger')
# #    )
#                                       
# )
process.DQMoutput = cms.OutputModule("PoolOutputModule",
    splitLevel = cms.untracked.int32(0),
    outputCommands = process.DQMEventContent.outputCommands,
    fileName = cms.untracked.string('file:step2_inDQM.root'),
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string(''),
        dataTier = cms.untracked.string('DQM')
    )
)

# Additional output definition

# Other statements

from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'GR_R_53_LV6::All', '')

#process.load("HLTrigger.HLTfilters.hltHighLevel_cfi")
#process.hltHighLevel.HLTPaths = cms.vstring("HLT_HIJet80_v*")
#process.hltHighLevel.eventSetupPathsKey = cms.string("")
#process.hltHighLevel.andOr = cms.bool(True) # False Only takes events with all triggers at same time.
#process.hltHighLevel.throw = False

# Path and EndPath definitions
#process.hltTrigger = cms.Path(process.hltHighLevel)
process.raw2digi_step = cms.Path(process.RawToDigi)
process.L1Reco_step = cms.Path(process.L1Reco)
process.reconstruction_step = cms.Path(process.reconstructionHeavyIons)
process.endjob_step = cms.EndPath(process.endOfProcess)
#process.RECOoutput_step = cms.EndPath(process.RECOoutput)
#DQM
process.dqmoffline_step = cms.Path(process.DQMOfflineHeavyIons)
process.DQMoutput_step = cms.EndPath(process.DQMoutput)

process.SimpleMemoryCheck=cms.Service("SimpleMemoryCheck",
                                       oncePerEventMode=cms.untracked.bool(False))

# Schedule definition
process.schedule = cms.Schedule(process.raw2digi_step,process.L1Reco_step,process.reconstruction_step,process.dqmoffline_step,process.endjob_step,process.DQMoutput_step)





from Configuration.PyReleaseValidation.ConfigBuilder import MassReplaceInputTag
MassReplaceInputTag(process)

