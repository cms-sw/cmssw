#########################
#
# Base file for L1 branch merging
#
# This script works on any official production sample
#
# Instruction to run this script are provided on this page:
#
# http://sviret.web.cern.ch/sviret/Welcome.php?n=CMS.HLLHCTuto
#
# Look at STEP V
#
# Author: S.Viret (viret@in2p3.fr)
# Date        : 17/02/2014
#
# Script tested with release CMSSW_6_2_0_SLHC7
#
#########################

import FWCore.ParameterSet.Config as cms

process = cms.Process('AMPR')

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('Configuration.Geometry.GeometryExtendedPhase2TkBE5DReco_cff')
process.load('Configuration.Geometry.GeometryExtendedPhase2TkBE5D_cff')
process.load('Configuration.StandardSequences.MagneticField_38T_PostLS1_cff')
process.load('L1Trigger.TrackFindingAM.L1AMTrack_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
process.load('L1Trigger.TrackTrigger.TrackTrigger_cff')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

process.source = cms.Source("PoolSource",
                            fileNames = cms.untracked.vstring('INPUTFILENAME'),
                            duplicateCheckMode = cms.untracked.string( 'noDuplicateCheck' )
)

# Additional output definition
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'MYGLOBALTAG', '')

process.MergePROutput.TTInputPatterns     = cms.VInputTag( INPUTBRANCHES )

process.RAWSIMoutput = cms.OutputModule("PoolOutputModule",
    splitLevel = cms.untracked.int32(0),
    eventAutoFlushCompressedSize = cms.untracked.int32(5242880),
    outputCommands = process.RAWSIMEventContent.outputCommands,
    fileName = cms.untracked.string('OUTPUTFILENAME'),
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string(''),
        dataTier = cms.untracked.string('GEN-SIM')
    )
)

# For the moment need to explicitely keep the following containers
# (not yet in the customizing scripts)

process.RAWSIMoutput.outputCommands.append('drop  *_*_*_AMPRBASE')
process.RAWSIMoutput.outputCommands.append('keep  *_*_*_AMPR')
process.RAWSIMoutput.outputCommands.append('keep  *_*_MergedTrackTruth_*')

# Path and EndPath definitions
process.L1AMPR_step          = cms.Path(process.MergePROutputs)
process.endjob_step          = cms.EndPath(process.endOfProcess)
process.RAWSIMoutput_step    = cms.EndPath(process.RAWSIMoutput)

process.schedule = cms.Schedule(process.L1AMPR_step,process.endjob_step,process.RAWSIMoutput_step)

# Automatic addition of the customisation function

from SLHCUpgradeSimulations.Configuration.phase2TkCustomsBE5D import customise as customiseBE5D
from SLHCUpgradeSimulations.Configuration.phase2TkCustomsBE5D import l1EventContent as customise_ev_BE5D

process=customiseBE5D(process)
process=customise_ev_BE5D(process)
