import FWCore.ParameterSet.Config as cms

#
# The purpose of this file is to make configuration changes for the Stage1
# L1 trigger, but *ONLY* if the era is active. If it is not, this file should
# do nothing.
# Hence it is safe to import this file all the time, and the changes will only
# be triggered when required.
#
# DO NOT add anything to this file that is not conditional on eras.stage1L1Trigger
# being active. Files importing this one assume that is safe to do so all the
# time.
#
# This file MUST be imported with the "*" format, i.e.
#    from L1Trigger.Configuration.ConditionalStage1Configuration_cff import *
# If you import with just a plain "import", i.e.
#    import L1Trigger.Configuration.ConditionalStage1Configuration_cff
# then the ProcessModifier will be in the wrong namespace and it will not be
# run, so the era customisations will not be applied.
#

from Configuration.StandardSequences.Eras import eras

def _loadStage1Fragments( processObject ) :
#    processObject.load('L1Trigger.L1TCalorimeter.caloStage1Params_cfi')
    processObject.load('L1Trigger.L1TCalorimeter.L1TCaloStage1_cff')
    processObject.load('L1Trigger.L1TCalorimeter.caloConfigStage1PP_cfi')

# A unique name is required so I'll use make sure the name includes the filename
modifyL1TriggerConfigurationConditionalStage1Configuration_cff_ = eras.stage1L1Trigger.makeProcessModifier( _loadStage1Fragments )
