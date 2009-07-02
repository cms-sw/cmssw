
import FWCore.ParameterSet.Config as cms

#
# add an output module to write out AODSIM. The use case is
# for production to be able to write out RECO and AOD/AODSIM
# in the same step.
# Memory consumption to be evaluated.
#

def customise(process):
    process.load("Configuration.EventContent.EventContent_cff")

    process.output2 = cms.OutputModule("PoolOutputModule",
                                   process.AODSIMEventContent,
                                   dataset = cms.untracked.PSet(dataTier = cms.untracked.string('AODSIM')),
                                   fileName = cms.untracked.string( 'file:AODSIM.root' )
                                   )

    process.out2 = cms.EndPath(process.output2)
    process.schedule.append(process.out2)

    return (process)
