import FWCore.ParameterSet.Config as cms

def customise_debug(process):
    # EMTF Phase2 Emulator
    process.load('L1Trigger.L1TMuonEndCapPhase2.simCscTriggerPrimitiveDigisForEMTF_cfi')
    process.load('L1Trigger.L1TMuonEndCapPhase2.rpcRecHitsForEMTF_cfi')
    process.load('L1Trigger.L1TMuonEndCapPhase2.simEmtfDigisPhase2_cfi')

    process.gemRecHits.gemDigiLabel = 'simMuonGEMDigis'
    process.simEmtfDigisPhase2.Verbosity = cms.untracked.int32(5)

    process.L1TMuonEndCapPhase2Task = cms.Task(
        process.simCscTriggerPrimitiveDigisForEMTF,
        process.rpcRecHitsForEMTF,
        process.simEmtfDigisPhase2
    )

    process.L1TMuonEndCapPhase2Sequence = cms.Sequence(
        process.L1TMuonEndCapPhase2Task
    )

    # Path
    process.L1TMuonEndCapPhase2_step = cms.Path(process.L1TMuonEndCapPhase2Sequence)

    process.schedule.extend([process.L1TMuonEndCapPhase2_step])

    # Remove cms.EndPath instances from schedule
    paths_in_schedule = [path for path in process.schedule if not isinstance(path, cms.EndPath)]
    process.schedule = cms.Schedule(*paths_in_schedule)
    return process

def customise_mc(process):
    # EMTF Phase2 Emulator
    process.load('L1Trigger.L1TMuonEndCapPhase2.simCscTriggerPrimitiveDigisForEMTF_cfi')
    process.load('L1Trigger.L1TMuonEndCapPhase2.rpcRecHitsForEMTF_cfi')
    process.load('L1Trigger.L1TMuonEndCapPhase2.simEmtfDigisPhase2_cfi')

    process.gemRecHits.gemDigiLabel = 'simMuonGEMDigis'
    process.simEmtfDigisPhase2.Verbosity = cms.untracked.int32(1)

    process.L1TMuonEndCapPhase2Task = cms.Task(
        process.simCscTriggerPrimitiveDigisForEMTF,
        process.rpcRecHitsForEMTF,
        process.simEmtfDigisPhase2
    )

    process.L1TMuonEndCapPhase2Sequence = cms.Sequence(
        process.L1TMuonEndCapPhase2Task
    )

    # Path
    process.L1TMuonEndCapPhase2_step = cms.Path(process.L1TMuonEndCapPhase2Sequence)

    process.schedule.extend([process.L1TMuonEndCapPhase2_step])

    # Remove cms.EndPath instances from schedule
    paths_in_schedule = [path for path in process.schedule if not isinstance(path, cms.EndPath)]
    process.schedule = cms.Schedule(*paths_in_schedule)
    return process
