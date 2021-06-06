import FWCore.ParameterSet.Config as cms

from CommonTools.PileupAlgos.Puppi_cff import *

from PhysicsTools.PatAlgos.tools.helpers import getPatAlgosToolsTask, addToProcessAndTask

def makePuppies( process ):
    task = getPatAlgosToolsTask(process)
    process.load('CommonTools.PileupAlgos.Puppi_cff')
    task.add(process.puppi)
    task.add(process.puppiNoLep)

def makePuppiesFromMiniAOD( process, createScheduledSequence=False ):
    task = getPatAlgosToolsTask(process)
    process.load('CommonTools.PileupAlgos.Puppi_cff')
    task.add(process.puppi)
    process.puppi.candName = 'packedPFCandidates'
    process.puppi.clonePackedCands = True
    process.puppi.vertexName = 'offlineSlimmedPrimaryVertices'
    process.puppi.useExistingWeights = True
    task.add(process.puppiNoLep)
    process.puppiNoLep.candName = 'packedPFCandidates'
    process.puppiNoLep.clonePackedCands = True
    process.puppiNoLep.vertexName = 'offlineSlimmedPrimaryVertices'
    process.puppiNoLep.useExistingWeights = True

    #making a sequence for people running the MET tool in scheduled mode
    if createScheduledSequence:
        puppiMETSequence = cms.Sequence(process.puppi*process.puppiNoLep)
        setattr(process, "puppiMETSequence", puppiMETSequence)
