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
    from CommonTools.ParticleFlow.pfCHS_cff import packedPrimaryVertexAssociationJME
    setattr(process, "packedPrimaryVertexAssociationJME", packedPrimaryVertexAssociationJME.clone())
    task.add(process.packedPrimaryVertexAssociationJME)
    process.load('CommonTools.PileupAlgos.Puppi_cff')
    task.add(process.puppi)
    process.puppi.candName = 'packedPFCandidates'
    process.puppi.clonePackedCands = True
    process.puppi.vertexName = 'offlineSlimmedPrimaryVertices'
    process.puppi.useExistingWeights = True
    process.puppi.vertexAssociation = 'packedPrimaryVertexAssociationJME:original'
    task.add(process.puppiNoLep)
    process.puppiNoLep.candName = 'packedPFCandidates'
    process.puppiNoLep.clonePackedCands = True
    process.puppiNoLep.vertexName = 'offlineSlimmedPrimaryVertices'
    process.puppiNoLep.useExistingWeights = True
    process.puppiNoLep.vertexAssociation = 'packedPrimaryVertexAssociationJME:original'

    #making a sequence for people running the MET tool in scheduled mode
    if createScheduledSequence:
        puppiMETTask = cms.Task(process.packedPrimaryVertexAssociationJME, process.puppi, process.puppiNoLep)
        setattr(process, "puppiMETTask", puppiMETTask)
        puppiMETSequence = cms.Sequence(puppiMETTask)
        setattr(process, "puppiMETSequence", puppiMETSequence)
