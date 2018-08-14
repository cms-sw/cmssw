import FWCore.ParameterSet.Config as cms

from CommonTools.PileupAlgos.Puppi_cff import *
from CommonTools.PileupAlgos.PhotonPuppi_cff        import setupPuppiPhoton,setupPuppiPhotonMiniAOD

from PhysicsTools.PatAlgos.tools.helpers import getPatAlgosToolsTask, addToProcessAndTask

def makePuppies( process ):

    task = getPatAlgosToolsTask(process)

    process.load('CommonTools.PileupAlgos.Puppi_cff')
    task.add(process.puppi)
    process.pfNoLepPUPPI = cms.EDFilter("PdgIdCandViewSelector",
                                        src = cms.InputTag("particleFlow"), 
                                        pdgId = cms.vint32( 1,2,22,111,130,310,2112,211,-211,321,-321,999211,2212,-2212 )
                                        )
    task.add(process.pfNoLepPUPPI)
    process.pfLeptonsPUPPET = cms.EDFilter("PdgIdCandViewSelector",
                                           src = cms.InputTag("particleFlow"),
                                           pdgId = cms.vint32(-11,11,-13,13),
                                           )
    task.add(process.pfLeptonsPUPPET)

    addToProcessAndTask('puppiNoLep', process.puppi.clone(), process, task)
    process.puppiNoLep.candName = cms.InputTag('pfNoLepPUPPI') 
    process.puppiMerged = cms.EDProducer("CandViewMerger",src = cms.VInputTag( 'puppiNoLep','pfLeptonsPUPPET'))
    task.add(process.puppiMerged)
    process.load('CommonTools.PileupAlgos.PhotonPuppi_cff')
    task.add(process.puppiPhoton)
    addToProcessAndTask('puppiForMET', process.puppiPhoton.clone(), process, task)
    #Line below replaces reference linking wiht delta R matching because the puppi references after merging are not consistent with those of the original PF collection
    process.puppiForMET.useRefs          = False
    #Line below points puppi MET to puppi no lepton which increases the response
    process.puppiForMET.puppiCandName    = 'puppiMerged'


def makePuppiesFromMiniAOD( process, createScheduledSequence=False ):
    task = getPatAlgosToolsTask(process)
    process.load('CommonTools.PileupAlgos.Puppi_cff')
    task.add(process.puppi)
    process.puppi.candName = cms.InputTag('packedPFCandidates')
    process.puppi.clonePackedCands = cms.bool(True)
    process.puppi.vertexName = cms.InputTag('offlineSlimmedPrimaryVertices')
    process.puppi.useExistingWeights = cms.bool(True)
    process.pfNoLepPUPPI = cms.EDFilter("CandPtrSelector", src = cms.InputTag("packedPFCandidates"), cut =  cms.string("abs(pdgId) != 13 && abs(pdgId) != 11 && abs(pdgId) != 15"))
    task.add(process.pfNoLepPUPPI)
    process.pfLeptonsPUPPET   = cms.EDFilter("CandPtrSelector", src = cms.InputTag("packedPFCandidates"), cut = cms.string("abs(pdgId) == 13 || abs(pdgId) == 11 || abs(pdgId) == 15"))
    task.add(process.pfLeptonsPUPPET)
    addToProcessAndTask('puppiNoLep', process.puppi.clone(), process, task)
    process.puppiNoLep.candName = cms.InputTag('pfNoLepPUPPI') 
    process.puppiNoLep.useWeightsNoLep = cms.bool(True)
    process.puppiNoLep.useExistingWeights = cms.bool(True)
    process.puppiMerged = cms.EDProducer("CandViewMerger",src = cms.VInputTag( 'puppiNoLep','pfLeptonsPUPPET'))
    task.add(process.puppiMerged)
    process.load('CommonTools.PileupAlgos.PhotonPuppi_cff')
    task.add(process.puppiPhoton)
    addToProcessAndTask('puppiForMET', process.puppiPhoton.clone(), process, task)
    process.puppiForMET.candName = cms.InputTag('packedPFCandidates')
    process.puppiForMET.photonName = cms.InputTag('slimmedPhotons')
    process.puppiForMET.runOnMiniAOD = cms.bool(True)
    setupPuppiPhotonMiniAOD(process)
    task.add(process.egmPhotonIDTask)
    #Line below replaces reference linking wiht delta R matching because the puppi references after merging are not consistent with those of the original packed candidate collection
    process.puppiForMET.useRefs          = False
    #Line below points puppi MET to puppi no lepton which increases the response
    process.puppiForMET.puppiCandName    = 'puppiMerged'

    #making a sequence for people running the MET tool in scheduled mode
    if createScheduledSequence:
        puppiMETSequence = cms.Sequence(process.puppi*process.pfLeptonsPUPPET*process.pfNoLepPUPPI*process.puppiNoLep*process.puppiMerged*process.puppiForMET)
        setattr(process, "puppiMETSequence", puppiMETSequence)
