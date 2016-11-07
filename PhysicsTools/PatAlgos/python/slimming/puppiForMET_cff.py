import FWCore.ParameterSet.Config as cms

from CommonTools.PileupAlgos.Puppi_cff import *
from CommonTools.PileupAlgos.PhotonPuppi_cff        import setupPuppiPhoton,setupPuppiPhotonMiniAOD

import PhysicsTools.PatAlgos.tools.helpers as configtools

def makePuppies( process ):
    
    process.load('CommonTools.PileupAlgos.Puppi_cff')
    patAlgosToolsTask = configtools.getPatAlgosToolsTask(process)
    patAlgosToolsTask.add(process.puppi)
    process.pfNoLepPUPPI = cms.EDFilter("PdgIdCandViewSelector",
                                        src = cms.InputTag("particleFlow"), 
                                        pdgId = cms.vint32( 1,2,22,111,130,310,2112,211,-211,321,-321,999211,2212,-2212 )
                                        )
    patAlgosToolsTask.add(process.pfNoLepPUPPI)
    process.pfLeptonsPUPPET = cms.EDFilter("PdgIdCandViewSelector",
                                           src = cms.InputTag("particleFlow"),
                                           pdgId = cms.vint32(-11,11,-13,13),
                                           )
    patAlgosToolsTask.add(process.pfLeptonsPUPPET)

    process.puppiNoLep = process.puppi.clone()
    patAlgosToolsTask.add(process.puppiNoLep)
    process.puppiNoLep.candName = cms.InputTag('pfNoLepPUPPI') 
    process.puppiMerged = cms.EDProducer("CandViewMerger",src = cms.VInputTag( 'puppiNoLep','pfLeptonsPUPPET'))
    patAlgosToolsTask.add(process.puppiMerged)
    process.load('CommonTools.PileupAlgos.PhotonPuppi_cff')
    patAlgosToolsTask.add(process.puppiPhoton)
    process.puppiForMET = process.puppiPhoton.clone()
    patAlgosToolsTask.add(process.puppiForMET)
    #Line below replaces reference linking wiht delta R matching because the puppi references after merging are not consistent with those of the original PF collection
    process.puppiForMET.useRefs          = False
    #Line below points puppi MET to puppi no lepton which increases the response
    process.puppiForMET.puppiCandName    = 'puppiMerged'


def makePuppiesFromMiniAOD( process, createScheduledSequence=False ):
    process.load('CommonTools.PileupAlgos.Puppi_cff')
    patAlgosToolsTask = configtools.getPatAlgosToolsTask(process)
    patAlgosToolsTask.add(process.puppi)
    process.puppi.candName = cms.InputTag('packedPFCandidates')
    process.puppi.clonePackedCands = cms.bool(True)
    process.puppi.vertexName = cms.InputTag('offlineSlimmedPrimaryVertices')
    process.puppi.useExistingWeights = cms.bool(True)
    process.pfNoLepPUPPI = cms.EDFilter("CandPtrSelector", src = cms.InputTag("packedPFCandidates"), cut =  cms.string("abs(pdgId) != 13 && abs(pdgId) != 11 && abs(pdgId) != 15"))
    patAlgosToolsTask.add(process.pfNoLepPUPPI)
    process.pfLeptonsPUPPET   = cms.EDFilter("CandPtrSelector", src = cms.InputTag("packedPFCandidates"), cut = cms.string("abs(pdgId) == 13 || abs(pdgId) == 11 || abs(pdgId) == 15"))
    patAlgosToolsTask.add(process.pfLeptonsPUPPET)
    process.puppiNoLep = process.puppi.clone()
    patAlgosToolsTask.add(process.puppiNoLep)
    process.puppiNoLep.candName = cms.InputTag('pfNoLepPUPPI') 
    process.puppiNoLep.useWeightsNoLep = cms.bool(True)
    process.puppiNoLep.useExistingWeights = cms.bool(True)
    process.puppiMerged = cms.EDProducer("CandViewMerger",src = cms.VInputTag( 'puppiNoLep','pfLeptonsPUPPET'))
    patAlgosToolsTask.add(process.puppiMerged)
    process.load('CommonTools.PileupAlgos.PhotonPuppi_cff')
    patAlgosToolsTask.add(process.puppiPhoton)
    process.puppiForMET = process.puppiPhoton.clone()
    patAlgosToolsTask.add(process.puppiForMET)
    process.puppiForMET.candName = cms.InputTag('packedPFCandidates')
    process.puppiForMET.photonName = cms.InputTag('slimmedPhotons')
    process.puppiForMET.runOnMiniAOD = cms.bool(True)
    setupPuppiPhotonMiniAOD(process)
    patAlgosToolsTask.add(process.egmPhotonIDsTask)
    #Line below replaces reference linking wiht delta R matching because the puppi references after merging are not consistent with those of the original packed candidate collection
    process.puppiForMET.useRefs          = False
    #Line below points puppi MET to puppi no lepton which increases the response
    process.puppiForMET.puppiCandName    = 'puppiMerged'

    #making a sequence for people running the MET tool in scheduled mode
    if createScheduledSequence:
        puppiMETSequence = cms.Sequence(process.puppi*process.pfLeptonsPUPPET*process.pfNoLepPUPPI*process.puppiNoLep*process.puppiMerged*process.puppiForMET)
        setattr(process, "puppiMETSequence", puppiMETSequence)
