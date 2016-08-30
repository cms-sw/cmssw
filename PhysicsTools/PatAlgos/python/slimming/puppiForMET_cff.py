import FWCore.ParameterSet.Config as cms

from CommonTools.PileupAlgos.Puppi_cff import *
from CommonTools.PileupAlgos.PhotonPuppi_cff        import setupPuppiPhoton

def makePuppies( process ):

    process.load('CommonTools.PileupAlgos.Puppi_cff')

    process.pfNoLepPUPPI = cms.EDFilter("PdgIdCandViewSelector",
                                        src = cms.InputTag("particleFlow"), 
                                        pdgId = cms.vint32( 1,2,22,111,130,310,2112,211,-211,321,-321,999211,2212,-2212 )
                                        )
    process.pfLeptonsPUPPET = cms.EDFilter("PdgIdCandViewSelector",
                                           src = cms.InputTag("particleFlow"),
                                           pdgId = cms.vint32(-11,11,-13,13),
                                           )
    process.puppiNoLep = process.puppi.clone()
    process.puppiNoLep.candName = cms.InputTag('pfNoLepPUPPI') 

    process.puppiMerged = cms.EDProducer("CandViewMerger",src = cms.VInputTag( 'puppiNoLep','pfLeptonsPUPPET'))
    process.load('CommonTools.PileupAlgos.PhotonPuppi_cff')
    process.puppiForMET = process.puppiPhoton.clone()
    #process.puppiForMET.puppiCandName    = 'puppiMerged'
    


def makePuppiesFromMiniAOD( process ):
    process.load('CommonTools.PileupAlgos.Puppi_cff')
    process.puppi.candName = cms.InputTag('packedPFCandidates')
    process.puppi.vertexName = cms.InputTag('offlineSlimmedPrimaryVertices')
    process.pfNoLepPUPPI = cms.EDFilter("CandPtrSelector", src = cms.InputTag("packedPFCandidates"), cut =  cms.string("abs(pdgId) != 13 && abs(pdgId) != 11 && abs(pdgId) != 15"))
    process.pfLeptonsPUPPET   = cms.EDFilter("CandPtrSelector", src = cms.InputTag("packedPFCandidates"), cut = cms.string("abs(pdgId) == 13 || abs(pdgId) == 11 || abs(pdgId) == 15"))
    process.puppiNoLep = process.puppi.clone()
    process.puppiNoLep.candName = cms.InputTag('pfNoLepPUPPI') 
    process.puppiNoLep.useWeightsNoLep = cms.bool(True)
    process.puppiMerged = cms.EDProducer("CandViewMerger",src = cms.VInputTag( 'puppiNoLep','pfLeptonsPUPPET'))
    process.load('CommonTools.PileupAlgos.PhotonPuppi_cff')
    process.puppiForMET = process.puppiPhoton.clone()
    setupPuppiPhoton(process)
    #Line below doesn't work because of an issue with references in MiniAOD
    #process.puppiForMET.puppiCandName    = 'puppiMerged'
    
