import FWCore.ParameterSet.Config as cms

from CommonTools.PileupAlgos.Puppi_cff import *
# from CommonTools.PileupAlgos.PhotonPuppi_cff import setupPuppiPhoton

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

## puppi met
    process.puppiForMET = cms.EDProducer("PuppiPhoton",
                                         candName       = cms.InputTag('packedPFCandidates'),
                                         puppiCandName  = cms.InputTag('puppi'),
                                         photonName     = cms.InputTag('slimmedPhotons'),
                                         photonId       = cms.InputTag("egmPhotonIDs:cutBasedPhotonID_PHYS14_PU20bx25_V2p1-standalone-loose"),
                                         pt             = cms.double(10),
                                         useRefs        = cms.bool(True),
                                         dRMatch        = cms.vdouble(10,10,10,10),
                                         pdgids         = cms.vint32 (22,11,211,130),
                                         weight         = cms.double(1.),
                                         useValueMap    = cms.bool(False),
                                         weightsName    = cms.InputTag('puppi'),
                                         )
    process.puppiForMET.puppiCandName    = 'puppiMerged'



def makePuppiesFromMiniAOD( process ):

    process.load('CommonTools.PileupAlgos.Puppi_cff')

    process.puppi.candName = cms.InputTag('packedPFCandidates')
    process.puppi.vertexName = cms.InputTag('offlineSlimmedPrimaryVertices')
# kind of ugly, is there a better way to do this?

    process.pfNoLepPUPPI = cms.EDFilter("CandPtrSelector", src = cms.InputTag("packedPFCandidates"), cut =  cms.string("abs(pdgId) != 13 && abs(pdgId) != 11 && abs(pdgId) != 15"))
    process.pfLeptonsPUPPET   = cms.EDFilter("CandPtrSelector", src = cms.InputTag("packedPFCandidates"), cut = cms.string("abs(pdgId) == 13 || abs(pdgId) == 11 || abs(pdgId) == 15"))

    process.puppiNoLep = process.puppi.clone()
    process.puppiNoLep.candName = cms.InputTag('pfNoLepPUPPI') 

    process.puppiMerged = cms.EDProducer("CandViewMerger",src = cms.VInputTag( 'puppiNoLep','pfLeptonsPUPPET'))

## puppi met
    process.puppiForMET = cms.EDProducer("PuppiPhoton",
                                         candName       = cms.InputTag('packedPFCandidates'),
                                         puppiCandName  = cms.InputTag('puppi'),
                                         photonName     = cms.InputTag('slimmedPhotons'),
                                         photonId       = cms.InputTag("egmPhotonIDs:cutBasedPhotonID_PHYS14_PU20bx25_V2p1-standalone-loose"),
                                         pt             = cms.double(10),
                                         useRefs        = cms.bool(True),
                                         dRMatch        = cms.vdouble(10,10,10,10),
                                         pdgids         = cms.vint32 (22,11,211,130),
                                         weight         = cms.double(1.),
                                         useValueMap    = cms.bool(False),
                                         weightsName    = cms.InputTag('puppi'),
                                         )
    process.puppiForMET.puppiCandName    = 'puppiMerged'
    
