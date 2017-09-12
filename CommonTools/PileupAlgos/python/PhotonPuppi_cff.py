import FWCore.ParameterSet.Config as cms
from PhysicsTools.SelectorUtils.tools.vid_id_tools import *

puppiPhoton = cms.EDProducer("PuppiPhoton",
                             candName       = cms.InputTag('particleFlow'),
                             puppiCandName  = cms.InputTag('puppi'),
                             photonName     = cms.InputTag('reducedEgamma','reducedGedPhotons'),
			     recoToPFMap    = cms.InputTag("reducedEgamma","reducedPhotonPfCandMap"),
                             photonId       = cms.InputTag("egmPhotonIDs:cutBasedPhotonID-Spring15-25ns-V1-standalone-loose"), 
                             pt             = cms.double(10),
                             eta            = cms.double(2.5),
			     runOnMiniAOD   = cms.bool(False),
                             useRefs        = cms.bool(True),
                             dRMatch        = cms.vdouble(0.005,0.005,0.005,0.005),
                             pdgids         = cms.vint32 (22,11,211,130),
                             weight         = cms.double(1.),
                             useValueMap    = cms.bool(False),
                             weightsName    = cms.InputTag('puppi'),
                             )


def setupPuppiPhoton(process):
    my_id_modules = ['RecoEgamma.PhotonIdentification.Identification.cutBasedPhotonID_Spring15_25ns_V1_cff']
    switchOnVIDPhotonIdProducer(process, DataFormat.AOD)
    for idmod in my_id_modules:
        setupAllVIDIdsInModule(process,idmod,setupVIDPhotonSelection)


def setupPuppiPhotonMiniAOD(process):
    my_id_modules = ['RecoEgamma.PhotonIdentification.Identification.cutBasedPhotonID_Spring15_25ns_V1_cff']
    switchOnVIDPhotonIdProducer(process, DataFormat.MiniAOD)
    for idmod in my_id_modules:
        setupAllVIDIdsInModule(process,idmod,setupVIDPhotonSelection)


#puppiPhotonSeq = cms.Sequence(egmPhotonIDSequence*puppiPhoton)
