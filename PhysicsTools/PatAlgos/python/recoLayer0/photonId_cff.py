import FWCore.ParameterSet.Config as cms

from RecoEgamma.PhotonIdentification.photonId_cff import *
##If you had a mind to, you could clone the sequence here
##and change the PhotonID cuts.  Or add more.
##Currently it is simply the default.

##Now copy these Ids
patPhotonIds = cms.EDFilter("CandManyValueMapsSkimmerBool",
    collection = cms.InputTag("allLayer0Photons"),
    backrefs   = cms.InputTag("allLayer0Photons"),
    associations = cms.VInputTag(
        cms.InputTag("PhotonIDProd:PhotonCutBasedIDLoose"),
        cms.InputTag("PhotonIDProd:PhotonCutBasedIDTight")
    ),
    failSilently = cms.untracked.bool(False),
)


## define the sequence, so we have consistent naming conventions
patPhotonId = cms.Sequence( photonIDSequence*patPhotonIds )

