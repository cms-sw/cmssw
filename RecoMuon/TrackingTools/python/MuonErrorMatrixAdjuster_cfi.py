import FWCore.ParameterSet.Config as cms

#values for correction
from RecoMuon.TrackingTools.MuonErrorMatrixValues_cff import *
muonErrorMatrixAdjuster = cms.EDProducer("MuonErrorMatrixAdjuster",
    #if replace is true this means error matrix from reco is replaced by new method of error matrix (reco minus sim of parameters to get the error)
    #if replace is false this means the error matrix from reco is rescaled by a factor
    rescale = cms.bool(True),
    #this is the root file with the TProfile 3D in it of the track collection. Make sure it corresponds to the boolean above
    errorMatrix_pset = cms.PSet(
        # use either one of the two following lines
        #string rootFileName = "errorMatrix_ScaleFactor.root"
        MuonErrorMatrixValues,
        action = cms.string('use')
    ),
    instanceName = cms.string(''),
    rechitLabel = cms.InputTag("standAloneMuons"),
    trackLabel = cms.InputTag("standAloneMuons","UpdatedAtVtx")
)



