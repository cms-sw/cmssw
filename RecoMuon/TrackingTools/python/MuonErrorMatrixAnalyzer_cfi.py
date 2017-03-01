# The following comments couldn't be translated into the new config version:

# if radius is not 0, z need to be specified to bound the cylinder. Z=0 means NO bound

import FWCore.ParameterSet.Config as cms

errorMatrix_default = cms.PSet(
    PtBins = cms.vdouble(), ## set NPt=0 and the vector of double for variable size binning

    minPhi = cms.string('-Pi'),
    minPt = cms.double(0.0),
    maxEta = cms.double(2.5),
    maxPhi = cms.string('Pi'),
    minEta = cms.double(0.0),
    EtaBins = cms.vdouble(), ## set NEta=0 and the vector of double for variable size binning

    NEta = cms.int32(10),
    NPt = cms.int32(10),
    maxPt = cms.double(200.0),
    NPhi = cms.int32(1)
)
muonErrorMatrixAnalyzer = cms.EDAnalyzer("MuonErrorMatrixAnalyzer",
    errorMatrix_Reported_pset = cms.PSet(
        errorMatrix_default,
        action = cms.string('constructor'),
        rootFileName = cms.string('errorMatrix_Reported.root')
    ),
    associatorName = cms.string('trackAssociatorByPosition'),
    errorMatrix_Pull_pset = cms.PSet(
        errorMatrix_default,
        action = cms.string('constructor'),
        rootFileName = cms.string('errorMatrix_Pull.root')
    ),
    gaussianPullFitRange = cms.untracked.double(2.0),
    # if radius is not 0, a propagator needs to be specified to go to that radius
    propagatorName = cms.string('SteppingHelixPropagatorAlong'),
    trackLabel = cms.InputTag("standAloneMuons","UpdatedAtVtx"),
    plotFileName = cms.string('controlErrorMatrixAnalyzer.root'), ##empty string. no root file

    radius = cms.double(0.0),
    z = cms.double(0.0),
    trackingParticleLabel = cms.InputTag("trackingParticles"),
    errorMatrix_Residual_pset = cms.PSet(
        errorMatrix_default,
        action = cms.string('constructor'),
        rootFileName = cms.string('errorMatrix_Residual.root')
    )
)



