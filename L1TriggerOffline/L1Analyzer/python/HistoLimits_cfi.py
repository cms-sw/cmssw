import FWCore.ParameterSet.Config as cms

histoLimits = cms.PSet(
    etProfMax = cms.untracked.double(100.0),
    etaResMin = cms.untracked.double(-1.0),
    etaResMax = cms.untracked.double(1.0),
    phiNBins = cms.untracked.int32(19),
    delRNBins = cms.untracked.int32(100),
    etaMin = cms.untracked.double(-5.0),
    etResMin = cms.untracked.double(-1.0),
    etaProfMax = cms.untracked.double(5.0),
    etaMax = cms.untracked.double(5.0),
    phiMax = cms.untracked.double(3.3161256),
    phiMin = cms.untracked.double(-3.3161256),
    # Histogram limits
    etMin = cms.untracked.double(0.0),
    etaProfNBins = cms.untracked.int32(100),
    delRMax = cms.untracked.double(1.0),
    etCorMin = cms.untracked.double(0.0),
    etaCorNBins = cms.untracked.int32(100),
    phiProfMax = cms.untracked.double(3.1415927),
    phiResMax = cms.untracked.double(1.0),
    etCorMax = cms.untracked.double(100.0),
    phiCorMin = cms.untracked.double(-3.1415927),
    phiCorMax = cms.untracked.double(3.1415927),
    etaResNBins = cms.untracked.int32(100),
    phiProfMin = cms.untracked.double(-3.1415927),
    # Bins and limits for the resolutions
    etResNBins = cms.untracked.int32(150),
    phiProfNBins = cms.untracked.int32(100),
    # Bins and limits for 2D correlations
    etCorNBins = cms.untracked.int32(100),
    delRMin = cms.untracked.double(0.0),
    etaNBins = cms.untracked.int32(22),
    etaCorMin = cms.untracked.double(-5.0),
    etaProfMin = cms.untracked.double(-5.0),
    etMax = cms.untracked.double(100.0),
    phiCorNBins = cms.untracked.int32(100),
    # Number of bins
    etNBins = cms.untracked.int32(50),
    etProfMin = cms.untracked.double(0.0),
    etaCorMax = cms.untracked.double(5.0),
    # Bins and limits for profiles
    etProfNBins = cms.untracked.int32(100),
    etResMax = cms.untracked.double(2.0),
    phiResNBins = cms.untracked.int32(100),
    phiResMin = cms.untracked.double(-1.0)
)

