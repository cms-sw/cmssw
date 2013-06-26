import FWCore.ParameterSet.Config as cms

generator = cms.EDFilter("AMPTGeneratorFilter",
    diquarky = cms.double(0.0),
    diquarkx = cms.double(0.0),
    diquarkpx = cms.double(7.0),
    ntmax = cms.int32(1000),
    dpcoal = cms.double(1000000.0),
    diquarkembedding = cms.int32(0),
    maxmiss = cms.int32(1000),
    ktkick = cms.int32(1),
    mu = cms.double(3.2264),
    quenchingpar = cms.double(2.0),
    popcornpar = cms.double(1.0),
    drcoal = cms.double(1000000.0),
    amptmode = cms.int32(1),
    izpc = cms.int32(0),
    popcornmode = cms.bool(True),
    minijetpt = cms.double(-7.0),
    ks0decay = cms.bool(False),
    alpha = cms.double(0.47140452),
    dt = cms.double(0.2),
    rotateEventPlane = cms.bool(True),
    shadowingmode = cms.bool(True),
    diquarkpy = cms.double(0.0),
    deuteronfactor = cms.int32(5),
    stringFragB = cms.double(0.9),#default value in Hijing. Good for pA
    quenchingmode = cms.bool(False),
    stringFragA = cms.double(0.5),
    deuteronmode = cms.int32(0),
    doInitialAndFinalRadiation = cms.int32(3),
    phidecay = cms.bool(True),
    deuteronxsec = cms.int32(1),
    pthard = cms.double(2.0),
    firstRun = cms.untracked.uint32(1),
    frame = cms.string('CMS'),
    targ = cms.string('P'),
    izp = cms.int32(82),
    bMin = cms.double(0),
    firstEvent = cms.untracked.uint32(1),
    izt = cms.int32(1),
    proj = cms.string('A'),
    comEnergy = cms.double(5020.0),
    iat = cms.int32(1),
    bMax = cms.double(15),
    iap = cms.int32(208)
)

configurationMetadata = cms.untracked.PSet(
    version = cms.untracked.string('$Revision: 1.1 $'),
    name = cms.untracked.string('$Source: /local/reps/CMSSW/CMSSW/Configuration/Generator/python/AMPT_PPb_5020GeV_MinimumBias_cfi.py,v $'),
    annotation = cms.untracked.string('AMPT PPb 5020 GeV Minimum Bias')
)

