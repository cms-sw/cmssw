import FWCore.ParameterSet.Config as cms

egHLTOffDQMBinData = cms.PSet (
    err = cms.PSet(nr=cms.int32(101),min=cms.double(-0.5),max=cms.double(100.5)),
    #---Morse----------
    energy = cms.PSet(nr=cms.int32(60),min=cms.double(0.),max=cms.double(300.)),
    #et = cms.PSet(nr=cms.int32(11),min=cms.double(-5.),max=cms.double(105.)),
    et = cms.PSet(nr=cms.int32(80),min=cms.double(-5.),max=cms.double(155.)),
    HLTenergy = cms.PSet(nr=cms.int32(60),min=cms.double(0.),max=cms.double(300.)),
    HLTeta =cms.PSet(nr=cms.int32(30),min=cms.double(-3.),max=cms.double(3.)),
    HLTphi =cms.PSet(nr=cms.int32(16),min=cms.double(-3.14),max=cms.double(3.14)),
    #--------------
    etHigh = cms.PSet(nr=cms.int32(30),min=cms.double(25),max=cms.double(2025)),
    eta =cms.PSet(nr=cms.int32(30),min=cms.double(-3.),max=cms.double(3.)),
    phi =cms.PSet(nr=cms.int32(16),min=cms.double(-3.14),max=cms.double(3.14)),
    charge =cms.PSet(nr=cms.int32(2),min=cms.double(-1.5),max=cms.double(1.5)),
    hOverE = cms.PSet(nr=cms.int32(30),min=cms.double(0.),max=cms.double(0.1)),
    dPhiIn = cms.PSet(nr=cms.int32(10),min=cms.double(-0.15),max=cms.double(0.15)),
    dEtaIn=cms.PSet(nr=cms.int32(10),min=cms.double(-0.02),max=cms.double(0.02)),
    sigEtaEta=cms.PSet(nr=cms.int32(30),min=cms.double(-0.01),max=cms.double(0.05)),
    e2x5=cms.PSet(nr=cms.int32(24),min=cms.double(-0.1),max=cms.double(1.1)),
    e1x5=cms.PSet(nr=cms.int32(24),min=cms.double(-0.1),max=cms.double(1.1)),
    #----Morse
    #r9=cms.PSet(nr=cms.int32(24),min=cms.double(-0.1),max=cms.double(1.1)),
    minr9=cms.PSet(nr=cms.int32(24),min=cms.double(-0.1),max=cms.double(1.1)),
    maxr9=cms.PSet(nr=cms.int32(24),min=cms.double(-0.1),max=cms.double(1.1)),
    nVertex=cms.PSet(nr=cms.int32(22),min=cms.double(-1.),max=cms.double(21.)),
    deltaE=cms.PSet(nr=cms.int32(50),min=cms.double(-50.),max=cms.double(50.)),
    #---------
    isolEm=cms.PSet(nr=cms.int32(22),min=cms.double(-1),max=cms.double(10)),
    isolHad=cms.PSet(nr=cms.int32(22),min=cms.double(-1),max=cms.double(10)),
    isolPtTrks=cms.PSet(nr=cms.int32(22),min=cms.double(-1),max=cms.double(10)),
    isolNrTrks=cms.PSet(nr=cms.int32(22),min=cms.double(-1),max=cms.double(10)),
    mass = cms.PSet(nr=cms.int32(26),min=cms.double(-10.),max=cms.double(120)),
    massHigh = cms.PSet(nr=cms.int32(20),min=cms.double(0.),max=cms.double(5000)),

    eOverP = cms.PSet(nr=cms.int32(12),min=cms.double(-1.),max=cms.double(5)),
    invEInvP = cms.PSet(nr=cms.int32(10),min=cms.double(-0.2),max=cms.double(0.2)),
    etaVsPhi = cms.PSet(nrX=cms.int32(15),xMin=cms.double(-3.),xMax=cms.double(3.),
                        nrY=cms.int32(8),yMin=cms.double(-3.14),yMax=cms.double(3.14)),
)
