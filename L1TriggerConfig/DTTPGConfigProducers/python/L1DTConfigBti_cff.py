import FWCore.ParameterSet.Config as cms

BtiParametersBlock = cms.PSet(
    BtiParameters = cms.PSet(
        WEN8 = cms.int32(1),
        ACH = cms.int32(1),
        DEAD = cms.int32(31), ## dead bti parameter

        ACL = cms.int32(2),
        PTMS20 = cms.int32(1),
        PTMS21 = cms.int32(1),
        PTMS22 = cms.int32(1),
        KMAX = cms.int32(64), ## Max K param accepted

        PTMS24 = cms.int32(1),
        PTMS25 = cms.int32(1),
        PTMS26 = cms.int32(1),
        PTMS27 = cms.int32(1),
        PTMS28 = cms.int32(1),
        PTMS29 = cms.int32(1),
        SET = cms.int32(7),
        RON = cms.bool(True), ## redundant patterns flag RON

        WEN2 = cms.int32(1),
        LL = cms.int32(2), ## angular window limits for traco

        LH = cms.int32(21),
        WEN3 = cms.int32(1),
        RE43 = cms.int32(2), ## drift velocity parameter 4RE3

        WEN0 = cms.int32(1), ## wire masks

        RL = cms.int32(42),
        WEN1 = cms.int32(1),
        RH = cms.int32(61),
        LTS = cms.int32(3), ## LTS and SET for low trigger suppression

        CH = cms.int32(41),
        CL = cms.int32(22),
        Debug = cms.untracked.int32(0), ## Debug flag 

        WEN6 = cms.int32(1),
        PTMS14 = cms.int32(1),
        PTMS17 = cms.int32(1),
        PTMS16 = cms.int32(1),
        PTMS11 = cms.int32(1),
        PTMS10 = cms.int32(1),
        PTMS13 = cms.int32(1),
        PTMS12 = cms.int32(1),
        XON = cms.bool(False), ## X-patterns (time ind. K eq.) not activated

        WEN7 = cms.int32(1),
        WEN4 = cms.int32(1),
        WEN5 = cms.int32(1),
        PTMS19 = cms.int32(1),
        PTMS18 = cms.int32(1),
        PTMS31 = cms.int32(0),
        PTMS30 = cms.int32(0),
        PTMS5 = cms.int32(1),
        PTMS4 = cms.int32(1),
        PTMS7 = cms.int32(1),
        PTMS6 = cms.int32(1),
        PTMS1 = cms.int32(0),
        PTMS0 = cms.int32(0), ## pattern masks

        PTMS3 = cms.int32(0),
        PTMS2 = cms.int32(0),
        PTMS15 = cms.int32(1),
        KACCTHETA = cms.int32(1), ## BTI angular acceptance in theta view

        PTMS9 = cms.int32(1),
        PTMS8 = cms.int32(1),
        ST43 = cms.int32(42), ## drift velocity parameter 4ST3

        AC2 = cms.int32(3),
        AC1 = cms.int32(0), ## pattern acceptance AC1, AC2, ACH, ACL

        PTMS23 = cms.int32(1)
    )
)


