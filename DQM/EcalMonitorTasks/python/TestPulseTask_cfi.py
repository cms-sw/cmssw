import FWCore.ParameterSet.Config as cms

from DQM.EcalCommon.CommonParams_cfi import *

ecalTestPulseTask = cms.untracked.PSet(
    params = cms.untracked.PSet(
        MGPAGains = ecaldqmMGPAGains,
        MGPAGainsPN = ecaldqmMGPAGainsPN
    ),
    MEs = cms.untracked.PSet(
        PNAmplitude = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sTestPulseTask/PN/Gain%(pngain)s/%(prefix)sTPT PNs amplitude %(sm)s G%(pngain)s'),
            otype = cms.untracked.string('SMMEM'),
            multi = cms.untracked.PSet(
                pngain = ecaldqmMGPAGainsPN
            ),
            kind = cms.untracked.string('TProfile'),
            btype = cms.untracked.string('Crystal'),
            description = cms.untracked.string('Test pulse amplitude in the PN diodes.')
        ),
        Amplitude = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sTestPulseTask/Gain%(gain)s/%(prefix)sTPT amplitude %(sm)s G%(gain)s'),
            otype = cms.untracked.string('SM'),
            multi = cms.untracked.PSet(
                gain = ecaldqmMGPAGains
            ),
            kind = cms.untracked.string('TProfile2D'),
            btype = cms.untracked.string('Crystal'),
            description = cms.untracked.string('Test pulse amplitude.')
        ),
        Shape = cms.untracked.PSet(
            multi = cms.untracked.PSet(
                gain = ecaldqmMGPAGains
            ),
            yaxis = cms.untracked.PSet(
                high = cms.untracked.double(10.0),
                nbins = cms.untracked.int32(10),
                low = cms.untracked.double(0.0)
            ),
            kind = cms.untracked.string('TProfile2D'),
            otype = cms.untracked.string('SM'),
            btype = cms.untracked.string('SuperCrystal'),
            path = cms.untracked.string('%(subdet)s/%(prefix)sTestPulseTask/Gain%(gain)s/%(prefix)sTPT shape %(sm)s G%(gain)s'),
            description = cms.untracked.string('Test pulse shape. One slice corresponds to one readout tower (5x5 crystals).')
        ),
        Occupancy = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sOccupancyTask/%(prefix)sOT test pulse digi occupancy%(suffix)s'),
            kind = cms.untracked.string('TH2F'),
            otype = cms.untracked.string('Ecal3P'),
            btype = cms.untracked.string('SuperCrystal'),
            description = cms.untracked.string('Occupancy in test pulse events.')
        )
    )
)

