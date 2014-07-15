import FWCore.ParameterSet.Config as cms

from DQM.EcalCommon.CommonParams_cfi import *

emptyLSLimit = 3

ecalLedTask = cms.untracked.PSet(
    params = cms.untracked.PSet(
        emptyLSLimit = cms.untracked.int32(emptyLSLimit),
        ledWavelengths = ecaldqmLedWavelengths
    ),
    MEs = cms.untracked.PSet(
        PNAmplitude = cms.untracked.PSet(
            path = cms.untracked.string('EcalEndcap/EELedTask/Led%(wl)s/PN/Gain16/EELDT PNs amplitude %(sm)s G16 L%(wl)s'),
            otype = cms.untracked.string('EESMMEM'),
            multi = cms.untracked.PSet(
                wl = ecaldqmLedWavelengths
            ),
            kind = cms.untracked.string('TProfile'),
            btype = cms.untracked.string('Crystal'),
            description = cms.untracked.string('Mean led pulse amplitude in the PN diodes. In general, a PN channel is filled only when a led pulse was observed in the crystals that are associated to the diode. When no led signal was observed for longer than ' + str(emptyLSLimit) + ' lumi sections, the channels start to get filled with 0 amplitude, causing the mean to drop.')
        ),
        Occupancy = cms.untracked.PSet(
            path = cms.untracked.string('EcalEndcap/EEOccupancyTask/EEOT led digi occupancy%(suffix)s'),
            kind = cms.untracked.string('TH2F'),
            otype = cms.untracked.string('EE2P'),
            btype = cms.untracked.string('SuperCrystal'),
            description = cms.untracked.string('Led signal digi occupancy. Channels are filled whenever the DCC event type is set to LED.')
        ),
        SignalRate = cms.untracked.PSet(
            path = cms.untracked.string('EcalEndcap/EELedTask/Led%(wl)s/EELDT signal rate L%(wl)s'),
            kind = cms.untracked.string('TProfile'),
            otype = cms.untracked.string('EE'),
            btype = cms.untracked.string('DCC'),
            description = cms.untracked.string('Fraction of laser events with measurable led pulse.'),
            multi = cms.untracked.int32(2)
        ),
        Shape = cms.untracked.PSet(
            multi = cms.untracked.PSet(
                wl = ecaldqmLedWavelengths
            ),
            yaxis = cms.untracked.PSet(
                high = cms.untracked.double(10.0),
                nbins = cms.untracked.int32(10),
                low = cms.untracked.double(0.0)
            ),
            kind = cms.untracked.string('TProfile2D'),
            otype = cms.untracked.string('EESM'),
            btype = cms.untracked.string('SuperCrystal'),
            path = cms.untracked.string('EcalEndcap/EELedTask/Led%(wl)s/EELDT shape %(sm)s L%(wl)s'),
            description = cms.untracked.string('Led mean pulse shape. One slice corresponds to one readout tower (5x5 crystals). In general, a slice is filled only when a led pulse was observed in the tower. When no led signal was observed for longer than ' + str(emptyLSLimit) + ' lumi sections, the slices start to get filled with 0 amplitude, causing the shape to flatten.')
        ),
        Amplitude = cms.untracked.PSet(
            path = cms.untracked.string('EcalEndcap/EELedTask/Led%(wl)s/EELDT amplitude %(sm)s L%(wl)s'),
            otype = cms.untracked.string('EESM'),
            multi = cms.untracked.PSet(
                wl = ecaldqmLedWavelengths
            ),
            kind = cms.untracked.string('TProfile2D'),
            btype = cms.untracked.string('Crystal'),
            description = cms.untracked.string('2D distribution of the mean led amplitude. In general, a channel is filled only when a led pulse was observed in it. When no led signal was observed for longer than ' + str(emptyLSLimit) + ' lumi sections, the channels start to get filled with 0 amplitude, causing the mean to drop.')
        ),
        AOverP = cms.untracked.PSet(
            path = cms.untracked.string('EcalEndcap/EELedTask/Led%(wl)s/EELDT amplitude over PN %(sm)s L%(wl)s'),
            otype = cms.untracked.string('EESM'),
            multi = cms.untracked.PSet(
                wl = ecaldqmLedWavelengths
            ),
            kind = cms.untracked.string('TProfile2D'),
            btype = cms.untracked.string('Crystal'),
            description = cms.untracked.string('2D distribution of the mean APD/PN value (event mean of per-event ratio).')
        ),
        Timing = cms.untracked.PSet(
            path = cms.untracked.string('EcalEndcap/EELedTask/Led%(wl)s/EELDT timing %(sm)s L%(wl)s'),
            otype = cms.untracked.string('EESM'),
            multi = cms.untracked.PSet(
                wl = ecaldqmLedWavelengths
            ),
            kind = cms.untracked.string('TProfile2D'),
            btype = cms.untracked.string('Crystal'),
            description = cms.untracked.string('2D distribution of the mean led timing. Z scale is in LHC clocks. Due to the difference in pulse shape between led and physics events, fit-based reconstruction is not completely reliable in extracting the timing. In general, a channel is filled only when a led pulse was observed in it. When no led signal was observed for longer than ' + str(emptyLSLimit) + ' lumi sections, the channels start to get filled with 0 amplitude, causing the timing to spread randomly.')
        ),
        AmplitudeSummary = cms.untracked.PSet(
            path = cms.untracked.string('EcalEndcap/EELedTask/Led%(wl)s/EELDT amplitude map L%(wl)s%(suffix)s'),
            otype = cms.untracked.string('EE2P'),
            multi = cms.untracked.int32(4),
            kind = cms.untracked.string('TProfile2D'),
            btype = cms.untracked.string('SuperCrystal'),
            description = cms.untracked.string('2D distribution of the mean led amplitude. In general, a channel is filled only when a led pulse was observed in it. When no led signal was observed for longer than ' + str(emptyLSLimit) + ' lumi sections, the channels start to get filled with 0 amplitude, causing the mean to drop.')
        )
    )
)
