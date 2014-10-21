import FWCore.ParameterSet.Config as cms

dccSizeBinEdges = []
for i in range(11) :
    dccSizeBinEdges.append(0.608 / 10. * i)
for i in range(11, 79) :
    dccSizeBinEdges.append(0.608 * (i - 10.))

ecalSelectiveReadoutTask = cms.untracked.PSet(
    params = cms.untracked.PSet(
        DCCZS1stSample = cms.untracked.int32(2),
        useCondDb = cms.untracked.bool(False),
        ZSFIRWeights = cms.untracked.vdouble(-0.374, -0.374, -0.3629, 0.2721, 0.4681, 0.3707)
    ),
    MEs = cms.untracked.PSet(
        HighIntOutput = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sSelectiveReadoutTask/%(prefix)sSRT high interest ZS filter output%(suffix)s'),
            kind = cms.untracked.string('TH1F'),
            otype = cms.untracked.string('Ecal3P'),
            xaxis = cms.untracked.PSet(
                high = cms.untracked.double(60.0),
                nbins = cms.untracked.int32(100),
                low = cms.untracked.double(-60.0),
                title = cms.untracked.string('ADC counts*4')
            ),
            btype = cms.untracked.string('User'),
            description = cms.untracked.string('Output of the ZS filter for high interest towers.')
        ),
        ZS1Map = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sSelectiveReadoutTask/Counters/%(prefix)sSRT tower ZS1 counter%(suffix)s'),
            kind = cms.untracked.string('TH2F'),
            otype = cms.untracked.string('Ecal3P'),
            btype = cms.untracked.string('SuperCrystal'),
            description = cms.untracked.string('Tower occupancy with ZS1 flags.')
        ),
        FullReadoutMap = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sSelectiveReadoutTask/Counters/%(prefix)sSRT tower full readout counter%(suffix)s'),
            kind = cms.untracked.string('TH2F'),
            otype = cms.untracked.string('Ecal3P'),
            btype = cms.untracked.string('SuperCrystal'),
            description = cms.untracked.string('Tower occupancy with FR flags.')
        ),
        ZSFullReadout = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sSelectiveReadoutTask/%(prefix)sSRT ZS Flagged Fully Readout Number%(suffix)s'),
            kind = cms.untracked.string('TH1F'),
            otype = cms.untracked.string('Ecal3P'),
            xaxis = cms.untracked.PSet(
                high = cms.untracked.double(20.0),
                nbins = cms.untracked.int32(20),
                low = cms.untracked.double(0.0),
                title = cms.untracked.string('number of towers')
            ),
            btype = cms.untracked.string('User'),
            description = cms.untracked.string('Number of ZS flagged but fully read out towers.')
        ),
        ZSFullReadoutMap = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sSelectiveReadoutTask/Counters/%(prefix)sSRT ZS flagged full readout counter%(suffix)s'),
            kind = cms.untracked.string('TH2F'),
            otype = cms.untracked.string('Ecal3P'),
            btype = cms.untracked.string('SuperCrystal'),
            description = cms.untracked.string('Number of ZS flagged but fully read out towers.')
        ),
        FRDroppedMap = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sSelectiveReadoutTask/Counters/%(prefix)sSRT FR flagged dropped counter%(suffix)s'),
            kind = cms.untracked.string('TH2F'),
            otype = cms.untracked.string('Ecal3P'),
            btype = cms.untracked.string('SuperCrystal'),
            description = cms.untracked.string('Number of FR flagged but dropped towers.')
        ),
        LowIntOutput = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sSelectiveReadoutTask/%(prefix)sSRT low interest ZS filter output%(suffix)s'),
            kind = cms.untracked.string('TH1F'),
            otype = cms.untracked.string('Ecal3P'),
            xaxis = cms.untracked.PSet(
                high = cms.untracked.double(60.0),
                nbins = cms.untracked.int32(100),
                low = cms.untracked.double(-60.0),
                title = cms.untracked.string('ADC counts*4')
            ),
            btype = cms.untracked.string('User'),
            description = cms.untracked.string('Output of the ZS filter for low interest towers.')
        ),
        LowIntPayload = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sSelectiveReadoutTask/%(prefix)sSRT low interest payload%(suffix)s'),
            kind = cms.untracked.string('TH1F'),
            otype = cms.untracked.string('Ecal3P'),
            xaxis = cms.untracked.PSet(
                high = cms.untracked.double(3.0),
                nbins = cms.untracked.int32(100),
                low = cms.untracked.double(0.0),
                title = cms.untracked.string('event size (kB)')
            ),
            btype = cms.untracked.string('User'),
            description = cms.untracked.string('Total data size from all low interest towers.')
        ),
        RUForcedMap = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sSelectiveReadoutTask/Counters/%(prefix)sSRT RU with forced SR counter%(suffix)s'),
            kind = cms.untracked.string('TH2F'),
            otype = cms.untracked.string('Ecal3P'),
            btype = cms.untracked.string('SuperCrystal'),
            description = cms.untracked.string('Tower occupancy of FORCED flag.')
        ),
        DCCSize = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sSelectiveReadoutTask/%(prefix)sSRT event size vs DCC'),
            kind = cms.untracked.string('TH2F'),
            yaxis = cms.untracked.PSet(
                edges = cms.untracked.vdouble(dccSizeBinEdges),
                title = cms.untracked.string('event size (kB)')
            ),
            otype = cms.untracked.string('Ecal2P'),
            btype = cms.untracked.string('DCC'),
            description = cms.untracked.string('Distribution of the per-DCC data size.')
        ),
        DCCSizeProf = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sSelectiveReadoutTask/%(prefix)sSRT DCC event size'),
            kind = cms.untracked.string('TProfile'),
            yaxis = cms.untracked.PSet(
                title = cms.untracked.string('event size (kB)')
            ),
            otype = cms.untracked.string('Ecal2P'),
            btype = cms.untracked.string('DCC'),
            description = cms.untracked.string('Mean and spread of the per-DCC data size.')
        ),
        ZSMap = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sSelectiveReadoutTask/Counters/%(prefix)sSRT tower ZS1+ZS2 counter%(suffix)s'),
            kind = cms.untracked.string('TH2F'),
            otype = cms.untracked.string('Ecal3P'),
            btype = cms.untracked.string('SuperCrystal'),
            description = cms.untracked.string('Tower occupancy of ZS1 and ZS2 flags.')
        ),
        HighIntPayload = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sSelectiveReadoutTask/%(prefix)sSRT high interest payload%(suffix)s'),
            kind = cms.untracked.string('TH1F'),
            otype = cms.untracked.string('Ecal3P'),
            xaxis = cms.untracked.PSet(
                high = cms.untracked.double(3.0),
                nbins = cms.untracked.int32(100),
                low = cms.untracked.double(0.0),
                title = cms.untracked.string('event size (kB)')
            ),
            btype = cms.untracked.string('User'),
            description = cms.untracked.string('Total data size from all high interest towers.')
        ),
        FlagCounterMap = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sSelectiveReadoutTask/Counters/%(prefix)sSRT tower flag counter%(suffix)s'),
            kind = cms.untracked.string('TH2F'),
            otype = cms.untracked.string('Ecal3P'),
            btype = cms.untracked.string('SuperCrystal'),
            description = cms.untracked.string('Tower occupancy of any SR flag.')
        ),
        FRDropped = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sSelectiveReadoutTask/%(prefix)sSRT FR Flagged Dropped Readout Number%(suffix)s'),
            kind = cms.untracked.string('TH1F'),
            otype = cms.untracked.string('Ecal3P'),
            xaxis = cms.untracked.PSet(
                high = cms.untracked.double(20.0),
                nbins = cms.untracked.int32(20),
                low = cms.untracked.double(0.0),
                title = cms.untracked.string('number of towers')
            ),
            btype = cms.untracked.string('User'),
            description = cms.untracked.string('Number of FR flagged but dropped towers.')
        ),
        EventSize = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sSelectiveReadoutTask/%(prefix)sSRT event size%(suffix)s'),
            kind = cms.untracked.string('TH1F'),
            otype = cms.untracked.string('Ecal3P'),
            xaxis = cms.untracked.PSet(
                high = cms.untracked.double(3.0),
                nbins = cms.untracked.int32(100),
                low = cms.untracked.double(0.0),
                title = cms.untracked.string('event size (kB)')
            ),
            btype = cms.untracked.string('User'),
            description = cms.untracked.string('Distribution of per-DCC data size.')
        ),
        FullReadout = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sSelectiveReadoutTask/%(prefix)sSRT full readout SR Flags Number%(suffix)s'),
            kind = cms.untracked.string('TH1F'),
            otype = cms.untracked.string('Ecal3P'),
            xaxis = cms.untracked.PSet(
                high = cms.untracked.double(200.0),
                nbins = cms.untracked.int32(100),
                low = cms.untracked.double(0.0),
                title = cms.untracked.string('number of towers')
            ),
            btype = cms.untracked.string('User'),
            description = cms.untracked.string('Number of FR flags per event.')
        ),
        TowerSize = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sSelectiveReadoutTask/%(prefix)sSRT tower event size%(suffix)s'),
            kind = cms.untracked.string('TProfile2D'),
            zaxis = cms.untracked.PSet(
                title = cms.untracked.string('size (bytes)')
            ),
            otype = cms.untracked.string('Ecal3P'),
            btype = cms.untracked.string('SuperCrystal'),
            description = cms.untracked.string('2D distribution of the mean data size from each readout unit.')
        )
    )
)
