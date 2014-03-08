import FWCore.ParameterSet.Config as cms

from DQM.EcalCommon.CommonParams_cfi import *

from DQM.EcalBarrelMonitorTasks.PedestalTask_cfi import ecalPedestalTask

minChannelEntries = 3
expectedMean = 200.
toleranceMean = 25.
toleranceRMS = [1., 1.2, 2.] # [G1, G6, G12]
expectedPNMean = 750.
tolerancePNMean = 100.
tolerancePNRMS = [20., 20.] # [G1, G16]

ecalPedestalClient = cms.untracked.PSet(
    params = cms.untracked.PSet(
        minChannelEntries = cms.untracked.int32(minChannelEntries),
        expectedMean = cms.untracked.double(expectedMean),
        toleranceMean = cms.untracked.double(toleranceMean),
        toleranceRMS = cms.untracked.vdouble(toleranceRMS),
        expectedPNMean = cms.untracked.double(expectedPNMean),
        tolerancePNMean = cms.untracked.double(tolerancePNMean),
        tolerancePNRMS = cms.untracked.vdouble(tolerancePNRMS)
    ),
    sources = cms.untracked.PSet(
        Pedestal = ecalPedestalTask.MEs.Pedestal,
        PNPedestal = ecalPedestalTask.MEs.PNPedestal
    ),
    MEs = cms.untracked.PSet(
        RMS = cms.untracked.PSet(
            kind = cms.untracked.string('TH1F'),
            multi = cms.untracked.PSet(
                gain = ecaldqmMGPAGains
            ),
            otype = cms.untracked.string('SM'),
            xaxis = cms.untracked.PSet(
                high = cms.untracked.double(10.0),
                nbins = cms.untracked.int32(100),
                low = cms.untracked.double(0.0)
            ),
            btype = cms.untracked.string('User'),
            path = cms.untracked.string('%(subdet)s/%(prefix)sPedestalClient/%(prefix)sPT pedestal rms G%(gain)s %(sm)s'),
            description = cms.untracked.string('Distribution of the pedestal RMS for each crystal channel. Channels with entries less than ' + str(minChannelEntries) + ' are not considered.')
        ),
        PNRMS = cms.untracked.PSet(
            kind = cms.untracked.string('TH1F'),
            multi = cms.untracked.PSet(
                pngain = ecaldqmMGPAGainsPN
            ),
            otype = cms.untracked.string('SMMEM'),
            xaxis = cms.untracked.PSet(
                high = cms.untracked.double(50.0),
                nbins = cms.untracked.int32(100),
                low = cms.untracked.double(0.0)
            ),
            btype = cms.untracked.string('User'),
            path = cms.untracked.string('%(subdet)s/%(prefix)sPedestalClient/%(prefix)sPDT PNs pedestal rms %(sm)s G%(pngain)s'),
            description = cms.untracked.string('Distribution of the pedestal RMS for each PN channel. Channels with entries less than ' + str(minChannelEntries) + ' are not considered.')
        ),
        PNQualitySummary = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sSummaryClient/%(prefix)sPT PN pedestal quality G%(pngain)s summary'),
            otype = cms.untracked.string('MEM2P'),
            multi = cms.untracked.PSet(
                pngain = ecaldqmMGPAGainsPN
            ),
            kind = cms.untracked.string('TH2F'),
            btype = cms.untracked.string('Crystal'),
            description = cms.untracked.string('Summary of the pedestal quality for PN diodes. A channel is red if the pedestal mean is off from ' + str(expectedPNMean) + ' by ' + str(tolerancePNMean) + ' or if the pedestal RMS is greater than threshold. RMS thresholds are ' + ('%.1f, %.1f' % tuple(tolerancePNRMS)) + ' for gains 1 and 16 respectively. Channels with entries less than ' + str(minChannelEntries) + ' are not considered.')
        ),
        QualitySummary = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sSummaryClient/%(prefix)sPT pedestal quality G%(gain)s summary%(suffix)s'),
            otype = cms.untracked.string('Ecal3P'),
            multi = cms.untracked.PSet(
                gain = ecaldqmMGPAGains
            ),
            kind = cms.untracked.string('TH2F'),
            btype = cms.untracked.string('SuperCrystal'),
            description = cms.untracked.string('Summary of the pedestal quality for crystals. A channel is red if the pedestal mean is off from ' + str(expectedMean) + ' by ' + str(toleranceMean) + ' or if the pedestal RMS is greater than threshold. RMS thresholds are ' + ('%.1f, %.1f, %.1f' % tuple(toleranceRMS)) + ' for gains 1, 6, and 12 respectively. Channels with entries less than ' + str(minChannelEntries) + ' are not considered.')
        ),
        Quality = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sPedestalClient/%(prefix)sPT pedestal quality G%(gain)s %(sm)s'),
            otype = cms.untracked.string('SM'),
            multi = cms.untracked.PSet(
                gain = ecaldqmMGPAGains
            ),
            kind = cms.untracked.string('TH2F'),
            btype = cms.untracked.string('Crystal'),
            description = cms.untracked.string('Summary of the pedestal quality for crystals. A channel is red if the pedestal mean is off from ' + str(expectedMean) + ' by ' + str(toleranceMean) + ' or if the pedestal RMS is greater than threshold. RMS thresholds are ' + ('%.1f, %.1f, %.1f' % tuple(toleranceRMS)) + ' for gains 1, 6, and 12 respectively. Channels with entries less than ' + str(minChannelEntries) + ' are not considered.')
        ),
        Mean = cms.untracked.PSet(
            kind = cms.untracked.string('TH1F'),
            multi = cms.untracked.PSet(
                gain = ecaldqmMGPAGains
            ),
            otype = cms.untracked.string('SM'),
            xaxis = cms.untracked.PSet(
                high = cms.untracked.double(230.0),
                nbins = cms.untracked.int32(120),
                low = cms.untracked.double(170.0)
            ),
            btype = cms.untracked.string('User'),
            path = cms.untracked.string('%(subdet)s/%(prefix)sPedestalClient/%(prefix)sPT pedestal mean G%(gain)s %(sm)s'),
            description = cms.untracked.string('Distribution of pedestal mean in each channel. Channels with entries less than ' + str(minChannelEntries) + ' are not considered.')
        )
    )
)
