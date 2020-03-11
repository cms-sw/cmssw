import FWCore.ParameterSet.Config as cms

ecalRecoSummaryTask = cms.untracked.PSet(
    params = cms.untracked.PSet(
        rechitThresholdEB = cms.untracked.double(0.8),
        rechitThresholdEE = cms.untracked.double(1.2),
        fillRecoFlagReduced = cms.untracked.bool(True)
    ),
    MEs = cms.untracked.PSet(
        EnergyMax = cms.untracked.PSet(
            path = cms.untracked.string('%(subdet)s/%(prefix)sRecoSummary/recHits_%(subdetshortsig)s_energyMax'),
            kind = cms.untracked.string('TH1F'),
            otype = cms.untracked.string('Ecal3P'),
            btype = cms.untracked.string('User'),
            xaxis = cms.untracked.PSet(
                high = cms.untracked.double(100.),
                low = cms.untracked.double(-10.),
                nbins = cms.untracked.int32(110)
            ),
            description = cms.untracked.string('Maximum energy of the rechit.')
        ),
        Chi2 = cms.untracked.PSet(
            path = cms.untracked.string("%(subdet)s/%(prefix)sRecoSummary/recHits_%(subdetshortsig)s_Chi2"),
            kind = cms.untracked.string('TH1F'),
            otype = cms.untracked.string('Ecal3P'),
            btype = cms.untracked.string('User'),
            xaxis = cms.untracked.PSet(
                high = cms.untracked.double(100.),
                low = cms.untracked.double(0.),
                nbins = cms.untracked.int32(100)
            ),
            description = cms.untracked.string('Chi2 of the pulse reconstruction.')
        ),
        Time = cms.untracked.PSet(
            path = cms.untracked.string("%(subdet)s/%(prefix)sRecoSummary/recHits_%(subdetshortsig)s_time"),
            kind = cms.untracked.string('TH1F'),
            otype = cms.untracked.string('Ecal3P'),
            btype = cms.untracked.string('User'),
            xaxis = cms.untracked.PSet(
                high = cms.untracked.double(50.),
                low = cms.untracked.double(-50.),
                nbins = cms.untracked.int32(100)
            ),
            description = cms.untracked.string('Rechit time.')
        ),
        SwissCross = cms.untracked.PSet(
            path = cms.untracked.string("%(subdet)s/%(prefix)sRecoSummary/recHits_%(subdetshort)s_E1oE4"),
            kind = cms.untracked.string('TH1F'),
            otype = cms.untracked.string('EB'),
            btype = cms.untracked.string('User'),
            xaxis = cms.untracked.PSet(
                high = cms.untracked.double(1.5),
                low = cms.untracked.double(0.),
                nbins = cms.untracked.int32(100)
            ),
            description = cms.untracked.string('Swiss cross.')
        ),
        RecoFlagAll = cms.untracked.PSet(
            path = cms.untracked.string("%(subdet)s/%(prefix)sRecoSummary/recHits_%(subdetshort)s_recoFlag"),
            kind = cms.untracked.string('TH1F'),
            otype = cms.untracked.string('Ecal2P'),
            btype = cms.untracked.string('User'),
            xaxis = cms.untracked.PSet(
                high = cms.untracked.double(15.5),
                low = cms.untracked.double(-0.5),
                nbins = cms.untracked.int32(16)
            ),
            description = cms.untracked.string('Reconstruction flags from all rechits.')
        ),
        RecoFlagReduced = cms.untracked.PSet(
            path = cms.untracked.string("%(subdet)s/%(prefix)sRecoSummary/redRecHits_%(subdetshort)s_recoFlag"),
            kind = cms.untracked.string('TH1F'),
            otype = cms.untracked.string('Ecal2P'),
            btype = cms.untracked.string('User'),
            xaxis = cms.untracked.PSet(
                high = cms.untracked.double(15.5),
                low = cms.untracked.double(-0.5),
                nbins = cms.untracked.int32(16)
            ),
            description = cms.untracked.string('Reconstruction flags from reduced rechits.')
        ),
        RecoFlagBasicCluster = cms.untracked.PSet(
            path = cms.untracked.string("%(subdet)s/%(prefix)sRecoSummary/basicClusters_recHits_%(subdetshort)s_recoFlag"),
            kind = cms.untracked.string('TH1F'),
            otype = cms.untracked.string('Ecal2P'),
            btype = cms.untracked.string('User'),
            xaxis = cms.untracked.PSet(
                high = cms.untracked.double(15.5),
                low = cms.untracked.double(-0.5),
                nbins = cms.untracked.int32(16)
            ),
            description = cms.untracked.string('Reconstruction flags from rechits in basic clusters.')
        )
    )
)
