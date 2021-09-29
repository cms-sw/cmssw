import FWCore.ParameterSet.Config as cms

pfClustersFromHGC3DClusters = cms.EDProducer("PFClusterProducerFromHGC3DClusters",
    corrector = cms.string('L1Trigger/Phase2L1ParticleFlow/data/hadcorr_HGCal3D_TC_110X.root'),
    correctorEmfMax = cms.double(1.125),
    emOnly = cms.bool(False),
    emVsPUID = cms.PSet(
        isPUFilter = cms.bool(True),
        method = cms.string('BDT'),
        preselection = cms.string(''),
        variables = cms.VPSet(
            cms.PSet(
                name = cms.string('fabs(eta)'),
                value = cms.string('abs(eta())')
            ),
            cms.PSet(
                name = cms.string('coreShowerLength'),
                value = cms.string('coreShowerLength()')
            ),
            cms.PSet(
                name = cms.string('maxLayer'),
                value = cms.string('maxLayer()')
            ),
            cms.PSet(
                name = cms.string('sigmaPhiPhiTot'),
                value = cms.string('sigmaPhiPhiTot()')
            )
        ),
        weightsFile = cms.string('L1Trigger/Phase2L1ParticleFlow/data/hgcal_egID/Photon_Pion_vs_Neutrino_BDTweights.xml.gz'),
        wp = cms.string('-0.02')
    ),
    emVsPionID = cms.PSet(
        isPUFilter = cms.bool(False),
        method = cms.string('BDT'),
        preselection = cms.string(''),
        variables = cms.VPSet(
            cms.PSet(
                name = cms.string('fabs(eta)'),
                value = cms.string('abs(eta())')
            ),
            cms.PSet(
                name = cms.string('coreShowerLength'),
                value = cms.string('coreShowerLength()')
            ),
            cms.PSet(
                name = cms.string('maxLayer'),
                value = cms.string('maxLayer()')
            ),
            cms.PSet(
                name = cms.string('hOverE'),
                value = cms.string('hOverE()')
            ),
            cms.PSet(
                name = cms.string('sigmaZZ'),
                value = cms.string('sigmaZZ()')
            )
        ),
        weightsFile = cms.string('L1Trigger/Phase2L1ParticleFlow/data/hgcal_egID/Photon_vs_Pion_BDTweights.xml.gz'),
        wp = cms.string('0.01')
    ),
    etMin = cms.double(1.0),
    preEmId = cms.string('hOverE < 0.3 && hOverE >= 0'),
    resol = cms.PSet(
        etaBins = cms.vdouble(
            1.7, 1.9, 2.2, 2.5, 2.8,
            2.9
        ),
        kind = cms.string('calo'),
        offset = cms.vdouble(
            1.793, 1.827, 2.363, 2.538, 2.812,
            2.642
        ),
        scale = cms.vdouble(
            0.138, 0.137, 0.124, 0.115, 0.106,
            0.121
        )
    ),
    src = cms.InputTag("hgcalBackEndLayer2Producer","HGCalBackendLayer2Processor3DClustering")
)
