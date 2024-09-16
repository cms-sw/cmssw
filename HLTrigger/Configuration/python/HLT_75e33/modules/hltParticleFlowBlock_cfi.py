import FWCore.ParameterSet.Config as cms

hltParticleFlowBlock = cms.EDProducer("PFBlockProducer",
    debug = cms.untracked.bool(False),
    elementImporters = cms.VPSet(
        cms.PSet(
            importerName = cms.string('SuperClusterImporter'),
            maximumHoverE = cms.double(0.5),
            minPTforBypass = cms.double(100.0),
            minSuperClusterPt = cms.double(10.0),
            source_eb = cms.InputTag("hltParticleFlowSuperClusterECAL","particleFlowSuperClusterECALBarrel"),
            source_ee = cms.InputTag("hltParticleFlowSuperClusterECAL","particleFlowSuperClusterECALEndcapWithPreshower"),
            hbheRecHitsTag = cms.InputTag("hltHbhereco"),
            maxSeverityHB = cms.int32(9),
            maxSeverityHE = cms.int32(9),
            usePFThresholdsFromDB = cms.bool(True),
            superClustersArePF = cms.bool(True)
        ),
        cms.PSet(
            DPtOverPtCuts_byTrackAlgo = cms.vdouble(
                10.0, 10.0, 10.0, 10.0, 10.0,
                5.0
            ),
            NHitCuts_byTrackAlgo = cms.vuint32(
                3, 3, 3, 3, 3,
                3
            ),
            cleanBadConvertedBrems = cms.bool(True),
            importerName = cms.string('GeneralTracksImporter'),
            maxDPtOPt = cms.double(1.0),
            muonMaxDPtOPt = cms.double(1),
            muonSrc = cms.InputTag("hltPhase2L3Muons"),
            source = cms.InputTag("hltPfTrack"),
            trackQuality = cms.string('highPurity'),
            useIterativeTracking = cms.bool(True),
            vetoEndcap = cms.bool(True),
            vetoMode = cms.uint32(2),
            vetoSrc = cms.InputTag("hltPfTICL")
        ),
        cms.PSet(
            BCtoPFCMap = cms.InputTag("hltParticleFlowSuperClusterECAL","PFClusterAssociationEBEE"),
            importerName = cms.string('ECALClusterImporter'),
            source = cms.InputTag("hltParticleFlowClusterECAL")
        ),
        cms.PSet(
            importerName = cms.string('GenericClusterImporter'),
            source = cms.InputTag("hltParticleFlowClusterHCAL")
        ),
        cms.PSet(
            importerName = cms.string('GenericClusterImporter'),
            source = cms.InputTag("hltParticleFlowBadHcalPseudoCluster")
        ),
        cms.PSet(
            importerName = cms.string('GenericClusterImporter'),
            source = cms.InputTag("hltParticleFlowClusterHO")
        ),
        cms.PSet(
            importerName = cms.string('GenericClusterImporter'),
            source = cms.InputTag("hltParticleFlowClusterHF")
        )
    ),
    linkDefinitions = cms.VPSet(
        cms.PSet(
            linkType = cms.string('TRACK:ECAL'),
            linkerName = cms.string('TrackAndECALLinker'),
            useKDTree = cms.bool(True)
        ),
        cms.PSet(
            linkType = cms.string('TRACK:HCAL'),
            linkerName = cms.string('TrackAndHCALLinker'),
            nMaxHcalLinksPerTrack = cms.int32(1),
            trajectoryLayerEntrance = cms.string('HCALEntrance'),
            trajectoryLayerExit = cms.string('HCALExit'),
            useKDTree = cms.bool(True)
        ),
        cms.PSet(
            linkType = cms.string('TRACK:HO'),
            linkerName = cms.string('TrackAndHOLinker'),
            useKDTree = cms.bool(False)
        ),
        cms.PSet(
            linkType = cms.string('ECAL:HCAL'),
            linkerName = cms.string('ECALAndHCALLinker'),
            minAbsEtaEcal = cms.double(2.5),
            useKDTree = cms.bool(False)
        ),
        cms.PSet(
            linkType = cms.string('HCAL:HO'),
            linkerName = cms.string('HCALAndHOLinker'),
            useKDTree = cms.bool(False)
        ),
        cms.PSet(
            linkType = cms.string('HFEM:HFHAD'),
            linkerName = cms.string('HFEMAndHFHADLinker'),
            useKDTree = cms.bool(False)
        ),
        cms.PSet(
            linkType = cms.string('TRACK:TRACK'),
            linkerName = cms.string('TrackAndTrackLinker'),
            useKDTree = cms.bool(False)
        ),
        cms.PSet(
            linkType = cms.string('ECAL:ECAL'),
            linkerName = cms.string('ECALAndECALLinker'),
            useKDTree = cms.bool(False)
        ),
        cms.PSet(
            linkType = cms.string('ECAL:BREM'),
            linkerName = cms.string('ECALAndBREMLinker'),
            useKDTree = cms.bool(False)
        ),
        cms.PSet(
            linkType = cms.string('HCAL:BREM'),
            linkerName = cms.string('HCALAndBREMLinker'),
            useKDTree = cms.bool(False)
        ),
        cms.PSet(
            SuperClusterMatchByRef = cms.bool(True),
            linkType = cms.string('SC:ECAL'),
            linkerName = cms.string('SCAndECALLinker'),
            useKDTree = cms.bool(False)
        ),
        cms.PSet(
            linkType = cms.string('TRACK:HFEM'),
            linkerName = cms.string('TrackAndHCALLinker'),
            nMaxHcalLinksPerTrack = cms.int32(-1),
            trajectoryLayerEntrance = cms.string('VFcalEntrance'),
            trajectoryLayerExit = cms.string(''),
            useKDTree = cms.bool(True)
        ),
        cms.PSet(
            linkType = cms.string('TRACK:HFHAD'),
            linkerName = cms.string('TrackAndHCALLinker'),
            nMaxHcalLinksPerTrack = cms.int32(-1),
            trajectoryLayerEntrance = cms.string('VFcalEntrance'),
            trajectoryLayerExit = cms.string(''),
            useKDTree = cms.bool(True)
        )
    ),
    verbose = cms.untracked.bool(False)
)
