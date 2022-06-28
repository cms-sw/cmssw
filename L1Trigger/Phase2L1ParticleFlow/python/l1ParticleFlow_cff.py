import FWCore.ParameterSet.Config as cms

from L1Trigger.Phase2L1ParticleFlow.pfTracksFromL1Tracks_cfi import pfTracksFromL1Tracks
from L1Trigger.Phase2L1ParticleFlow.pfClustersFromL1EGClusters_cfi import pfClustersFromL1EGClusters
from L1Trigger.Phase2L1ParticleFlow.pfClustersFromCombinedCalo_cff import pfClustersFromCombinedCaloHCal, pfClustersFromCombinedCaloHF
from L1Trigger.Phase2L1ParticleFlow.pfClustersFromHGC3DClusters_cfi import pfClustersFromHGC3DClusters
from L1Trigger.Phase2L1ParticleFlow.l1pfProducer_cfi import l1pfProducer
from L1Trigger.Phase2L1ParticleFlow.l1TkEgAlgo_cfi import tkEgConfig

pfTracksFromL1TracksBarrel = pfTracksFromL1Tracks.clone(
    resolCalo = pfClustersFromCombinedCaloHCal.resol.clone(),
)


l1ParticleFlow_calo_Task = cms.Task(
    pfClustersFromL1EGClusters ,
    pfClustersFromCombinedCaloHCal ,
    pfClustersFromCombinedCaloHF ,
    pfClustersFromHGC3DClusters
)
l1ParticleFlow_calo = cms.Sequence(l1ParticleFlow_calo_Task)

l1TkEgConfigBarrel = tkEgConfig.clone(
    doBremRecovery=False,
    writeEgSta=False
)

# PF in the barrel
l1pfProducerBarrel = l1pfProducer.clone(
    # inputs
    tracks = cms.InputTag('pfTracksFromL1TracksBarrel'),
    emClusters = [ cms.InputTag('pfClustersFromL1EGClusters') ],
    hadClusters = [ cms.InputTag('pfClustersFromCombinedCaloHCal:calibrated') ],
    # track-based PUPPI
    puppiUsingBareTracks = True,
    puppiDrMin = 0.07,
    puppiPtMax = 50.,
    vtxAlgo = "external",
    vtxFormat = cms.string("TkPrimaryVertex"),
    vtxCollection = cms.InputTag("L1TkPrimaryVertex",""),
    # eg algo configuration
    tkEgAlgoConfig = l1TkEgConfigBarrel,
    # puppi tuning
    puAlgo = "LinearizedPuppi",
    puppiEtaCuts            = cms.vdouble( 1.6 ), # just one bin
    puppiPtCuts             = cms.vdouble( 1.0 ),
    puppiPtCutsPhotons      = cms.vdouble( 1.0 ),
    puppiPtSlopes           = cms.vdouble( 0.3 ), # coefficient for pT
    puppiPtSlopesPhotons    = cms.vdouble( 0.3 ),
    puppiPtZeros            = cms.vdouble( 4.0 ), # ballpark pT from PU
    puppiPtZerosPhotons     = cms.vdouble( 2.5 ),
    puppiAlphaSlopes        = cms.vdouble( 0.7 ), # coefficient for alpha
    puppiAlphaSlopesPhotons = cms.vdouble( 0.7 ),
    puppiAlphaZeros         = cms.vdouble( 6.0 ), # ballpark alpha from PU
    puppiAlphaZerosPhotons  = cms.vdouble( 6.0 ),
    puppiAlphaCrops         = cms.vdouble(  4  ), # max. absolute value for alpha term
    puppiAlphaCropsPhotons  = cms.vdouble(  4  ),
    puppiPriors             = cms.vdouble( 5.0 ),
    puppiPriorsPhotons      = cms.vdouble( 1.0 ),
    # regionalize
    useRelativeRegionalCoordinates = cms.bool(False),
    trackRegionMode = cms.string("atCalo"),
    regions = cms.VPSet(
        cms.PSet(
            etaBoundaries = cms.vdouble(-1.5,1.5),
            phiSlices = cms.uint32(1),
            etaExtra = cms.double(0.3),
            phiExtra = cms.double(0.0)
        ),
    ),
)
l1ParticleFlow_pf_barrel_Task = cms.Task(
    pfTracksFromL1TracksBarrel ,
    l1pfProducerBarrel
)
l1ParticleFlow_pf_barrel = cms.Sequence(l1ParticleFlow_pf_barrel_Task)


l1TkEgConfigHGCal = tkEgConfig.clone(
    debug=0
)

# PF in HGCal
pfTracksFromL1TracksHGCal = pfTracksFromL1Tracks.clone(
    resolCalo = pfClustersFromHGC3DClusters.resol.clone(),
)
l1pfProducerHGCal = l1pfProducer.clone(
    # algo
    pfAlgo = "PFAlgo2HGC",
    # inputs
    tracks = cms.InputTag('pfTracksFromL1TracksHGCal'),
    emClusters  = [ cms.InputTag("pfClustersFromHGC3DClusters:egamma")],  # used only for E/gamma
    hadClusters = [ cms.InputTag("pfClustersFromHGC3DClusters") ],
    # track-based PUPPI
    puppiDrMin = 0.04,
    puppiPtMax = 50.,
    puppiUsingBareTracks = True,
    vtxAlgo = "external",
    vtxFormat = cms.string("TkPrimaryVertex"),
    vtxCollection = cms.InputTag("L1TkPrimaryVertex",""),
    # eg algo configuration
    tkEgAlgoConfig = l1TkEgConfigHGCal,
    # puppi tuning
    puAlgo = "LinearizedPuppi",
    puppiEtaCuts            = cms.vdouble( 2.0, 2.4, 3.1 ), # two bins in the tracker (different pT), one outside
    puppiPtCuts             = cms.vdouble( 1.0, 2.0, 4.0 ),
    puppiPtCutsPhotons      = cms.vdouble( 1.0, 2.0, 4.0 ),
    puppiPtSlopes           = cms.vdouble( 0.3, 0.3, 0.3 ), # coefficient for pT
    puppiPtSlopesPhotons    = cms.vdouble( 0.4, 0.4, 0.4 ), #When e/g ID not applied, use: cms.vdouble( 0.3, 0.3, 0.3 ),
    puppiPtZeros            = cms.vdouble( 5.0, 7.0, 9.0 ), # ballpark pT from PU
    puppiPtZerosPhotons     = cms.vdouble( 3.0, 4.0, 5.0 ),
    puppiAlphaSlopes        = cms.vdouble( 1.5, 1.5, 2.2 ),
    puppiAlphaSlopesPhotons = cms.vdouble( 1.5, 1.5, 2.2 ),
    puppiAlphaZeros         = cms.vdouble( 6.0, 6.0, 9.0 ),
    puppiAlphaZerosPhotons  = cms.vdouble( 6.0, 6.0, 9.0 ),
    puppiAlphaCrops         = cms.vdouble(  3 ,  3 ,  4  ), # max. absolute value for alpha term
    puppiAlphaCropsPhotons  = cms.vdouble(  3 ,  3 ,  4  ),
    puppiPriors             = cms.vdouble( 5.0, 5.0, 7.0 ),
    puppiPriorsPhotons      = cms.vdouble( 1.5, 1.5, 5.0 ), #When e/g ID not applied, use: cms.vdouble( 3.5, 3.5, 7.0 ),
    # regionalize
    useRelativeRegionalCoordinates = cms.bool(False),
    trackRegionMode = cms.string("atCalo"),
    regions = cms.VPSet(
        cms.PSet(
            etaBoundaries = cms.vdouble(-2.5,-1.5),
            phiSlices = cms.uint32(1),
            etaExtra = cms.double(0.3),
            phiExtra = cms.double(0.0)
        ),
        cms.PSet(
            etaBoundaries = cms.vdouble(1.5,2.5),
            phiSlices = cms.uint32(1),
            etaExtra = cms.double(0.3),
            phiExtra = cms.double(0.0)
        ),
    ),
)
l1pfProducerHGCal.linking.trackCaloDR = 0.1 # more precise cluster positions
l1pfProducerHGCal.linking.ecalPriority = False
l1pfProducerHGCalNoTK = l1pfProducerHGCal.clone(regions = cms.VPSet(
    cms.PSet(
        etaBoundaries = cms.vdouble(-3,-2.5),
        phiSlices = cms.uint32(1),
        etaExtra = cms.double(0.3),
        phiExtra = cms.double(0.0)
    ),
    cms.PSet(
        etaBoundaries = cms.vdouble(2.5,3),
        phiSlices = cms.uint32(1),
        etaExtra = cms.double(0.3),
        phiExtra = cms.double(0.0)
    ),
))

l1ParticleFlow_pf_hgcal_Task = cms.Task(
    pfTracksFromL1TracksHGCal ,
    l1pfProducerHGCal ,
    l1pfProducerHGCalNoTK
)
l1ParticleFlow_pf_hgcal = cms.Sequence(l1ParticleFlow_pf_hgcal_Task)

l1TkEgConfigHF = tkEgConfig.clone(
    debug=0
)
# PF in HF
l1pfProducerHF = l1pfProducer.clone(
    # inputs
    tracks = cms.InputTag(''), # no tracks
    emClusters = [ ],
    hadClusters = [ cms.InputTag('pfClustersFromCombinedCaloHF:calibrated') ],
    hadPtCut = 15,
    # not really useful, but for consistency
    puppiDrMin = 0.1,
    puppiPtMax = 100.,
    vtxAlgo = "external",
    vtxFormat = cms.string("TkPrimaryVertex"),
    vtxCollection = cms.InputTag("L1TkPrimaryVertex",""),
    # eg algo configuration
    tkEgAlgoConfig = l1TkEgConfigHF,
    # puppi tuning
    puAlgo = "LinearizedPuppi",
    puppiEtaCuts            = cms.vdouble( 5.5 ), # one bin
    puppiPtCuts             = cms.vdouble( 10. ),
    puppiPtCutsPhotons      = cms.vdouble( 10. ), # not used (no photons in HF)
    puppiPtSlopes           = cms.vdouble( 0.25),
    puppiPtSlopesPhotons    = cms.vdouble( 0.25), # not used (no photons in HF)
    puppiPtZeros            = cms.vdouble( 14. ), # ballpark pT from PU
    puppiPtZerosPhotons     = cms.vdouble( 14. ), # not used (no photons in HF)
    puppiAlphaSlopes        = cms.vdouble( 0.6 ),
    puppiAlphaSlopesPhotons = cms.vdouble( 0.6 ), # not used (no photons in HF)
    puppiAlphaZeros         = cms.vdouble( 9.0 ),
    puppiAlphaZerosPhotons  = cms.vdouble( 9.0 ), # not used (no photons in HF)
    puppiAlphaCrops         = cms.vdouble(  4  ),
    puppiAlphaCropsPhotons  = cms.vdouble(  4  ), # not used (no photons in HF)
    puppiPriors             = cms.vdouble( 6.0 ),
    puppiPriorsPhotons      = cms.vdouble( 6.0 ), # not used (no photons in HF)
    # regionalize
    useRelativeRegionalCoordinates = cms.bool(False),
    trackRegionMode = cms.string("atCalo"),
    regions = cms.VPSet(
        cms.PSet(
            etaBoundaries = cms.vdouble(-5.5,-3),
            phiSlices = cms.uint32(1),
            etaExtra = cms.double(0.0),
            phiExtra = cms.double(0.0)
        ),
        cms.PSet(
            etaBoundaries = cms.vdouble(3,5.5),
            phiSlices = cms.uint32(1),
            etaExtra = cms.double(0.0),
            phiExtra = cms.double(0.0)
        ),
    )
)
l1ParticleFlow_pf_hf_Task = cms.Task(
    l1pfProducerHF
)
l1ParticleFlow_pf_hf = cms.Sequence(l1ParticleFlow_pf_hf_Task)


# PF in the TSA Region
l1pfProducerTSA = l1pfProducerBarrel.clone(
    trackRegionMode = cms.string("atVertex"),
    regions = cms.VPSet(
        cms.PSet(
            etaBoundaries = cms.vdouble(-3,3),
            phiSlices = cms.uint32(18),
            etaExtra = cms.double(0.0),
            phiExtra = cms.double(0.0)
        ),
    ),
)
l1ParticleFlow_pf_tsa = cms.Sequence(
    pfTracksFromL1TracksBarrel +
    l1pfProducerTSA
)

# Merging all outputs
l1pfCandidates = cms.EDProducer("L1TPFCandMultiMerger",
    pfProducers = cms.VInputTag(
        cms.InputTag("l1pfProducerBarrel"),
        cms.InputTag("l1pfProducerHGCal"),
        cms.InputTag("l1pfProducerHGCalNoTK"),
        cms.InputTag("l1pfProducerHF")
    ),
    labelsToMerge = cms.vstring("Calo", "TK", "TKVtx", "PF", "Puppi"),
    regionalLabelsToMerge = cms.vstring(),
)

l1tCorrelatorEG = cms.EDProducer(
    "L1TEGMultiMerger",
    tkElectrons=cms.VPSet(
        cms.PSet(
            instance=cms.string("L1TkEleEE"),
            pfProducers=cms.VInputTag(
                cms.InputTag("l1pfProducerHGCal", 'L1TkEle')
            )
        ),
        cms.PSet(
            instance=cms.string("L1TkEleEB"),
            pfProducers=cms.VInputTag(
                cms.InputTag("l1pfProducerBarrel", 'L1TkEle')
            )
        )
    ),
    tkEms=cms.VPSet(
        cms.PSet(
            instance=cms.string("L1TkEmEE"),
            pfProducers=cms.VInputTag(
                cms.InputTag("l1pfProducerHGCal", 'L1TkEm'),
                cms.InputTag("l1pfProducerHGCalNoTK", 'L1TkEm')
            )
        ),
        cms.PSet(
            instance=cms.string("L1TkEmEB"),
            pfProducers=cms.VInputTag(
                cms.InputTag("l1pfProducerBarrel", 'L1TkEm')
            )
        )
    ),
    tkEgs=cms.VPSet(
        cms.PSet(
            instance=cms.string("L1EgEE"),
            pfProducers=cms.VInputTag(
                cms.InputTag("l1pfProducerHGCal", 'L1Eg'),
                cms.InputTag("l1pfProducerHGCalNoTK", 'L1Eg')
            )
        )    
    )
)

l1ParticleFlow_proper = cms.Sequence(
    l1ParticleFlow_calo +
    l1ParticleFlow_pf_barrel +
    l1ParticleFlow_pf_hgcal +
    l1ParticleFlow_pf_hf +
    l1pfCandidates +
    l1tCorrelatorEG
)

l1ParticleFlow = cms.Sequence(l1ParticleFlow_proper)

l1ParticleFlowTask = cms.Task(
    l1ParticleFlow_calo_Task,
    l1ParticleFlow_pf_barrel_Task,
    l1ParticleFlow_pf_hgcal_Task,
    l1ParticleFlow_pf_hf_Task,
    cms.Task(l1pfCandidates),
    cms.Task(l1tCorrelatorEG),
)
