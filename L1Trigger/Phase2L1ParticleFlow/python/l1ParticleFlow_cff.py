import FWCore.ParameterSet.Config as cms

from L1Trigger.Phase2L1ParticleFlow.pfTracksFromL1Tracks_cfi import pfTracksFromL1Tracks
from L1Trigger.Phase2L1ParticleFlow.pfClustersFromL1EGClusters_cfi import pfClustersFromL1EGClusters
from L1Trigger.Phase2L1ParticleFlow.pfClustersFromCombinedCalo_cfi import pfClustersFromCombinedCalo
from L1Trigger.Phase2L1ParticleFlow.l1pfProducer_cfi import l1pfProducer

# Using phase2_hgcalV10 to customize the config for all 106X samples, since there's no other modifier for it
from Configuration.Eras.Modifier_phase2_hgcalV10_cff import phase2_hgcalV10
from Configuration.Eras.Modifier_phase2_hgcalV11_cff import phase2_hgcalV11

# Calorimeter part: ecal + hcal + hf only
pfClustersFromCombinedCaloHCal = pfClustersFromCombinedCalo.clone(
    hcalHGCTowers = [], hcalDigis = [],
    hcalDigisBarrel = True, hcalDigisHF = False,
    hadCorrector = cms.string("L1Trigger/Phase2L1ParticleFlow/data/hadcorr_barrel.root"),
    resol = cms.PSet(
            etaBins = cms.vdouble( 0.700,  1.200,  1.600),
            offset  = cms.vdouble( 2.582,  2.191, -0.077),
            scale   = cms.vdouble( 0.122,  0.143,  0.465),
            kind    = cms.string('calo'),
    ))
phase2_hgcalV10.toModify(pfClustersFromCombinedCaloHCal,
    hadCorrector  = "L1Trigger/Phase2L1ParticleFlow/data/hadcorr_barrel_106X.root",
    resol = cms.PSet(
            etaBins = cms.vdouble( 0.700,  1.200,  1.600),
            offset  = cms.vdouble( 3.084,  2.715,  0.107),
            scale   = cms.vdouble( 0.118,  0.130,  0.442),
            kind    = cms.string('calo'),
    )
)
phase2_hgcalV11.toModify(pfClustersFromCombinedCaloHCal,
    hadCorrector  = "L1Trigger/Phase2L1ParticleFlow/data/hadcorr_barrel_110X.root",
    resol = cms.PSet(
            etaBins = cms.vdouble( 0.700,  1.200,  1.600),
            offset  = cms.vdouble( 2.909,  2.864,  0.294),
            scale   = cms.vdouble( 0.119,  0.127,  0.442),
            kind    = cms.string('calo'),
    )
)

pfTracksFromL1TracksBarrel = pfTracksFromL1Tracks.clone(
    resolCalo = pfClustersFromCombinedCaloHCal.resol.clone(),
)

pfClustersFromCombinedCaloHF = pfClustersFromCombinedCalo.clone(
    ecalCandidates = [], hcalHGCTowers = [],
    phase2barrelCaloTowers = [],
    hadCorrector = cms.string("L1Trigger/Phase2L1ParticleFlow/data/hfcorr.root"),
    resol = cms.PSet(
            etaBins = cms.vdouble( 3.500,  4.000,  4.500,  5.000),
            offset  = cms.vdouble( 1.099,  0.930,  1.009,  1.369),
            scale   = cms.vdouble( 0.152,  0.151,  0.144,  0.179),
            kind    = cms.string('calo'),
    ))
phase2_hgcalV10.toModify(pfClustersFromCombinedCaloHF,
    hcalCandidates = cms.VInputTag(cms.InputTag("hgcalBackEndLayer2Producer","HGCalBackendLayer2Processor3DClustering")),
    hadCorrector  = "L1Trigger/Phase2L1ParticleFlow/data/hfcorr_106X.root",
    resol = cms.PSet(
            etaBins = cms.vdouble( 3.500,  4.000,  4.500,  5.000),
            offset  = cms.vdouble(-0.846,  0.696,  1.313,  1.044),
            scale   = cms.vdouble( 0.815,  0.164,  0.146,  0.192),
            kind    = cms.string('calo'),
    )
)
phase2_hgcalV11.toModify(pfClustersFromCombinedCaloHF,
    hcalCandidates = cms.VInputTag(cms.InputTag("hgcalBackEndLayer2Producer","HGCalBackendLayer2Processor3DClustering")),
    hadCorrector  = "L1Trigger/Phase2L1ParticleFlow/data/hfcorr_110X.root",
    resol = cms.PSet(
            etaBins = cms.vdouble( 3.500,  4.000,  4.500,  5.000),
            offset  = cms.vdouble(-1.125,  1.220,  1.514,  1.414),
            scale   = cms.vdouble( 0.868,  0.159,  0.148,  0.194),
            kind    = cms.string('calo'),
    )
)

# Calorimeter part: hgcal
from L1Trigger.Phase2L1ParticleFlow.pfClustersFromHGC3DClusters_cfi import pfClustersFromHGC3DClusters

l1ParticleFlow_calo_Task = cms.Task(
    pfClustersFromL1EGClusters ,
    pfClustersFromCombinedCaloHCal ,
    pfClustersFromCombinedCaloHF ,
    pfClustersFromHGC3DClusters
)
l1ParticleFlow_calo = cms.Sequence(l1ParticleFlow_calo_Task)


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



# PF in HGCal
pfTracksFromL1TracksHGCal = pfTracksFromL1Tracks.clone(
    resolCalo = pfClustersFromHGC3DClusters.resol.clone(),
)
l1pfProducerHGCal = l1pfProducer.clone(
    # algo
    pfAlgo = "PFAlgo2HGC",
    # inputs
    tracks = cms.InputTag('pfTracksFromL1TracksHGCal'),
    emClusters  = [ ],  # EM clusters are not used (only added to NTuple for calibration/monitoring)
    hadClusters = [ cms.InputTag("pfClustersFromHGC3DClusters") ],
    # track-based PUPPI
    puppiDrMin = 0.04,
    puppiPtMax = 50.,
    puppiUsingBareTracks = True,
    vtxAlgo = "external",
    vtxFormat = cms.string("TkPrimaryVertex"),
    vtxCollection = cms.InputTag("L1TkPrimaryVertex",""),
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
)

l1ParticleFlow_proper = cms.Sequence(
    l1ParticleFlow_calo +
    l1ParticleFlow_pf_barrel +
    l1ParticleFlow_pf_hgcal +
    l1ParticleFlow_pf_hf +
    l1pfCandidates
)

l1ParticleFlow = cms.Sequence(l1ParticleFlow_proper)

l1ParticleFlowTask = cms.Task(
    l1ParticleFlow_calo_Task,
    l1ParticleFlow_pf_barrel_Task,
    l1ParticleFlow_pf_hgcal_Task,
    l1ParticleFlow_pf_hf_Task,
    cms.Task(l1pfCandidates)
)
