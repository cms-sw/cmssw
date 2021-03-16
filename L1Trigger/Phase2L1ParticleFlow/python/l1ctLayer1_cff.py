import FWCore.ParameterSet.Config as cms

from l1TkEgAlgoEmulator_cfi import tkEgAlgoParameters

l1ctLayer1Barrel = cms.EDProducer("L1TCorrelatorLayer1Producer",
    tracks = cms.InputTag('pfTracksFromL1Tracks'),
    muons = cms.InputTag('simGmtStage2Digis',),
    useStandaloneMuons = cms.bool(True),
    useTrackerMuons = cms.bool(False),
    emClusters = cms.VInputTag(cms.InputTag('pfClustersFromL1EGClusters')),
    hadClusters = cms.VInputTag(cms.InputTag('pfClustersFromCombinedCaloHCal:calibrated')),
    vtxCollection = cms.InputTag("L1TkPrimaryVertex",""),
    emPtCut  = cms.double(0.5),
    hadPtCut = cms.double(1.0),
    trkPtCut    = cms.double(2.0),
    regionizerAlgo = cms.string("Ideal"),
    pfAlgo = cms.string("PFAlgo3"),
    puAlgo = cms.string("LinearizedPuppi"),
    regionizerAlgoParameters = cms.PSet(
        useAlsoVtxCoords = cms.bool(True),
    ),
    pfAlgoParameters = cms.PSet(
        nTrack = cms.uint32(50), # very large numbers for first test
        nCalo = cms.uint32(50), # very large numbers for first test
        nMu = cms.uint32(5), # very large numbers for first test
        nSelCalo = cms.uint32(50), # very large numbers for first test
        nEmCalo = cms.uint32(50), # very large numbers for first test
        nPhoton = cms.uint32(50), # very large numbers for first test
        nAllNeutral = cms.uint32(50), # very large numbers for first test
        trackMuDR    = cms.double(0.2), # accounts for poor resolution of standalone, and missing propagations
        trackEmDR   = cms.double(0.04), # 1 Ecal crystal size is 0.02, and ~2 cm in HGCal is ~0.007
        emCaloDR    = cms.double(0.10),    # 1 Hcal tower size is ~0.09
        trackCaloDR = cms.double(0.15),
        maxInvisiblePt = cms.double(10.0), # max allowed pt of a track with no calo energy
        tightTrackMaxInvisiblePt = cms.double(20),
        caloResolution = cms.PSet(
            etaBins = cms.vdouble( 0.700,  1.200,  1.600),
            offset  = cms.vdouble( 2.909,  2.864,  0.294),
            scale   = cms.vdouble( 0.119,  0.127,  0.442),
        ),
        debug = cms.untracked.bool(False)
    ),
    puAlgoParameters = cms.PSet(
        nTrack = cms.uint32(50), # very large numbers for first test
        nIn = cms.uint32(50), # very large numbers for first test
        nOut = cms.uint32(50), # very large numbers for first test
        dZ     = cms.double(0.5),
        dr     = cms.double(0.3),
        drMin  = cms.double(0.07),
        ptMax  = cms.double(50.),
        absEtaCuts         = cms.vdouble( ), # just one bin, so no edge needd
        ptCut             = cms.vdouble( 1.0 ),
        ptSlopes           = cms.vdouble( 0.3 ), # coefficient for pT
        ptSlopesPhoton    = cms.vdouble( 0.3 ),
        ptZeros            = cms.vdouble( 4.0 ), # ballpark pT from PU
        ptZerosPhoton     = cms.vdouble( 2.5 ),
        alphaSlopes        = cms.vdouble( 0.7 ), # coefficient for alpha
        alphaZeros         = cms.vdouble( 6.0 ), # ballpark alpha from PU
        alphaCrop         = cms.vdouble(  4  ), # max. absolute value for alpha term
        priors             = cms.vdouble( 5.0 ),
        priorsPhoton      = cms.vdouble( 1.0 ),
        debug = cms.untracked.bool(False)
    ),
    tkEgAlgoParameters=tkEgAlgoParameters.clone(),
    caloSectors = cms.VPSet(
        cms.PSet( 
            etaBoundaries = cms.vdouble(-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5),
            phiSlices     = cms.uint32(6)
        )
    ),
    regions = cms.VPSet(
        cms.PSet( 
            etaBoundaries = cms.vdouble(-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5),
            phiSlices     = cms.uint32(9),
            etaExtra = cms.double(0.25),
            phiExtra = cms.double(0.25),
        )
    ),

)

_hgcalSectors = cms.VPSet(
    cms.PSet( 
        etaBoundaries = cms.vdouble(-3.0, -1.5),
        phiSlices     = cms.uint32(3)
    ),
    cms.PSet( 
        etaBoundaries = cms.vdouble(+1.5, +3.0),
        phiSlices     = cms.uint32(3)
    )

)

l1ctLayer1HGCal = cms.EDProducer("L1TCorrelatorLayer1Producer",
    tracks = cms.InputTag('pfTracksFromL1Tracks'),
    muons = cms.InputTag('simGmtStage2Digis',),
    useStandaloneMuons = cms.bool(True),
    useTrackerMuons = cms.bool(False),
    emClusters = cms.VInputTag(cms.InputTag('pfClustersFromHGC3DClusters:em')),
    hadClusters = cms.VInputTag(cms.InputTag('pfClustersFromHGC3DClusters')),
    vtxCollection = cms.InputTag("L1TkPrimaryVertex",""),
    emPtCut  = cms.double(0.5),
    hadPtCut = cms.double(1.0),
    trkPtCut    = cms.double(2.0),
    regionizerAlgo = cms.string("Ideal"),
    pfAlgo = cms.string("PFAlgo2HGC"),
    puAlgo = cms.string("LinearizedPuppi"),
    regionizerAlgoParameters = cms.PSet(
        useAlsoVtxCoords = cms.bool(True),
    ),
    pfAlgoParameters = cms.PSet(
        nTrack = cms.uint32(50), # very large numbers for first test
        nCalo = cms.uint32(50), # very large numbers for first test
        nMu = cms.uint32(5), # very large numbers for first test
        nSelCalo = cms.uint32(50), # very large numbers for first test
        trackMuDR    = cms.double(0.2), # accounts for poor resolution of standalone, and missing propagations
        trackCaloDR = cms.double(0.1),
        maxInvisiblePt = cms.double(10.0), # max allowed pt of a track with no calo energy
        tightTrackMaxInvisiblePt = cms.double(20),
        caloResolution = cms.PSet(
            etaBins = cms.vdouble( 1.700,  1.900,  2.200,  2.500,  2.800,  2.900),
            offset  = cms.vdouble( 1.793,  1.827,  2.363,  2.538,  2.812,  2.642),
            scale   = cms.vdouble( 0.138,  0.137,  0.124,  0.115,  0.106,  0.121),
        ),
        debug = cms.untracked.bool(False)
    ),
    puAlgoParameters = cms.PSet(
        nTrack = cms.uint32(50), # very large numbers for first test
        nIn = cms.uint32(50), # very large numbers for first test
        nOut = cms.uint32(50), # very large numbers for first test
        dZ     = cms.double(1.33),
        dr     = cms.double(0.3),
        drMin  = cms.double(0.04),
        ptMax  = cms.double(50.),
        absEtaCuts         = cms.vdouble( 2.0 ), # two bins in the tracker (different eta); give only the one boundary between them 
        ptCut             = cms.vdouble( 1.0, 2.0 ),
        ptSlopes           = cms.vdouble( 0.3, 0.3 ), # coefficient for pT
        ptSlopesPhoton    = cms.vdouble( 0.4, 0.4 ), #When e/g ID not applied, use: cms.vdouble( 0.3, 0.3, 0.3 ),
        ptZeros            = cms.vdouble( 5.0, 7.0 ), # ballpark pT from PU
        ptZerosPhoton     = cms.vdouble( 3.0, 4.0 ),
        alphaSlopes        = cms.vdouble( 1.5, 1.5 ),
        alphaZeros         = cms.vdouble( 6.0, 6.0 ),
        alphaCrop         = cms.vdouble(  3 ,  3  ), # max. absolute value for alpha term
        priors             = cms.vdouble( 5.0, 5.0 ),
        priorsPhoton      = cms.vdouble( 1.5, 1.5 ), #When e/g ID not applied, use: cms.vdouble( 3.5, 3.5, 7.0 ),
        debug = cms.untracked.bool(False)
    ),
    tkEgAlgoParameters=tkEgAlgoParameters.clone(
        doBremRecovery=True,
        filterHwQuality=True,
        writeEGSta=True),
    caloSectors = _hgcalSectors,
    regions = cms.VPSet(
        cms.PSet( 
            etaBoundaries = cms.vdouble(-2.5, -1.5),
            phiSlices     = cms.uint32(9),
            etaExtra = cms.double(0.25),
            phiExtra = cms.double(0.25),
        ),
        cms.PSet( 
            etaBoundaries = cms.vdouble(+1.5, +2.5),
            phiSlices     = cms.uint32(9),
            etaExtra = cms.double(0.25),
            phiExtra = cms.double(0.25),
        )

    ),
)


l1ctLayer1HGCalNoTK = cms.EDProducer("L1TCorrelatorLayer1Producer",
    tracks = cms.InputTag(''),
    muons = cms.InputTag('simGmtStage2Digis',),
    useStandaloneMuons = cms.bool(False),
    useTrackerMuons = cms.bool(False),
    emClusters = cms.VInputTag(cms.InputTag('pfClustersFromHGC3DClusters:em')),
    hadClusters = cms.VInputTag(cms.InputTag('pfClustersFromHGC3DClusters')),
    vtxCollection = cms.InputTag("L1TkPrimaryVertex",""),
    emPtCut  = cms.double(0.5),
    hadPtCut = cms.double(1.0),
    trkPtCut    = cms.double(2.0),
    regionizerAlgo = cms.string("Ideal"),
    pfAlgo = cms.string("PFAlgoDummy"),
    puAlgo = cms.string("LinearizedPuppi"),
    regionizerAlgoParameters = cms.PSet(
        useAlsoVtxCoords = cms.bool(True),
    ),
    pfAlgoParameters = cms.PSet(
        nCalo = cms.uint32(50), # very large numbers for first test
        nMu = cms.uint32(5), # very large numbers for first test
        debug = cms.untracked.bool(False)
    ),
    puAlgoParameters = cms.PSet(
        nTrack = cms.uint32(50), # very large numbers for first test
        nIn = cms.uint32(50), # very large numbers for first test
        nOut = cms.uint32(50), # very large numbers for first test
        dZ     = cms.double(1.33),
        dr     = cms.double(0.3),
        drMin  = cms.double(0.04),
        ptMax  = cms.double(50.),
        absEtaCuts         = cms.vdouble( ), # just one bin
        ptCut             = cms.vdouble( 4.0 ),
        ptSlopes           = cms.vdouble( 0.3 ), # coefficient for pT
        ptSlopesPhoton    = cms.vdouble( 0.4 ), #When e/g ID not applied, use: cms.vdouble( 0.3, 0.3, 0.3 ),
        ptZeros            = cms.vdouble( 9.0 ), # ballpark pT from PU
        ptZerosPhoton     = cms.vdouble( 5.0 ),
        alphaSlopes        = cms.vdouble( 2.2 ),
        alphaZeros         = cms.vdouble( 9.0 ),
        alphaCrop         = cms.vdouble(  4  ), # max. absolute value for alpha term
        priors             = cms.vdouble( 7.0 ),
        priorsPhoton      = cms.vdouble( 5.0 ), #When e/g ID not applied, use: cms.vdouble( 3.5, 3.5, 7.0 ),
        debug = cms.untracked.bool(False)
    ),
    tkEgAlgoParameters=tkEgAlgoParameters.clone(
        doBremRecovery=True,
        filterHwQuality=True,
        writeEGSta=True),
    caloSectors = _hgcalSectors,
    regions = cms.VPSet(
        cms.PSet( 
            etaBoundaries = cms.vdouble(-3.0, -2.5),
            phiSlices     = cms.uint32(9),
            etaExtra = cms.double(0.25),
            phiExtra = cms.double(0.25),
        ),
        cms.PSet( 
            etaBoundaries = cms.vdouble(+2.5, +3.0),
            phiSlices     = cms.uint32(9),
            etaExtra = cms.double(0.25),
            phiExtra = cms.double(0.25),
        )

    ),
)

l1ctLayer1HF = cms.EDProducer("L1TCorrelatorLayer1Producer",
    tracks = cms.InputTag(''),
    muons = cms.InputTag('simGmtStage2Digis',),
    useStandaloneMuons = cms.bool(False),
    useTrackerMuons = cms.bool(False),
    emClusters = cms.VInputTag(),
    hadClusters = cms.VInputTag(cms.InputTag('pfClustersFromCombinedCaloHF:calibrated')),
    vtxCollection = cms.InputTag("L1TkPrimaryVertex",""),
    emPtCut  = cms.double(0.5),
    hadPtCut = cms.double(15.0),
    trkPtCut    = cms.double(2.0),
    regionizerAlgo = cms.string("Ideal"),
    pfAlgo = cms.string("PFAlgoDummy"),
    puAlgo = cms.string("LinearizedPuppi"),
    regionizerAlgoParameters = cms.PSet(
        useAlsoVtxCoords = cms.bool(True),
    ),
    pfAlgoParameters = cms.PSet(
        nCalo = cms.uint32(50), # very large numbers for first test
        nMu = cms.uint32(5), # very large numbers for first test
        debug = cms.untracked.bool(False)
    ),
    puAlgoParameters = cms.PSet(
        nTrack = cms.uint32(0), # very large numbers for first test
        nIn = cms.uint32(50), # very large numbers for first test
        nOut = cms.uint32(50), # very large numbers for first test
        dZ     = cms.double(1.33),
        dr     = cms.double(0.3),
        drMin  = cms.double(0.1),
        ptMax  = cms.double(100.),
        absEtaCuts         = cms.vdouble(   ), # just one bin
        ptCut             = cms.vdouble( 10.0  ),
        ptSlopes           = cms.vdouble(  0.25 ),
        ptSlopesPhoton    = cms.vdouble(  0.25 ),
        ptZeros            = cms.vdouble( 14.0  ),
        ptZerosPhoton     = cms.vdouble( 14.0  ),
        alphaSlopes        = cms.vdouble(  0.6  ),
        alphaZeros         = cms.vdouble(  9.0  ),
        alphaCrop         = cms.vdouble(   4   ),
        priors             = cms.vdouble(  6.0  ),
        priorsPhoton      = cms.vdouble(  6.0  ),
        debug = cms.untracked.bool(False)
    ),
    tkEgAlgoParameters=tkEgAlgoParameters.clone(
        doBremRecovery=True,
        filterHwQuality=True,
        writeEGSta=True),
    caloSectors = cms.VPSet(
        cms.PSet( 
            etaBoundaries = cms.vdouble(-5.5, -3.0),
            phiSlices     = cms.uint32(9)
        ),
        cms.PSet( 
            etaBoundaries = cms.vdouble(+3.0, +5.5),
            phiSlices     = cms.uint32(9)
        )
    ),
    regions = cms.VPSet(
        cms.PSet( 
            etaBoundaries = cms.vdouble(-5.5, -3.0),
            phiSlices     = cms.uint32(9),
            etaExtra = cms.double(0.25),
            phiExtra = cms.double(0.25),
        ),
        cms.PSet( 
            etaBoundaries = cms.vdouble(+3.0, +5.5),
            phiSlices     = cms.uint32(9),
            etaExtra = cms.double(0.25),
            phiExtra = cms.double(0.25),
        )
    ),
)


l1ctLayer1 = cms.EDProducer("L1TPFCandMultiMerger",
    pfProducers = cms.VInputTag(
        cms.InputTag("l1ctLayer1Barrel"),
        cms.InputTag("l1ctLayer1HGCal"),
        cms.InputTag("l1ctLayer1HGCalNoTK"),
        cms.InputTag("l1ctLayer1HF")
    ),
    labelsToMerge = cms.vstring("PF", "Puppi"),
    regionalLabelsToMerge = cms.vstring("Puppi"),
)

l1ctLayer1EG = cms.EDProducer(
    "L1TEGMultiMerger",
    tkElectrons=cms.VPSet(
        cms.PSet(
            instance=cms.string("L1TkEleEE"),
            pfProducers=cms.VInputTag(
                cms.InputTag("l1ctLayer1HGCal", 'L1TkEle')
            )
        ),
        cms.PSet(
            instance=cms.string("L1TkEleEB"),
            pfProducers=cms.VInputTag(
                cms.InputTag("l1ctLayer1Barrel", 'L1TkEle')
            )
        )
    ),
    tkEms=cms.VPSet(
        cms.PSet(
            instance=cms.string("L1TkEmEE"),
            pfProducers=cms.VInputTag(
                cms.InputTag("l1ctLayer1HGCal", 'L1TkEm'),
                cms.InputTag("l1ctLayer1HGCalNoTK", 'L1TkEm')
            )
        ),
        cms.PSet(
            instance=cms.string("L1TkEmEB"),
            pfProducers=cms.VInputTag(
                cms.InputTag("l1ctLayer1Barrel", 'L1TkEm')
            )
        )
    ),
    tkEgs=cms.VPSet(
        cms.PSet(
            instance=cms.string("L1EgEE"),
            pfProducers=cms.VInputTag(
                cms.InputTag("l1ctLayer1HGCal", 'L1Eg'),
                cms.InputTag("l1ctLayer1HGCalNoTK", 'L1Eg')
            )
        )    
    )
)
