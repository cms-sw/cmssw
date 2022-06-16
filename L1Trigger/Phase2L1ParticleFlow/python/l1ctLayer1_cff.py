import FWCore.ParameterSet.Config as cms

import math

from L1Trigger.Phase2L1ParticleFlow.pfTracksFromL1Tracks_cfi import pfTracksFromL1Tracks
from L1Trigger.Phase2L1ParticleFlow.pfClustersFromL1EGClusters_cfi import pfClustersFromL1EGClusters
from L1Trigger.Phase2L1ParticleFlow.pfClustersFromCombinedCalo_cff import pfClustersFromCombinedCaloHCal, pfClustersFromCombinedCaloHF
from L1Trigger.Phase2L1ParticleFlow.pfClustersFromHGC3DClusters_cfi import pfClustersFromHGC3DClusters

from L1Trigger.Phase2L1ParticleFlow.l1TkEgAlgoEmulator_cfi import tkEgAlgoParameters,tkEgSorterParameters

muonInputConversionParameters = cms.PSet(
    z0Scale = cms.double(1.875),
    dxyScale = cms.double(3.85)
)

l1ctLayer1Barrel = cms.EDProducer("L1TCorrelatorLayer1Producer",
    tracks = cms.InputTag('pfTracksFromL1Tracks'),
    muons = cms.InputTag('L1SAMuonsGmt','promptSAMuons'),
    emClusters = cms.VInputTag(cms.InputTag('pfClustersFromL1EGClusters')),
    hadClusters = cms.VInputTag(cms.InputTag('pfClustersFromCombinedCaloHCal:calibrated')),
    vtxCollection = cms.InputTag("L1VertexFinderEmulator","l1verticesEmulation"),
    vtxCollectionEmulation = cms.bool(True),
    emPtCut  = cms.double(0.5),
    hadPtCut = cms.double(1.0),
    trkPtCut    = cms.double(2.0),
    trackInputConversionAlgo = cms.string("Emulator"),
    trackInputConversionParameters = cms.PSet(
        region = cms.string("barrel"),
        trackWordEncoding = cms.string("biased"),
        bitwiseAccurate = cms.bool(True),
        ptLUTBits = cms.uint32(11),
        etaLUTBits = cms.uint32(10),
        etaPreOffs = cms.int32(0),
        etaShift = cms.uint32(15-10),
        etaPostOffs = cms.int32(0),
        etaSigned = cms.bool(True),
        phiBits = cms.uint32(10),
        z0Bits = cms.uint32(12),
        dEtaBarrelBits = cms.uint32(8),
        dEtaBarrelZ0PreShift = cms.uint32(2),
        dEtaBarrelZ0PostShift = cms.uint32(2),
        dEtaBarrelFloatOffs = cms.double(0.0),
        dPhiBarrelBits = cms.uint32(4),
        dPhiBarrelRInvPreShift = cms.uint32(4),
        dPhiBarrelRInvPostShift = cms.uint32(4),
        dPhiBarrelFloatOffs = cms.double(0.0)
        ),
    muonInputConversionAlgo = cms.string("Emulator"),
    muonInputConversionParameters = muonInputConversionParameters.clone(),
    regionizerAlgo = cms.string("Ideal"),
    pfAlgo = cms.string("PFAlgo3"),
    puAlgo = cms.string("LinearizedPuppi"),
    nVtx        = cms.int32(1),    
    regionizerAlgoParameters = cms.PSet(
        useAlsoVtxCoords = cms.bool(True),
    ),
    pfAlgoParameters = cms.PSet(
        nTrack = cms.uint32(25), 
        nCalo = cms.uint32(18), 
        nMu = cms.uint32(2), 
        nSelCalo = cms.uint32(18), 
        nEmCalo = cms.uint32(12), 
        nPhoton = cms.uint32(12), 
        nAllNeutral = cms.uint32(25), 
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
        nTrack = cms.uint32(22), 
        nIn = cms.uint32(25), 
        nOut = cms.uint32(25), 
        nVtx = cms.uint32(1),
        nFinalSort = cms.uint32(18), 
        finalSortAlgo = cms.string("Insertion"),
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
    tkEgAlgoParameters=tkEgAlgoParameters.clone(
        nTRACK = 25,
        nTRACK_EGIN = 13,
        nEMCALO_EGIN = 10,
        nEM_EGOUT = 10,
    ),
    tkEgSorterParameters=tkEgSorterParameters.clone(
        nObjToSort = 10
    ),
    caloSectors = cms.VPSet(
        cms.PSet( 
            etaBoundaries = cms.vdouble(-1.5, 1.5),
            phiSlices     = cms.uint32(6),
            phiZero       = cms.double(0),
        )
    ),
    regions = cms.VPSet(
        cms.PSet( 
            etaBoundaries = cms.vdouble(-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5),
            phiSlices     = cms.uint32(9),
            etaExtra = cms.double(0.25),
            phiExtra = cms.double(0.25),
        ),
    ),
    boards=cms.VPSet(
        cms.PSet(
              regions=cms.vuint32(*[0+9*ie+i for ie in range(6) for i in range(3)])), # phi splitting
            # regions=cms.vuint32(range(0, 18))), # eta splitting
        cms.PSet(
              regions=cms.vuint32(*[3+9*ie+i for ie in range(6) for i in range(3)])), # phi splitting
            # regions=cms.vuint32(range(18, 36))), # eta splitting
        cms.PSet(
              regions=cms.vuint32(*[6+9*ie+i for ie in range(6) for i in range(3)])), # phi splitting
            # regions=cms.vuint32(range(36, 54))), # eta splitting
    )
)

_hgcalSectors = cms.VPSet(
    cms.PSet( 
        etaBoundaries = cms.vdouble(-3.0, -1.5),
        phiSlices     = cms.uint32(3),
        phiZero       = cms.double(math.pi/6) # L1 TrackFinder phi sector and HGCal sectors shifted by 30deg,
    ),
    cms.PSet( 
        etaBoundaries = cms.vdouble(+1.5, +3.0),
        phiSlices     = cms.uint32(3),
        phiZero       = cms.double(math.pi/6) # L1 TrackFinder phi sector and HGCal sectors shifted by 30deg,
    )

)

l1ctLayer1HGCal = cms.EDProducer("L1TCorrelatorLayer1Producer",
    tracks = cms.InputTag('pfTracksFromL1Tracks'),
    muons = cms.InputTag('L1SAMuonsGmt','promptSAMuons'),
    emClusters = cms.VInputTag(cms.InputTag('pfClustersFromHGC3DClusters:egamma')), # used only for E/gamma
    hadClusters = cms.VInputTag(cms.InputTag('pfClustersFromHGC3DClusters')),
    vtxCollection = cms.InputTag("L1VertexFinderEmulator","l1verticesEmulation"),
    vtxCollectionEmulation = cms.bool(True),
    nVtx        = cms.int32(1),    
    emPtCut  = cms.double(0.5),
    hadPtCut = cms.double(1.0),
    trkPtCut    = cms.double(2.0),
    trackInputConversionAlgo = cms.string("Emulator"),
    trackInputConversionParameters = cms.PSet(
        region = cms.string("endcap"),
        trackWordEncoding = cms.string("biased"),
        bitwiseAccurate = cms.bool(True),
        ptLUTBits = cms.uint32(11),
        etaLUTBits = cms.uint32(11),
        etaPreOffs = cms.int32(0),
        etaShift = cms.uint32(15-11),
        etaPostOffs = cms.int32(150),
        etaSigned = cms.bool(True),
        phiBits = cms.uint32(10),
        z0Bits = cms.uint32(12),
        dEtaHGCalBits = cms.uint32(10),
        dEtaHGCalZ0PreShift = cms.uint32(2),
        dEtaHGCalRInvPreShift = cms.uint32(6),
        dEtaHGCalLUTBits = cms.uint32(10),
        dEtaHGCalLUTShift = cms.uint32(2),
        dEtaHGCalFloatOffs = cms.double(0.0),
        dPhiHGCalBits = cms.uint32(4),
        dPhiHGCalZ0PreShift = cms.uint32(4),
        dPhiHGCalZ0PostShift = cms.uint32(6),
        dPhiHGCalRInvShift = cms.uint32(4),
        dPhiHGCalTanlInvShift = cms.uint32(22),
        dPhiHGCalTanlLUTBits = cms.uint32(10),
        dPhiHGCalFloatOffs = cms.double(0.0)
        ),
    muonInputConversionAlgo = cms.string("Emulator"),
    muonInputConversionParameters = muonInputConversionParameters.clone(),
    regionizerAlgo = cms.string("Multififo"),
    regionizerAlgoParameters = cms.PSet(
        useAlsoVtxCoords = cms.bool(True),
        nEndcaps = cms.uint32(2),
        nClocks = cms.uint32(54),
        nTrack = cms.uint32(30),
        nCalo = cms.uint32(20),
        nEmCalo = cms.uint32(10),
        nMu = cms.uint32(4),
        egInterceptMode = cms.PSet(
            afterFifo = cms.bool(True),
            emIDMask = cms.uint32(0x1E),
            nHADCALO_IN = cms.uint32(20),
            nEMCALO_OUT = cms.uint32(10),
            )
        ),
    pfAlgo = cms.string("PFAlgo2HGC"),
    pfAlgoParameters = cms.PSet(
        nTrack = cms.uint32(30),
        nCalo = cms.uint32(20),
        nMu = cms.uint32(4),
        nSelCalo = cms.uint32(20),
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
    puAlgo = cms.string("LinearizedPuppi"),
    puAlgoParameters = cms.PSet(
        nTrack = cms.uint32(30),
        nIn = cms.uint32(20),
        nOut = cms.uint32(20),
        nVtx        = cms.uint32(1),    
        nFinalSort = cms.uint32(18), 
        finalSortAlgo = cms.string("FoldedHybrid"),
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
        nTRACK = 30,
        nTRACK_EGIN = 10,
        nEMCALO_EGIN = 10, 
        nEM_EGOUT = 5,
        doBremRecovery=True,
        writeBeforeBremRecovery=False,
        writeEGSta=True),
    tkEgSorterParameters=tkEgSorterParameters.clone(
        nObjToSort = 5
    ),
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
    boards=cms.VPSet(
        cms.PSet(
            regions=cms.vuint32(range(0, 9))),
        cms.PSet(
            regions=cms.vuint32(range(9, 18))),
    ),
    writeRawHgcalCluster = cms.untracked.bool(True)
)


l1ctLayer1HGCalNoTK = cms.EDProducer("L1TCorrelatorLayer1Producer",
    tracks = cms.InputTag(''),
    muons = cms.InputTag('L1SAMuonsGmt','promptSAMuons'),
    emClusters = cms.VInputTag(cms.InputTag('pfClustersFromHGC3DClusters:egamma')), # used only for E/gamma
    hadClusters = cms.VInputTag(cms.InputTag('pfClustersFromHGC3DClusters')),
    vtxCollection = cms.InputTag("L1VertexFinderEmulator","l1verticesEmulation"),
    vtxCollectionEmulation = cms.bool(True),
    nVtx        = cms.int32(1),        
    emPtCut  = cms.double(0.5),
    hadPtCut = cms.double(1.0),
    trkPtCut    = cms.double(2.0),
    muonInputConversionAlgo = cms.string("Emulator"),
    muonInputConversionParameters = muonInputConversionParameters.clone(),
    regionizerAlgo = cms.string("Ideal"),
    pfAlgo = cms.string("PFAlgoDummy"),
    puAlgo = cms.string("LinearizedPuppi"),
    regionizerAlgoParameters = cms.PSet(
        useAlsoVtxCoords = cms.bool(True),
    ),
    pfAlgoParameters = cms.PSet(
        nCalo = cms.uint32(12), 
        nMu = cms.uint32(4), # unused
        debug = cms.untracked.bool(False)
    ),
    puAlgoParameters = cms.PSet(
        nTrack = cms.uint32(0),  # unused
        nIn = cms.uint32(12), 
        nOut = cms.uint32(12), 
        nFinalSort = cms.uint32(12), # to be tuned
        finalSortAlgo = cms.string("Hybrid"), 
        nVtx = cms.uint32(1),    
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
        nTRACK = 30,
        nTRACK_EGIN = 10,
        nEMCALO_EGIN = 10, 
        nEM_EGOUT = 5,
        doBremRecovery=True,
        writeBeforeBremRecovery=False,
        writeEGSta=True),
    tkEgSorterParameters=tkEgSorterParameters.clone(
        nObjToSort=5
    ),
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
    boards=cms.VPSet(
        cms.PSet(
            regions=cms.vuint32(range(0,18))),
    ),
    writeRawHgcalCluster = cms.untracked.bool(True)
)

l1ctLayer1HF = cms.EDProducer("L1TCorrelatorLayer1Producer",
    tracks = cms.InputTag(''),
    muons = cms.InputTag('L1SAMuonsGmt','promptSAMuons'),
    useStandaloneMuons = cms.bool(False),
    useTrackerMuons = cms.bool(False),
    emClusters = cms.VInputTag(),
    hadClusters = cms.VInputTag(cms.InputTag('pfClustersFromCombinedCaloHF:calibrated')),
    vtxCollection = cms.InputTag("L1VertexFinderEmulator","l1verticesEmulation"),
    vtxCollectionEmulation = cms.bool(True),
    nVtx        = cms.int32(1),    
    emPtCut  = cms.double(0.5),
    hadPtCut = cms.double(15.0),
    trkPtCut    = cms.double(2.0),
    muonInputConversionAlgo = cms.string("Ideal"),
    muonInputConversionParameters = muonInputConversionParameters.clone(),
    regionizerAlgo = cms.string("Ideal"),
    pfAlgo = cms.string("PFAlgoDummy"),
    puAlgo = cms.string("LinearizedPuppi"),
    regionizerAlgoParameters = cms.PSet(
        useAlsoVtxCoords = cms.bool(True),
    ),
    pfAlgoParameters = cms.PSet(
        nCalo = cms.uint32(18), 
        nMu = cms.uint32(4), # unused
        debug = cms.untracked.bool(False)
    ),
    puAlgoParameters = cms.PSet(
        nTrack = cms.uint32(0), # unused
        nIn = cms.uint32(18), 
        nOut = cms.uint32(18), 
        nVtx = cms.uint32(1),
        nFinalSort = cms.uint32(10), # to be tuned
        finalSortAlgo = cms.string("Insertion"),
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
        nTRACK = 5,           # to be defined
        nTRACK_EGIN = 5,          # to be defined
        nEMCALO_EGIN = 5,  # to be defined
        nEM_EGOUT = 5,        # to be defined
        doBremRecovery=True,
        writeEGSta=True),
    tkEgSorterParameters=tkEgSorterParameters.clone(),
    caloSectors = cms.VPSet(
        cms.PSet( 
            etaBoundaries = cms.vdouble(-5.5, -3.0),
            phiSlices     = cms.uint32(9),
            phiZero       = cms.double(0),
        ),
        cms.PSet( 
            etaBoundaries = cms.vdouble(+3.0, +5.5),
            phiSlices     = cms.uint32(9),
            phiZero       = cms.double(0),
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
    boards=cms.VPSet(),
)


l1ctLayer1 = cms.EDProducer("L1TPFCandMultiMerger",
    pfProducers = cms.VInputTag(
        cms.InputTag("l1ctLayer1Barrel"),
        cms.InputTag("l1ctLayer1HGCal"),
        cms.InputTag("l1ctLayer1HGCalNoTK"),
        cms.InputTag("l1ctLayer1HF")
    ),
    labelsToMerge = cms.vstring("PF", "Puppi", "Calo", "TK"),
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

l1ctLayer1TaskInputsTask = cms.Task(
    pfClustersFromL1EGClusters,
    pfClustersFromCombinedCaloHCal,
    pfClustersFromCombinedCaloHF,
    pfClustersFromHGC3DClusters,
    pfTracksFromL1Tracks
)

l1ctLayer1Task = cms.Task(
     l1ctLayer1Barrel,
     l1ctLayer1HGCal,
     l1ctLayer1HGCalNoTK,
     l1ctLayer1HF,
     l1ctLayer1,
     l1ctLayer1EG
)
