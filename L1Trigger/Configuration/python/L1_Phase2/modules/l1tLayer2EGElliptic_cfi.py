import FWCore.ParameterSet.Config as cms

l1tLayer2EGElliptic = cms.EDProducer("L1TCtL2EgProducer",
    egStaInstanceLabel = cms.string('L1CtEgEE'),
    encoder = cms.PSet(
        nTKELE_OUT = cms.uint32(12),
        nTKPHO_OUT = cms.uint32(12)
    ),
    inPatternFile = cms.PSet(
        TMUX = cms.uint32(6),
        channels = cms.VPSet(
            cms.PSet(
                TMUX = cms.uint32(6),
                channels = cms.vuint32(0),
                id = cms.uint32(0),
                interface = cms.string('eglayer1Barrel'),
                nWords = cms.uint32(48)
            ),
            cms.PSet(
                TMUX = cms.uint32(6),
                channels = cms.vuint32(1),
                id = cms.uint32(1),
                interface = cms.string('eglayer1Barrel'),
                nWords = cms.uint32(48)
            ),
            cms.PSet(
                TMUX = cms.uint32(6),
                channels = cms.vuint32(2),
                id = cms.uint32(2),
                interface = cms.string('eglayer1Barrel'),
                nWords = cms.uint32(48)
            ),
            cms.PSet(
                TMUX = cms.uint32(6),
                channels = cms.vuint32(3),
                id = cms.uint32(3),
                interface = cms.string('eglayer1Endcap'),
                nWords = cms.uint32(48)
            ),
            cms.PSet(
                TMUX = cms.uint32(6),
                channels = cms.vuint32(4),
                id = cms.uint32(4),
                interface = cms.string('eglayer1Endcap'),
                nWords = cms.uint32(48)
            )
        ),
        eventsPerFile = cms.uint32(12),
        format = cms.string('EMPv2'),
        maxLinesPerFile = cms.uint32(1024),
        nFramesPerBX = cms.uint32(9),
        outputFileExtension = cms.string('txt.gz'),
        outputFilename = cms.string('L1TCTL2EG_InPattern')
    ),
    l1PFObjects = cms.InputTag("l1tLayer2Deregionizer","Puppi"),
    outPatternFile = cms.PSet(
        TMUX = cms.uint32(6),
        channels = cms.VPSet(cms.PSet(
            TMUX = cms.uint32(6),
            channels = cms.vuint32(0),
            id = cms.uint32(0),
            interface = cms.string('eglayer2'),
            nWords = cms.uint32(36)
        )),
        eventsPerFile = cms.uint32(12),
        format = cms.string('EMPv2'),
        maxLinesPerFile = cms.uint32(1024),
        nFramesPerBX = cms.uint32(9),
        outputFileExtension = cms.string('txt.gz'),
        outputFilename = cms.string('L1TCTL2EG_OutPattern')
    ),
    puppiIsoParametersTkEle = cms.PSet(
        dRMax = cms.double(0.2),
        dRMin = cms.double(0.03),
        dZ = cms.double(0.6),
        pfCandReuse = cms.bool(True),
        pfIsoType = cms.string('PUPPI'),
        pfPtMin = cms.double(1.0)
    ),
    puppiIsoParametersTkEm = cms.PSet(
        dRMax = cms.double(0.3),
        dRMin = cms.double(0.07),
        dZ = cms.double(0.6),
        pfCandReuse = cms.bool(True),
        pfIsoType = cms.string('PUPPI'),
        pfPtMin = cms.double(1.0)
    ),
    sorter = cms.PSet(
        debug = cms.untracked.uint32(0),
        nEGOut = cms.uint32(12),
        nEGPerRegion = cms.uint32(16),
        nREGIONS = cms.uint32(5)
    ),
    tkEgs = cms.VPSet(
        cms.PSet(
            pfProducer = cms.InputTag("l1tLayer1HGCalElliptic","L1Eg"),
            regions = cms.vint32(-1)
        ),
        cms.PSet(
            pfProducer = cms.InputTag("l1tLayer1HGCalNoTK","L1Eg"),
            regions = cms.vint32(-1)
        )
    ),
    tkEleInstanceLabel = cms.string('L1CtTkElectron'),
    tkElectrons = cms.VPSet(
        cms.PSet(
            pfProducer = cms.InputTag("l1tLayer1HGCalElliptic","L1TkElePerBoard"),
            regions = cms.vint32(3, 4)
        ),
        cms.PSet(
            pfProducer = cms.InputTag("l1tLayer1Barrel","L1TkElePerBoard"),
            regions = cms.vint32(0, 1, 2)
        )
    ),
    tkEmInstanceLabel = cms.string('L1CtTkEm'),
    tkEms = cms.VPSet(
        cms.PSet(
            pfProducer = cms.InputTag("l1tLayer1HGCalElliptic","L1TkEmPerBoard"),
            regions = cms.vint32(3, 4)
        ),
        cms.PSet(
            pfProducer = cms.InputTag("l1tLayer1HGCalNoTK","L1TkEmPerBoard"),
            regions = cms.vint32(-1)
        ),
        cms.PSet(
            pfProducer = cms.InputTag("l1tLayer1Barrel","L1TkEmPerBoard"),
            regions = cms.vint32(0, 1, 2)
        )
    ),
    writeInPattern = cms.bool(False),
    writeOutPattern = cms.bool(False)
)
