import FWCore.ParameterSet.Config as cms

l1ctLayer2EG = cms.EDProducer(
    "L1TCtL2EgProducer",
    tkElectrons=cms.VPSet(
        cms.PSet(
            pfProducer=cms.InputTag("l1ctLayer1HGCal", 'L1TkElePerBoard'),
            channels=cms.vint32(3, 4)
        ),
        cms.PSet(
            pfProducer=cms.InputTag("l1ctLayer1Barrel", 'L1TkElePerBoard'),
            channels=cms.vint32(0, 1, 2)
        ),
    ),
    tkEms=cms.VPSet(
        cms.PSet(
            pfProducer=cms.InputTag("l1ctLayer1HGCal", 'L1TkEmPerBoard'),
            channels=cms.vint32(3, 4)
        ),
        cms.PSet(
            pfProducer=cms.InputTag("l1ctLayer1HGCalNoTK", 'L1TkEmPerBoard'),
            channels=cms.vint32(-1)
        ),
        cms.PSet(
            pfProducer=cms.InputTag("l1ctLayer1Barrel", 'L1TkEmPerBoard'),
            channels=cms.vint32(0, 1, 2)
        ),
    ),
    tkEgs=cms.VPSet(
        cms.PSet(
            pfProducer=cms.InputTag("l1ctLayer1HGCal", 'L1Eg'),
            channels=cms.vint32(-1)
        ),
        cms.PSet(
            pfProducer=cms.InputTag("l1ctLayer1HGCalNoTK", 'L1Eg'),
            channels=cms.vint32(-1)
        ),
    ),
    egStaInstanceLabel=cms.string("L1CtEgEE"),
    tkEmInstanceLabel=cms.string("L1CtTkEm"),
    tkEleInstanceLabel=cms.string("L1CtTkElectron"),
    sorter=cms.PSet(
        nBOARDS=cms.uint32(5),
        nEGPerBoard=cms.uint32(16),
        nEGOut=cms.uint32(12),
        debug=cms.untracked.uint32(0),
    ),
    encoder=cms.PSet(
        nTKELE_OUT=cms.uint32(12),
        nTKPHO_OUT=cms.uint32(12),
    ),
    writeInPattern=cms.bool(False),
    writeOutPattern=cms.bool(False),
    inPatternFile=cms.PSet(
        nFramesPerBX=cms.uint32(9),  # 360 MHz clock or 25 Gb/s link
        format=cms.string("EMP"),
        outputFilename=cms.string("L1TCTL2EG_InPattern"),
        TMUX=cms.uint32(6),
        maxLinesPerFile=cms.uint32(1024),
        channels=cms.VPSet(
            cms.PSet(
                TMUX=cms.uint32(6),
                nWords=cms.uint32(48),  # = 16*2words ele + 16words photons
                interface=cms.string("eglayer1"),
                id=cms.uint32(0),
                channels=cms.vuint32(0)
                ),
            cms.PSet(
                TMUX=cms.uint32(6),
                nWords=cms.uint32(48),
                interface=cms.string("eglayer1"),
                id=cms.uint32(1),
                channels=cms.vuint32(1)
                ),
            cms.PSet(
                TMUX=cms.uint32(6),
                nWords=cms.uint32(48),
                interface=cms.string("eglayer1"),
                id=cms.uint32(2),
                channels=cms.vuint32(2)
                ),
            cms.PSet(
                TMUX=cms.uint32(6),
                nWords=cms.uint32(48),
                interface=cms.string("eglayer1"),
                id=cms.uint32(3),
                channels=cms.vuint32(3)
                ),
            cms.PSet(
                TMUX=cms.uint32(6),
                nWords=cms.uint32(48),
                interface=cms.string("eglayer1"),
                id=cms.uint32(4),
                channels=cms.vuint32(4)
                ),

        )
    ),
    outPatternFile=cms.PSet(
        nFramesPerBX=cms.uint32(9),  # 360 MHz clock or 25 Gb/s link
        format=cms.string("EMP"),
        outputFilename=cms.string("L1TCTL2EG_OuPattern"),
        TMUX=cms.uint32(6),
        maxLinesPerFile=cms.uint32(1024),
        channels=cms.VPSet(
            cms.PSet(
                TMUX=cms.uint32(6),
                nWords=cms.uint32(36),  # 36 = 12*3/2words ele + 12*3/2words phhotons
                interface=cms.string("eglayer2"),
                id=cms.uint32(0),
                channels=cms.vuint32(0)
                )
        )
    ),
    # NOTE: to write out the GT input from 6TS 
    # outPatternFile=cms.PSet(
    #     nFramesPerBX=cms.uint32(9),  # 360 MHz clock or 25 Gb/s link
    #     format=cms.string("EMP"),
    #     outputFilename=cms.string("L1TCTL2EG_ToGTPattern"),
    #     TMUX=cms.uint32(1),
    #     maxLinesPerFile=cms.uint32(1024),
    #     channels=cms.VPSet(
    #         cms.PSet(
    #             TMUX=cms.uint32(6),
    #             nWords=cms.uint32(36),  # 36 = 12*3/2words ele + 12*3/2words phhotons
    #             interface=cms.string("eglayer2"),
    #             id=cms.uint32(0),
    #             channels=cms.vuint32(0, 1, 2, 3, 4, 5)
    #             )
    #     )
    # )
)


l1ctLayer2EGTask = cms.Task(
     l1ctLayer2EG
)
