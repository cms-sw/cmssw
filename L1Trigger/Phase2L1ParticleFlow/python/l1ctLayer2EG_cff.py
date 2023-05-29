import FWCore.ParameterSet.Config as cms

from L1Trigger.Phase2L1ParticleFlow.l1tDeregionizerProducer_cfi import l1tDeregionizerProducer as l1tLayer2Deregionizer

l1tLayer2EG = cms.EDProducer(
    "L1TCtL2EgProducer",
    tkElectrons=cms.VPSet(
        cms.PSet(
            pfProducer=cms.InputTag("l1tLayer1HGCal", 'L1TkElePerBoard'),
            channels=cms.vint32(3, 4)
        ),
        cms.PSet(
            pfProducer=cms.InputTag("l1tLayer1Barrel", 'L1TkElePerBoard'),
            channels=cms.vint32(0, 1, 2)
        ),
    ),
    tkEms=cms.VPSet(
        cms.PSet(
            pfProducer=cms.InputTag("l1tLayer1HGCal", 'L1TkEmPerBoard'),
            channels=cms.vint32(3, 4)
        ),
        cms.PSet(
            pfProducer=cms.InputTag("l1tLayer1HGCalNoTK", 'L1TkEmPerBoard'),
            channels=cms.vint32(-1)
        ),
        cms.PSet(
            pfProducer=cms.InputTag("l1tLayer1Barrel", 'L1TkEmPerBoard'),
            channels=cms.vint32(0, 1, 2)
        ),
    ),
    tkEgs=cms.VPSet(
        cms.PSet(
            pfProducer=cms.InputTag("l1tLayer1HGCal", 'L1Eg'),
            channels=cms.vint32(-1)
        ),
        cms.PSet(
            pfProducer=cms.InputTag("l1tLayer1HGCalNoTK", 'L1Eg'),
            channels=cms.vint32(-1)
        ),
    ),
    l1PFObjects = cms.InputTag("l1tLayer2Deregionizer", "Puppi"),
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
    puppiIsoParametersTkEm = cms.PSet(
        pfIsoType = cms.string("PUPPI"),
        pfPtMin = cms.double(1.),
        dZ = cms.double(0.6),
        dRMin = cms.double(0.07),
        dRMax = cms.double(0.3),
        pfCandReuse = cms.bool(True)
    ),
    puppiIsoParametersTkEle = cms.PSet(
        pfIsoType = cms.string("PUPPI"),
        pfPtMin = cms.double(1.),
        dZ = cms.double(0.6),
        dRMin = cms.double(0.03),
        dRMax = cms.double(0.2),
        pfCandReuse = cms.bool(True)
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
        outputFilename=cms.string("L1TCTL2EG_OutPattern"),
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

l1tLayer2EGElliptic = l1tLayer2EG.clone(
     tkElectrons=cms.VPSet(
        cms.PSet(
            pfProducer=cms.InputTag("l1tLayer1HGCalElliptic", 'L1TkElePerBoard'),
            channels=cms.vint32(3, 4)
        ),
        cms.PSet(
            pfProducer=cms.InputTag("l1tLayer1Barrel", 'L1TkElePerBoard'),
            channels=cms.vint32(0, 1, 2)
        ),
    ),
    tkEms=cms.VPSet(
        cms.PSet(
            pfProducer=cms.InputTag("l1tLayer1HGCalElliptic", 'L1TkEmPerBoard'),
            channels=cms.vint32(3, 4)
        ),
        cms.PSet(
            pfProducer=cms.InputTag("l1tLayer1HGCalNoTK", 'L1TkEmPerBoard'),
            channels=cms.vint32(-1)
        ),
        cms.PSet(
            pfProducer=cms.InputTag("l1tLayer1Barrel", 'L1TkEmPerBoard'),
            channels=cms.vint32(0, 1, 2)
        ),
    ),
    tkEgs=cms.VPSet(
        cms.PSet(
            pfProducer=cms.InputTag("l1tLayer1HGCalElliptic", 'L1Eg'),
            channels=cms.vint32(-1)
        ),
        cms.PSet(
            pfProducer=cms.InputTag("l1tLayer1HGCalNoTK", 'L1Eg'),
            channels=cms.vint32(-1)
        ),
    ),
)


L1TLayer2EGTask = cms.Task(
     l1tLayer2Deregionizer,
     l1tLayer2EG,
     l1tLayer2EGElliptic
)
