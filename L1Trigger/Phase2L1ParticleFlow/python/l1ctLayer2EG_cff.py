import FWCore.ParameterSet.Config as cms

from L1Trigger.Phase2L1ParticleFlow.l1tDeregionizerProducer_cfi import l1tDeregionizerProducer as l1tLayer2Deregionizer

l1tLayer2EG = cms.EDProducer(
    "L1TCtL2EgProducer",
    tkElectrons=cms.VPSet(
        cms.PSet(
            pfProducer=cms.InputTag("l1tLayer1HGCal", 'L1TkElePerBoard'),
            regions=cms.vint32(3, 4)
        ),
        cms.PSet(
            pfProducer=cms.InputTag("l1tLayer1Barrel", 'L1TkElePerBoard'),
            regions=cms.vint32(0, 1, 2)
        ),
    ),
    tkEms=cms.VPSet(
        cms.PSet(
            pfProducer=cms.InputTag("l1tLayer1HGCal", 'L1TkEmPerBoard'),
            regions=cms.vint32(3, 4)
        ),
        cms.PSet(
            pfProducer=cms.InputTag("l1tLayer1HGCalNoTK", 'L1TkEmPerBoard'),
            regions=cms.vint32(-1)
        ),
        cms.PSet(
            pfProducer=cms.InputTag("l1tLayer1Barrel", 'L1TkEmPerBoard'),
            regions=cms.vint32(0, 1, 2)
        ),
    ),
    tkEgs=cms.VPSet(
        cms.PSet(
            pfProducer=cms.InputTag("l1tLayer1HGCal", 'L1Eg'),
            regions=cms.vint32(-1)
        ),
        cms.PSet(
            pfProducer=cms.InputTag("l1tLayer1HGCalNoTK", 'L1Eg'),
            regions=cms.vint32(-1)
        ),
    ),
    l1PFObjects = cms.InputTag("l1tLayer2Deregionizer", "Puppi"),
    egStaInstanceLabel=cms.string("L1CtEgEE"),
    tkEmInstanceLabel=cms.string("L1CtTkEm"),
    tkEleInstanceLabel=cms.string("L1CtTkElectron"),
    sorter=cms.PSet(
        nREGIONS=cms.uint32(5),
        nEGPerRegion=cms.uint32(16),
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
        format=cms.string("EMPv2"),
        outputFilename=cms.string("L1TCTL2EG_InPattern"),
        outputFileExtension=cms.string("txt.gz"),
        TMUX=cms.uint32(6),
        maxLinesPerFile=cms.uint32(1024),
        eventsPerFile=cms.uint32(12),
        channels=cms.VPSet(
            cms.PSet(
                TMUX=cms.uint32(6),
                nWords=cms.uint32(48),  # = 16*2words ele + 16words photons
                interface=cms.string("eglayer1Barrel"),
                id=cms.uint32(0),
                channels=cms.vuint32(0)
                ),
            cms.PSet(
                TMUX=cms.uint32(6),
                nWords=cms.uint32(48),
                interface=cms.string("eglayer1Barrel"),
                id=cms.uint32(1),
                channels=cms.vuint32(1)
                ),
            cms.PSet(
                TMUX=cms.uint32(6),
                nWords=cms.uint32(48),
                interface=cms.string("eglayer1Barrel"),
                id=cms.uint32(2),
                channels=cms.vuint32(2)
                ),
            cms.PSet(
                TMUX=cms.uint32(6),
                nWords=cms.uint32(48),
                interface=cms.string("eglayer1Endcap"),
                id=cms.uint32(3),
                channels=cms.vuint32(3)
                ),
            cms.PSet(
                TMUX=cms.uint32(6),
                nWords=cms.uint32(48),
                interface=cms.string("eglayer1Endcap"),
                id=cms.uint32(4),
                channels=cms.vuint32(4)
                ),

        )
    ),
    outPatternFile=cms.PSet(
        nFramesPerBX=cms.uint32(9),  # 360 MHz clock or 25 Gb/s link
        format=cms.string("EMPv2"),
        outputFilename=cms.string("L1TCTL2EG_OutPattern"),
        outputFileExtension=cms.string("txt.gz"),
        TMUX=cms.uint32(6),
        maxLinesPerFile=cms.uint32(1024),
        eventsPerFile=cms.uint32(12),
        channels=cms.VPSet(
            cms.PSet(
                TMUX=cms.uint32(6),
                nWords=cms.uint32(36),  # 36 = 12*3/2words ele + 12*3/2words photons
                interface=cms.string("eglayer2"),
                id=cms.uint32(0),
                channels=cms.vuint32(0)
                )
        )
    ),
    # NOTE: to write out the GT input from 6TS 
    # outPatternFile=cms.PSet(
    #     nFramesPerBX=cms.uint32(9),  # 360 MHz clock or 25 Gb/s link
    #     format=cms.string("EMPv2"),
    #     outputFilename=cms.string("L1TCTL2EG_ToGTPattern"),
    #     outputFileExtension=cms.string("txt.gz"),
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
            regions=cms.vint32(3, 4)
        ),
        cms.PSet(
            pfProducer=cms.InputTag("l1tLayer1Barrel", 'L1TkElePerBoard'),
            regions=cms.vint32(0, 1, 2)
        ),
    ),
    tkEms=cms.VPSet(
        cms.PSet(
            pfProducer=cms.InputTag("l1tLayer1HGCalElliptic", 'L1TkEmPerBoard'),
            regions=cms.vint32(3, 4)
        ),
        cms.PSet(
            pfProducer=cms.InputTag("l1tLayer1HGCalNoTK", 'L1TkEmPerBoard'),
            regions=cms.vint32(-1)
        ),
        cms.PSet(
            pfProducer=cms.InputTag("l1tLayer1Barrel", 'L1TkEmPerBoard'),
            regions=cms.vint32(0, 1, 2)
        ),
    ),
    tkEgs=cms.VPSet(
        cms.PSet(
            pfProducer=cms.InputTag("l1tLayer1HGCalElliptic", 'L1Eg'),
            regions=cms.vint32(-1)
        ),
        cms.PSet(
            pfProducer=cms.InputTag("l1tLayer1HGCalNoTK", 'L1Eg'),
            regions=cms.vint32(-1)
        ),
    ),
)

# EG Layer2 with Layer1 @ TMUX18
l1tLayer2EGTM18 = l1tLayer2EG.clone(
    tkElectrons=cms.VPSet(
        cms.PSet(
            pfProducer=cms.InputTag("l1tLayer1HGCalTM18", 'L1TkElePerBoard'),
            regions=cms.vint32(3, 4)
        ),
        cms.PSet(
            pfProducer=cms.InputTag("l1tLayer1BarrelSerenityTM18", 'L1TkElePerBoard'),
            regions=cms.vint32(0, 1, 2)
        ),
    ),
    tkEms=cms.VPSet(
        cms.PSet(
            pfProducer=cms.InputTag("l1tLayer1HGCalTM18", 'L1TkEmPerBoard'),
            regions=cms.vint32(3, 4)
        ),
        cms.PSet(
            pfProducer=cms.InputTag("l1tLayer1HGCalNoTKTM18", 'L1TkEmPerBoard'),
            regions=cms.vint32(-1)
        ),
        cms.PSet(
            pfProducer=cms.InputTag("l1tLayer1BarrelSerenityTM18", 'L1TkEmPerBoard'),
            regions=cms.vint32(0, 1, 2)
        ),
    ),
    tkEgs=cms.VPSet(
        cms.PSet(
            pfProducer=cms.InputTag("l1tLayer1HGCalTM18", 'L1Eg'),
            regions=cms.vint32(-1)
        ),
        cms.PSet(
            pfProducer=cms.InputTag("l1tLayer1HGCalNoTKTM18", 'L1Eg'),
            regions=cms.vint32(-1)
        ),
    ),
)

l1tLayer2EGTM18.inPatternFile.outputFilename = "L1TCTL2EG_TMUX18_InPattern"
l1tLayer2EGTM18.inPatternFile.channels = cms.VPSet(
    cms.PSet(
        TMUX=cms.uint32(18),
        nWords=cms.uint32(156),  # = (16*2words ele + 16words photons) * 3 (regions) every 6 BX (54 words) = 48+6(empty)+48+6(empty)+48 = 156
        interface=cms.string("eglayer1Barrel"),
        id=cms.uint32(0),
        channels=cms.vuint32(0,2,4)
        ),
    cms.PSet(
        TMUX=cms.uint32(18),
        nWords=cms.uint32(129), # (16*2words ele + 16words photons) * 2 (regions) every 9 BX (81 words) = 48+33(empty)+48
        interface=cms.string("eglayer1Endcap"),
        id=cms.uint32(1),
        channels=cms.vuint32(1,3,5)
        ),
)
l1tLayer2EGTM18.outPatternFile.outputFilename = 'L1TCTL2EG_TMUX18_OutPattern'
# FIXME: we need to schedule a new deregionizer for TM18
# l1tLayer2EGTM18.l1PFObjects = cms.InputTag("l1tLayer2Deregionizer", "Puppi"),


L1TLayer2EGTask = cms.Task(
     l1tLayer2Deregionizer,
     l1tLayer2EG,
     l1tLayer2EGElliptic
)
