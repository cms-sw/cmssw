import FWCore.ParameterSet.Config as cms

_eventsPerFile = 12
_gttLatency = 156+120
_gttNumberOfPVs = 10

#####################################################################################################################
## Barrel configurations: 54 regions, 6 puppi output links, only write out the layer 1 outputs for now

srOrder = (5, 14, 23, 32, 41, 50, 6, 15, 24, 33, 42, 51, 7, 16, 25, 34, 43, 52, 8, 17, 26, 35, 44, 53, 0, 9, 18,
           27, 36, 45, 1, 10, 19, 28, 37, 46, 2, 11, 20, 29, 38, 47, 3, 12, 21, 30, 39, 48, 4, 13, 22, 31, 40, 49)

_barrelWriterOutputOnly = cms.PSet(
    partition = cms.string("Barrel"),
    tmuxFactor = cms.uint32(18),
    outputLinksPuppi = cms.vuint32(*range(6)),
    outputLinkEgamma = cms.int32(6),
    nEgammaObjectsOut = cms.uint32(16),
    nInputFramesPerBX = cms.uint32(9),
    nOutputFramesPerBX = cms.uint32(9),
    fileFormat = cms.string("APx"),
    inputFileExtension = cms.string("txt.gz"),
    outputFileExtension = cms.string("txt.gz"),
    maxLinesPerInputFile = cms.uint32(1024),
    maxLinesPerOutputFile = cms.uint32(1024),
    eventsPerFile = cms.uint32(_eventsPerFile),
    gmtNumberOfMuons = cms.uint32(12),
    gmtLink = cms.int32(-1),
    gctSectors = cms.VPSet(),
    tfSectors = cms.VPSet(),
    gttLink = cms.int32(-1),
    gttLatency = cms.uint32(167),
    gttNumberOfPVs = cms.uint32(_gttNumberOfPVs),
    tfNumberOfTracks = cms.uint32(108),
    gctNumberOfObjects = cms.uint32(162),
    outputRegions = cms.vuint32(*srOrder),
    outputBoard = cms.int32(0),
)
## Barrel (54) split in 3 phi slices (EMP format)
barrelWriterOutputOnlyPhiConfigs = [
    _barrelWriterOutputOnly.clone(
        fileFormat = cms.string("EMPv2"),
        tmuxFactor = cms.uint32(6),
        outputRegions = cms.vuint32(*[3*ip+9*ie+i for ie in range(6) for i in range(3) ]),
        outputBoard = cms.int32(ip),
        outputFileName = cms.string("l1BarrelPhi%d-outputs" % (ip+1))
    ) for ip in range(3)
]

# I do not think these work any more, particularly for the gct
barrelSerenityPhi1Config = barrelWriterOutputOnlyPhiConfigs[0].clone(
    tfTimeSlices = cms.VPSet(*[cms.PSet(tfSectors = cms.VPSet(*[ cms.PSet(tfLink = cms.int32(-1)) for s in range(18) ])) for t in range(3)]),
    gctEmSectors = cms.VPSet(*[ cms.PSet(gctEmLink = cms.int32(-1)) for s in range(12) ]),
    gctHadSectors = cms.VPSet(*[ cms.PSet(gctHadLink = cms.int32(-1)) for s in range(12) ]),
    gmtTimeSlices = cms.VPSet(*[cms.PSet(gmtLink = cms.int32(4*17+t)) for t in range(3)]),
)
barrelSerenityVU9PPhi1Config = barrelSerenityPhi1Config.clone(
    gttLink = cms.int32(4*28+3),
    inputFileName = cms.string("l1BarrelPhi1Serenity-inputs-vu9p"),
    outputFileName = cms.string("l1BarrelPhi1Serenity-outputs")
)
barrelSerenityVU13PPhi1Config = barrelSerenityPhi1Config.clone(
    gttLink = cms.int32(4*25+3),
    gmtTimeSlices = cms.VPSet(*[cms.PSet(gmtLink = cms.int32(4*18+t)) for t in range(3)]),
    inputFileName = cms.string("l1BarrelPhi1Serenity-inputs-vu13p"),
)


barrelWriterConfigs =  barrelWriterOutputOnlyPhiConfigs

#####################################################################################################################
## HGcal configuration: write out both inputs and outputs
_hgcalWriterConfig = cms.PSet(
    partition = cms.string("HGCal"),
    tmuxFactor = cms.uint32(6),
    nEgammaObjectsOut = cms.uint32(16),
    nInputFramesPerBX = cms.uint32(9),
    nOutputFramesPerBX = cms.uint32(9),
    fileFormat = cms.string("EMPv2"),
    inputFileExtension = cms.string("txt.gz"),
    outputFileExtension = cms.string("txt.gz"),
    maxLinesPerInputFile = cms.uint32(1024),
    maxLinesPerOutputFile = cms.uint32(1024),
    eventsPerFile = cms.uint32(_eventsPerFile),
    tfTimeSlices = cms.VPSet(*[cms.PSet(tfSectors = cms.VPSet()) for i in range(3)]),
    hgcTimeSlices = cms.VPSet(*[cms.PSet(hgcSectors = cms.VPSet()) for i in range(3)]),
    gmtTimeSlices = cms.VPSet(*[cms.PSet(gmtLink = cms.int32(-1)) for i in range(3)]),
    gmtNumberOfMuons = cms.uint32(12),
    gttLink = cms.int32(-1),
    gttLatency = cms.uint32(_gttLatency),
    gttNumberOfPVs = cms.uint32(_gttNumberOfPVs),
    outputLinksPuppi = cms.vuint32(*range(3)),
    outputLinkEgamma = cms.int32(3),
)
## Ideal configuration: 27 input links from tf, 36 from hgc, 3 from gmt, 1 from gtt, in this order; output 3 puppi + 1 e/gamma
_hgcalPosWriterConfig = _hgcalWriterConfig.clone(
    outputRegions = cms.vuint32(*[i+9 for i in range(9)]),
    outputBoard = cms.int32(1),
)
_hgcalNegWriterConfig = _hgcalPosWriterConfig.clone(
    outputRegions = [i for i in range(9)],
    outputBoard = 0,
)
hgcalPosOutputWriterConfig = _hgcalPosWriterConfig.clone(
   outputFileName = cms.string("l1HGCalPos-outputs")
)
hgcalNegOutputWriterConfig = _hgcalNegWriterConfig.clone(
   outputFileName = cms.string("l1HGCalNeg-outputs")
)
## Current configurations for VU9P
hgcalPosVU9PWriterConfig = _hgcalPosWriterConfig.clone()
hgcalNegVU9PWriterConfig = _hgcalNegWriterConfig.clone()
for t in range(3):
    hgcalPosVU9PWriterConfig.tfTimeSlices[t].tfSectors += [ cms.PSet(tfLink = cms.int32(-1))         for i in range(9) ] # neg
    hgcalPosVU9PWriterConfig.tfTimeSlices[t].tfSectors += [ cms.PSet(tfLink = cms.int32(3*i+t+4*1))  for i in range(4) ] # pos, left quads
    hgcalPosVU9PWriterConfig.tfTimeSlices[t].tfSectors += [ cms.PSet(tfLink = cms.int32(3*i+t+4*26)) for i in range(5) ] # pos, right quads
    hgcalNegVU9PWriterConfig.tfTimeSlices[t].tfSectors += [ cms.PSet(tfLink = cms.int32(3*i+t+4*1))  for i in range(4) ] # neg, left quads
    hgcalNegVU9PWriterConfig.tfTimeSlices[t].tfSectors += [ cms.PSet(tfLink = cms.int32(3*i+t+4*26)) for i in range(5) ] # neg, right quads
    hgcalNegVU9PWriterConfig.tfTimeSlices[t].tfSectors += [ cms.PSet(tfLink = cms.int32(-1))         for i in range(9) ] # pos
    hgcQuads =  [list(range(4*i,4*i+4)) for i in [10,11,12,13]]
    hgcQuads += [[4*14+1,4*14+2,4*14+3,4*15+3]] # mixed quad
    hgcQuads += [list(range(4*i,4*i+4)) for i in [16,17,18,19]]
    hgcalPosVU9PWriterConfig.hgcTimeSlices[t].hgcSectors += [ cms.PSet(hgcLinks = cms.vint32(-1,-1,-1,-1))      for i in range(3) ] # neg
    hgcalPosVU9PWriterConfig.hgcTimeSlices[t].hgcSectors += [ cms.PSet(hgcLinks = cms.vint32(*hgcQuads[3*i+t])) for i in range(3) ] # pos
    hgcalNegVU9PWriterConfig.hgcTimeSlices[t].hgcSectors += [ cms.PSet(hgcLinks = cms.vint32(*hgcQuads[3*i+t])) for i in range(3) ] # neg
    hgcalNegVU9PWriterConfig.hgcTimeSlices[t].hgcSectors += [ cms.PSet(hgcLinks = cms.vint32(-1,-1,-1,-1))      for i in range(3) ] # pos
    hgcalPosVU9PWriterConfig.gmtTimeSlices[t].gmtLink = cms.int32(4*15+((t+2)%3))
    hgcalNegVU9PWriterConfig.gmtTimeSlices[t].gmtLink = cms.int32(4*15+((t+2)%3))
hgcalPosVU9PWriterConfig.gttLink = 4*14+0
hgcalNegVU9PWriterConfig.gttLink = 4*14+0
hgcalPosVU9PWriterConfig.inputFileName = cms.string("l1HGCalPos-inputs-vu9p")
hgcalNegVU9PWriterConfig.inputFileName = cms.string("l1HGCalNeg-inputs-vu9p")
## Current configurations for VU13P
hgcalPosVU13PWriterConfig = _hgcalPosWriterConfig.clone()
hgcalNegVU13PWriterConfig = _hgcalNegWriterConfig.clone()
for t in range(3):
    hgcalPosVU13PWriterConfig.tfTimeSlices[t].tfSectors += [ cms.PSet(tfLink = cms.int32(-1))         for i in range(9) ] # neg
    hgcalPosVU13PWriterConfig.tfTimeSlices[t].tfSectors += [ cms.PSet(tfLink = cms.int32(3*i+t+4*0))  for i in range(5) ] # pos, left quads
    hgcalPosVU13PWriterConfig.tfTimeSlices[t].tfSectors += [ cms.PSet(tfLink = cms.int32(3*i+t+4*28)) for i in range(4) ] # pos, right quads
    hgcalNegVU13PWriterConfig.tfTimeSlices[t].tfSectors += [ cms.PSet(tfLink = cms.int32(3*i+t+4*0))  for i in range(5) ] # neg, left quads
    hgcalNegVU13PWriterConfig.tfTimeSlices[t].tfSectors += [ cms.PSet(tfLink = cms.int32(3*i+t+4*28)) for i in range(4) ] # neg, right quads
    hgcalNegVU13PWriterConfig.tfTimeSlices[t].tfSectors += [ cms.PSet(tfLink = cms.int32(-1))         for i in range(9) ] # pos
    hgcQuads =  [list(range(4*i,4*i+4)) for i in [12,13,14,  16,17,  19,20,21,22]]
    hgcalPosVU13PWriterConfig.hgcTimeSlices[t].hgcSectors += [ cms.PSet(hgcLinks = cms.vint32(-1,-1,-1,-1)) for i in range(3) ] # neg
    hgcalPosVU13PWriterConfig.hgcTimeSlices[t].hgcSectors += [ cms.PSet(hgcLinks = cms.vint32(*hgcQuads[3*i+t])) for i in range(3) ] # pos
    hgcalNegVU13PWriterConfig.hgcTimeSlices[t].hgcSectors += [ cms.PSet(hgcLinks = cms.vint32(*hgcQuads[3*i+t])) for i in range(3) ] # neg
    hgcalNegVU13PWriterConfig.hgcTimeSlices[t].hgcSectors += [ cms.PSet(hgcLinks = cms.vint32(-1,-1,-1,-1)) for i in range(3) ] # pos
    hgcalPosVU13PWriterConfig.gmtTimeSlices[t].gmtLink = cms.int32(4*18+t)
    hgcalNegVU13PWriterConfig.gmtTimeSlices[t].gmtLink = cms.int32(4*18+t)
hgcalPosVU13PWriterConfig.gttLink = 4*25+3
hgcalNegVU13PWriterConfig.gttLink = 4*25+3
hgcalPosVU13PWriterConfig.inputFileName = cms.string("l1HGCalPos-inputs-vu13p")
hgcalNegVU13PWriterConfig.inputFileName = cms.string("l1HGCalNeg-inputs-vu13p")

## Enable outputs and both boards
hgcalWriterConfigs = [
    hgcalPosOutputWriterConfig,
    hgcalNegOutputWriterConfig,
    hgcalPosVU9PWriterConfig,
    hgcalNegVU9PWriterConfig,
    hgcalPosVU13PWriterConfig,
    hgcalNegVU13PWriterConfig
]

#####################################################################################################################
## Forward HGCal configuration: only outputs for now, 18 regions, 12 candidates x region, 4 output fibers
_hgcalNoTKWriterConfig = cms.PSet(
    partition = cms.string("HGCalNoTk"),
    tmuxFactor = cms.uint32(6),
    outputRegions = cms.vuint32(*range(18)),
    nInputFramesPerBX = cms.uint32(9),
    nOutputFramesPerBX = cms.uint32(9),
    fileFormat = cms.string("EMPv2"),
    inputFileExtension = cms.string("txt.gz"),
    outputFileExtension = cms.string("txt.gz"),
    maxLinesPerInputFile = cms.uint32(1024),
    maxLinesPerOutputFile = cms.uint32(1024),
    eventsPerFile = cms.uint32(_eventsPerFile),
    hgcTimeSlices = cms.VPSet(*[cms.PSet(hgcSectors = cms.VPSet()) for i in range(3)]),
    gmtTimeSlices = cms.VPSet(*[cms.PSet(gmtLink = cms.int32(-1)) for  i in range(3)]),
    gmtNumberOfMuons = cms.uint32(12),
)
hgcalNoTKOutputWriterConfig = _hgcalNoTKWriterConfig.clone(
    outputLinksPuppi = cms.vuint32(*range(4)),
    outputFileName = cms.string("l1HGCalNoTK-outputs")
)
hgcalNoTKVU13PWriterConfig = _hgcalNoTKWriterConfig.clone()
for t in range(3):
    for isec in range(6):
        q0 = 3*isec + (6 if isec < 3 else 8)
        hgcalNoTKVU13PWriterConfig.hgcTimeSlices[t].hgcSectors += [ cms.PSet(hgcLinks = cms.vint32(*[4*q0+4*t+j for j in range(4)])) ]
hgcalNoTKVU13PWriterConfig.inputFileName = cms.string("l1HGCalNoTK-inputs-vu13p") # vu9p uses the same cabling for now

hgcalNoTKWriterConfigs = [
    hgcalNoTKOutputWriterConfig,
    hgcalNoTKVU13PWriterConfig
]

#####################################################################################################################
## TM18 configuration
_barrelSerenityTM18 = _barrelWriterOutputOnly.clone(
    fileFormat = cms.string("EMPv2"),
    tmuxFactor = cms.uint32(18),
    tfSectors = cms.VPSet(*[cms.PSet(tfLink = cms.int32(-1)) for i in range(18)]),
    gmtLink = cms.int32(4*18+0),
    gttLink = 4*28+3,
    eventsPerFile = 4,
)
_barrelSerenityTM18.gctEmSectors = cms.VPSet(*[ cms.PSet(gctEmLink = cms.int32(-1)) for s in range(6) ])
_barrelSerenityTM18.gctHadSectors = cms.VPSet(*[ cms.PSet(gctHadLink = cms.int32(-1)) for s in range(6) ])

barrelSerenityOutputTM18WriterConfig = _barrelSerenityTM18.clone(
    outputRegions = cms.vuint32(*range(54)),
    outputBoard = cms.int32(0),
    outputFileName = cms.string("l1BarrelSerenityTM18-outputs")
)
barrelSerenityVU13PTM18WriterConfig = _barrelSerenityTM18.clone(
    inputFileName = cms.string("l1BarrelSerenityTM18-inputs-vu13p"),
    gttLatency = cms.uint32(167), # shorter, to fit 6 events in 1024 lines
    maxLinesPerInputFile = cms.uint32(1024+167), # anything beyond 986 will be nulls
)
for ie in range(2):
    for iphi in range(9):
        isec = 9*ie+iphi
        barrelSerenityVU13PTM18WriterConfig.tfSectors[isec].tfLink = (isec if isec < 12 else (4*30+(isec-12)))

barrelSerenityTM18WriterConfigs = [
    barrelSerenityOutputTM18WriterConfig,
    barrelSerenityVU13PTM18WriterConfig
]

## Barrel (54) TM18 (APx format)
barrelWriterOutputOnlyPhiConfigsAPx = _barrelWriterOutputOnly.clone(
    fileFormat = cms.string("APx"),
    outputFileName = cms.string("l1BarrelApx-outputs")
)

barrelWriterDebugPFInConfigsAPx = _barrelWriterOutputOnly.clone(
    fileFormat = cms.string("APx"),
    nPFInTrack = cms.uint32(22),
    nPFInEmCalo = cms.uint32(12),
    nPFInHadCalo = cms.uint32(15),
    nPFInMuon = cms.uint32(2),
    debugFileName = cms.string("l1BarrelApx-pfin")
)

barrelWriterDebugPFOutConfigsAPx = _barrelWriterOutputOnly.clone(
    fileFormat = cms.string("APx"),
    nPFOutCharged = cms.uint32(22),
    nPFOutPhoton = cms.uint32(12),
    nPFOutNeutral = cms.uint32(15),
    nPFOutMuon = cms.uint32(2),
    debugFileName = cms.string("l1BarrelApx-pfout")
)

# For the tracker, the logical (firmware) fiber order sorts negative eta first, then positive,
# and in a given eta from most negative to most positive phi. That is reflected in this remapping
trackFiberOrder = (4, 5, 6, 7, 8, 0, 1, 2, 3, 13, 14, 15, 16, 17, 9, 10, 11, 12) # phys to logical
# This includes the tracker, GCT, muon and GTT sector mapping. There is only one GTT fiber, with a logical firmware link of 123.
barrelApxWriterConfig = _barrelWriterOutputOnly.clone(
    fileFormat = cms.string("APx"),
    gttLink = cms.int32(123),
    gmtLink = cms.int32(21),
    gctSectors = cms.VPSet(*[cms.PSet(gctLink = cms.int32(i)) for i in (18, 19, 20)]),
    tfSectors = cms.VPSet(*[cms.PSet(tfLink = cms.int32(i)) for i in trackFiberOrder]),
    inputFileName = cms.string("l1BarrelApx-inputs")
)

barrelOutputWriterConfigsAPx =  barrelWriterOutputOnlyPhiConfigsAPx
barrelInputWriterConfigsAPx =  barrelApxWriterConfig

_hgcalWriterTM18 = _hgcalWriterConfig.clone(
    tmuxFactor = cms.uint32(18),
    tfTimeSlices = None,
    tfSectors = cms.VPSet(*[cms.PSet(tfLink = cms.int32(-1)) for i in range(18)]),
    hgcTimeSlices = None,
    hgcSectors = cms.VPSet(*[cms.PSet(hgcLinks = cms.vint32()) for i in range(6)]),
    gmtTimeSlices = None,
    gmtLink = cms.int32(4*27+0),
    gttLink = 4*27+3,
    eventsPerFile = 4,
)
hgcalWriterOutputTM18WriterConfig = _hgcalWriterTM18.clone(
   outputFileName = cms.string("l1HGCalTM18-outputs"),
   outputRegions = cms.vuint32(*range(18)),
   outputLinksPuppi = cms.vuint32(*range(2)),
   outputLinkEgamma = cms.int32(2),
)
hgcalWriterVU9PTM18WriterConfig = _hgcalWriterTM18.clone(
   inputFileName = cms.string("l1HGCalTM18-inputs-vu9p"),
   gttLatency = cms.uint32(167), # shorter, to fit 6 events in 1024 lines
   maxLinesPerInputFile = cms.uint32(1024+167), # anything beyond 986 will be nulls
   gmtLink = 4*15+2,
   gttLink = 0,
)
hgcalWriterVU13PTM18WriterConfig = hgcalWriterVU9PTM18WriterConfig.clone(
   inputFileName = cms.string("l1HGCalTM18-inputs-vu13p"),
   gmtLink = 4*18+0,
   gttLink = 4*28+3,
)
for ie in range(2):
    for iphi in range(9):
        isec, ilink = 9*ie+iphi, 2*iphi+ie
        hgcalWriterVU9PTM18WriterConfig.tfSectors[isec].tfLink = (ilink+2 if ilink < 10 else (4*28+(ilink-10)))
        hgcalWriterVU13PTM18WriterConfig.tfSectors[isec].tfLink = (ilink if ilink < 12 else (4*30+(ilink-12)))
    for iphi in range(3):
        isec, ilink = 3*ie+iphi, 2*iphi+ie
        if ilink < 2:
            hgcalWriterVU9PTM18WriterConfig.hgcSectors[isec].hgcLinks += range(4*(12+ilink),4*(12+ilink)+4)
        else:
            hgcalWriterVU9PTM18WriterConfig.hgcSectors[isec].hgcLinks += range(4*(14+ilink),4*(14+ilink)+4)
        if ilink < 3:
            hgcalWriterVU13PTM18WriterConfig.hgcSectors[isec].hgcLinks += range(4*(12+ilink),4*(12+ilink)+4)
        elif ilink < 5:
            hgcalWriterVU13PTM18WriterConfig.hgcSectors[isec].hgcLinks += range(4*(13+ilink),4*(13+ilink)+4)
        else:
            hgcalWriterVU13PTM18WriterConfig.hgcSectors[isec].hgcLinks += range(4*(14+ilink),4*(14+ilink)+4)

hgcalTM18WriterConfigs = [
    hgcalWriterOutputTM18WriterConfig,
    hgcalWriterVU9PTM18WriterConfig,
    hgcalWriterVU13PTM18WriterConfig
]
hgcalNoTKOutputTM18WriterConfig = _hgcalWriterTM18.clone(
   outputFileName = cms.string("l1HGCalTM18-outputs-fwd"),
   outputRegions = cms.vuint32(*range(18)),
   outputBoard = cms.int32(-1),#0,1),
   outputLinksPuppi = cms.vuint32(*range(2)),
   outputLinkEgamma = cms.int32(-1),
)

#####################################################################################################################
## HF configuration (to be better defined later)
#####################################################################################################################
## HF configuration not realistic, 3 links per endcap, write out the layer 1 outputs for now
_hfWriterOutputOnly = cms.PSet(
    partition = cms.string("HF"),
    tmuxFactor = cms.uint32(6),
    outputLinksPuppi = cms.vuint32(*range(3)),
    nOutputFramesPerBX = cms.uint32(9),
    fileFormat = cms.string("EMPv2"),
    outputFileExtension = cms.string("txt.gz"),
    maxLinesPerOutputFile = cms.uint32(1024),
    eventsPerFile = cms.uint32(_eventsPerFile),
)
hfWriterConfigs = [
    _hfWriterOutputOnly.clone(
        outputRegions = cms.vuint32(*[9*ie+i for i in range(9)]),
        outputFileName = cms.string("l1HF%s-outputs" % ("Pos" if ie else "Neg")),
    ) for ie in range(2)
]


