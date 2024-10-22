import FWCore.ParameterSet.Config as cms

_eventsPerFile = 12
_gttLatency = 156+120
_gttNumberOfPVs = 10

#####################################################################################################################
## Barrel configurations: 54 regions, 6 puppi output links, only write out the layer 1 outputs for now
_barrelWriterOutputOnly = cms.PSet(
    partition = cms.string("Barrel"),
    tmuxFactor = cms.uint32(6),
    outputLinksPuppi = cms.vuint32(*range(6)),
    outputLinkEgamma = cms.int32(6),
    nEgammaObjectsOut = cms.uint32(16),
    nInputFramesPerBX = cms.uint32(9),
    nOutputFramesPerBX = cms.uint32(9),
    fileFormat = cms.string("EMPv2"),
    inputFileExtension = cms.string("txt.gz"),
    outputFileExtension = cms.string("txt.gz"),
    maxLinesPerInputFile = cms.uint32(1024),
    maxLinesPerOutputFile = cms.uint32(1024),
    eventsPerFile = cms.uint32(_eventsPerFile),
    tfTimeSlices = cms.VPSet(),
    gctNLinksEcal = cms.uint32(1),
    gctNLinksHad = cms.uint32(2),
    gctSectors = cms.VPSet(),
    gmtTimeSlices = cms.VPSet(),
    gmtNumberOfMuons = cms.uint32(12),
    gttLink = cms.int32(-1),
    gttLatency = cms.uint32(156+10),
    gttNumberOfPVs = cms.uint32(_gttNumberOfPVs),
)
## Barrel (54) split in 3 phi slices
barrelWriterOutputOnlyPhiConfigs = [
    _barrelWriterOutputOnly.clone(
        outputRegions = cms.vuint32(*[3*ip+9*ie+i for ie in range(6) for i in range(3) ]),
        outputBoard = cms.int32(ip),
        outputFileName = cms.string("l1BarrelPhi%d-outputs" % (ip+1))
    ) for ip in range(3)
]

barrelSerenityPhi1Config = barrelWriterOutputOnlyPhiConfigs[0].clone(
    tfTimeSlices = cms.VPSet(*[cms.PSet(tfSectors = cms.VPSet(*[ cms.PSet(tfLink = cms.int32(-1)) for s in range(18) ])) for t in range(3)]),
    gctSectors = cms.VPSet(*[cms.PSet(
        gctLinksHad = cms.vint32(-1,-1),
        gctLinksEcal = cms.vint32(-1),
        ) for s in range(3)]),
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
for t in range(3):
   for ie in range(2):
    for i,s in enumerate([8, 0, 1, 2, 3]):
        loglink = 3*(i+5*ie)+t
        physlink = loglink+4*1 if loglink < 15 else (loglink-15)+4*25
        barrelSerenityVU9PPhi1Config.tfTimeSlices[t].tfSectors[s+9*ie].tfLink = physlink
        physlink = loglink+4*0 if loglink < 15 else (loglink-15)+4*28
        barrelSerenityVU13PPhi1Config.tfTimeSlices[t].tfSectors[s+9*ie].tfLink = physlink
for i,s in enumerate([0,1]):
   barrelSerenityVU9PPhi1Config.gctSectors[s].gctLinksHad  = [3*i+4*18, 3*i+4*18+1]
   barrelSerenityVU9PPhi1Config.gctSectors[s].gctLinksEcal = [3*i+4*18+2]
   gctLinks = list(range(4*17,4*17+4)) + list(range(4*19,4*19+2))
   barrelSerenityVU13PPhi1Config.gctSectors[s].gctLinksHad  = [gctLinks[3*i], gctLinks[3*i+1]]
   barrelSerenityVU13PPhi1Config.gctSectors[s].gctLinksEcal = [gctLinks[3*i+2]]

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
    #gttLink = cms.int32(-1),
    #gttLatency = cms.uint32(_gttLatency),
    #gttNumberOfPVs = cms.uint32(_gttNumberOfPVs),
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
    tmuxFactor = cms.uint32(18),
    tfTimeSlices = None,
    tfSectors = cms.VPSet(*[cms.PSet(tfLink = cms.int32(-1)) for i in range(18)]),
    gmtTimeSlices = None,
    gmtLink = cms.int32(4*18+0),
    gttLink = 4*28+3,
    eventsPerFile = 4,
)
barrelSerenityOutputTM18WriterConfig = _barrelSerenityTM18.clone(
    outputRegions = cms.vuint32(*range(54)),
    outputBoard = cms.int32(0),
    outputFileName = cms.string("l1BarrelSerenityTM18-outputs")
)
barrelSerenityVU13PTM18WriterConfig = _barrelSerenityTM18.clone(
    inputFileName = cms.string("l1BarrelSerenityTM18-inputs-vu13p"),
    gttLatency = cms.uint32(167), # shorter, to fit 6 events in 1024 lines
    maxLinesPerInputFile = cms.uint32(1024+167), # anything beyond 986 will be nulls
    gctNLinksEcal = 1,
    gctNLinksHad = 1,
    gctSectors = cms.VPSet(*[cms.PSet(
        gctLinksHad = cms.vint32(4*18+1+s),
        gctLinksEcal = cms.vint32(4*18+1+s),
    ) for s in range(3)]),
)
for ie in range(2):
    for iphi in range(9):
        isec = 9*ie+iphi 
        barrelSerenityVU13PTM18WriterConfig.tfSectors[isec].tfLink = (isec if isec < 12 else (4*30+(isec-12)))

barrelSerenityTM18WriterConfigs = [
    barrelSerenityOutputTM18WriterConfig,
    barrelSerenityVU13PTM18WriterConfig
]

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


