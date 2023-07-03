import FWCore.ParameterSet.Config as cms

_eventsPerFile = 12
_gttLatency = 156+120
_gttNumberOfPVs = 10

#####################################################################################################################
## Barrel configurations: 54 regions, 6 puppi output links, only write out the layer 1 outputs for now
_barrelWriterOutputOnly = cms.PSet(
    partition = cms.string("Barrel"),
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
    gmtNumberOfMuons = cms.uint32(12),
    gttLatency = cms.uint32(156+10),
    gttNumberOfPVs = cms.uint32(_gttNumberOfPVs),
    inputFileName = cms.string("l1BarrelPhi1Serenity-inputs-vu9p"),
    outputFileName = cms.string("l1BarrelPhi1Serenity-outputs")
)
barrelSerenityVU9PPhi1Config = barrelSerenityPhi1Config.clone(
    gttLink = cms.int32(4*28+3),
    inputFileName = cms.string("l1BarrelPhi1Serenity-inputs-vu9p"),
    outputFileName = cms.string("l1BarrelPhi1Serenity-outputs")
)
barrelSerenityVU13PPhi1Config = barrelSerenityPhi1Config.clone(
    gttLink = cms.int32(4*31+3),
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
   barrelSerenityVU13PPhi1Config.gctSectors[s].gctLinksHad  = [3*i+4*18, 3*i+4*18+1]
   barrelSerenityVU13PPhi1Config.gctSectors[s].gctLinksEcal = [3*i+4*18+2]

barrelWriterConfigs =  barrelWriterOutputOnlyPhiConfigs


#####################################################################################################################
## HGcal configuration: write out both inputs and outputs
_hgcalWriterConfig = cms.PSet(
    partition = cms.string("HGCal"),
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
    gmtTimeSlices = cms.VPSet(cms.PSet(),cms.PSet(),cms.PSet()),
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
    hgcalPosVU9PWriterConfig.tfTimeSlices[t].tfSectors += [ cms.PSet(tfLink = cms.int32(3*i+t+4*2))  for i in range(4) ] # pos, left quads
    hgcalPosVU9PWriterConfig.tfTimeSlices[t].tfSectors += [ cms.PSet(tfLink = cms.int32(3*i+t+4*25)) for i in range(5) ] # pos, right quads
    hgcalNegVU9PWriterConfig.tfTimeSlices[t].tfSectors += [ cms.PSet(tfLink = cms.int32(3*i+t+4*2))  for i in range(4) ] # neg, left quads
    hgcalNegVU9PWriterConfig.tfTimeSlices[t].tfSectors += [ cms.PSet(tfLink = cms.int32(3*i+t+4*25)) for i in range(5) ] # neg, right quads
    hgcalNegVU9PWriterConfig.tfTimeSlices[t].tfSectors += [ cms.PSet(tfLink = cms.int32(-1))         for i in range(9) ] # pos
    hgcalPosVU9PWriterConfig.hgcTimeSlices[t].hgcSectors += [ cms.PSet(hgcLinks = cms.vint32(-1,-1,-1,-1))                          for i in range(3) ] # neg
    hgcalPosVU9PWriterConfig.hgcTimeSlices[t].hgcSectors += [ cms.PSet(hgcLinks = cms.vint32(*[4*11+12*i+4*t+j for j in range(4)])) for i in range(3) ] # pos
    hgcalNegVU9PWriterConfig.hgcTimeSlices[t].hgcSectors += [ cms.PSet(hgcLinks = cms.vint32(*[4*11+12*i+4*t+j for j in range(4)])) for i in range(3) ] # neg
    hgcalNegVU9PWriterConfig.hgcTimeSlices[t].hgcSectors += [ cms.PSet(hgcLinks = cms.vint32(-1,-1,-1,-1))                          for i in range(3) ] # pos
    hgcalPosVU9PWriterConfig.gmtTimeSlices[t].gmtLink = cms.int32(4+t)
    hgcalNegVU9PWriterConfig.gmtTimeSlices[t].gmtLink = cms.int32(4+t)
hgcalPosVU9PWriterConfig.gttLink = 4+3
hgcalNegVU9PWriterConfig.gttLink = 4+3
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
    hgcalPosVU13PWriterConfig.hgcTimeSlices[t].hgcSectors += [ cms.PSet(hgcLinks = cms.vint32(-1,-1,-1,-1)) for i in range(3) ] # neg
    for isec,q0 in (0,12),(1,17),(2,20):
        hgcalPosVU13PWriterConfig.hgcTimeSlices[t].hgcSectors += [ cms.PSet(hgcLinks = cms.vint32(*[4*q0+4*t+j for j in range(4)])) ] # pos
        hgcalNegVU13PWriterConfig.hgcTimeSlices[t].hgcSectors += [ cms.PSet(hgcLinks = cms.vint32(*[4*q0+4*t+j for j in range(4)])) ] # neg
    hgcalNegVU13PWriterConfig.hgcTimeSlices[t].hgcSectors += [ cms.PSet(hgcLinks = cms.vint32(-1,-1,-1,-1)) for i in range(3) ] # pos
    hgcalPosVU13PWriterConfig.gmtTimeSlices[t].gmtLink = cms.int32(4*27+t)
    hgcalNegVU13PWriterConfig.gmtTimeSlices[t].gmtLink = cms.int32(4*27+t)
hgcalPosVU13PWriterConfig.gttLink = 4*27+3
hgcalNegVU13PWriterConfig.gttLink = 4*27+3
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
    outputRegions = cms.vuint32(*range(18)),
    nInputFramesPerBX = cms.uint32(9),
    nOutputFramesPerBX = cms.uint32(9),
    fileFormat = cms.string("EMPv2"),
    inputFileExtension = cms.string("txt.gz"),
    outputFileExtension = cms.string("txt.gz"),
    maxLinesPerInputFile = cms.uint32(1024),
    maxLinesPerOutputFile = cms.uint32(1024),
    eventsPerFile = cms.uint32(_eventsPerFile),
)
hgcalNoTKOutputWriterConfig = _hgcalNoTKWriterConfig.clone(
    outputLinksPuppi = cms.vuint32(*range(4)),
    outputFileName = cms.string("l1HGCalNoTK-outputs")
)
hgcalNoTKVU13PWriterConfig = _hgcalNoTKWriterConfig.clone(
    hgcTimeSlices = cms.VPSet(*[cms.PSet(hgcSectors = cms.VPSet()) for i in range(3)]),
    gmtTimeSlices = cms.VPSet(*[cms.PSet(gmtLink = cms.int32(-1)) for  i in range(3)]),
    gmtNumberOfMuons = cms.uint32(12),
    gttLink = cms.int32(-1),
    gttLatency = cms.uint32(_gttLatency),
    gttNumberOfPVs = cms.uint32(_gttNumberOfPVs),
)
for t in range(3):
    for isec in range(6):
        q0 = 3*isec + 6
        hgcalNoTKVU13PWriterConfig.hgcTimeSlices[t].hgcSectors += [ cms.PSet(hgcLinks = cms.vint32(*[4*q0+4*t+j for j in range(4)])) ] # pos
hgcalNoTKVU13PWriterConfig.inputFileName = cms.string("l1HGCalNoTK-inputs-vu13p") # vu9p uses the same cabling for now

hgcalNoTKWriterConfigs = [ 
    hgcalNoTKOutputWriterConfig,
    hgcalNoTKVU13PWriterConfig
]

#####################################################################################################################
## HF configuration (to be better defined later)
#####################################################################################################################
## HF configuration not realistic, 3 links per endcap, write out the layer 1 outputs for now
_hfWriterOutputOnly = cms.PSet(
    partition = cms.string("HF"),
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


