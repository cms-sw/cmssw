import FWCore.ParameterSet.Config as cms

eventsPerFile_ = 12
gttLatency_ = 156+120
gttNumberOfPVs_ = 10

#####################################################################################################################
## Barrel configurations: 54 regions, 6 puppi output links, only write out the layer 1 outputs for now
barrelWriterOutputOnly_ = cms.PSet(
    partition = cms.string("Barrel"),
    outputLinksPuppi = cms.vuint32(*range(6)),
    outputLinkEgamma = cms.int32(6),
    nEgammaObjectsOut = cms.uint32(16),
    nOutputFramesPerBX = cms.uint32(9),
    fileFormat = cms.string("EMP"),
    maxLinesPerOutputFile = cms.uint32(1024),
    eventsPerFile = cms.uint32(eventsPerFile_),
)
## Barrel (54) split in 3 eta slices
barrelWriterOutputOnlyEtaConfigs = [
    barrelWriterOutputOnly_.clone(
        outputRegions = cms.vuint32(*[18*ie+i for i in range(18)]),
        outputFileName = cms.string("l1BarrelEta%d-outputs-ideal" % (ie+1)),
        outputBoard      = cms.int32(-1), ## can't output e/gamma in eta split regions
        outputLinkEgamma = cms.int32(-1), ## since the boards are defined in phi regions
    ) for ie in range(3)
]
## Barrel (54) split in 3 phi slices
barrelWriterOutputOnlyPhiConfigs = [
    barrelWriterOutputOnly_.clone(
        outputRegions = cms.vuint32(*[3*ip+9*ie+i for ie in range(6) for i in range(3) ]),
        outputBoard = cms.int32(ip),
        outputFileName = cms.string("l1BarrelPhi%d-outputs-ideal" % (ip+1))
    ) for ip in range(3)
]
## Barrel9 (27) split in phi eta slices
barrel9WriterOutputOnlyPhiConfigs = [
    barrelWriterOutputOnly_.clone(
        outputRegions = cms.vuint32(*[3*ip+9*ie+i for ie in range(3) for i in range(3) ]),
        outputBoard = cms.int32(ip),
        outputFileName = cms.string("l1Barrel9Phi%d-outputs-ideal" % (ip+1))
    ) for ip in range(3)
]

barrelWriterConfigs =  barrelWriterOutputOnlyPhiConfigs # + barrelWriterOutputOnlyEtaConfigs  
barrel9WriterConfigs = [] #barrel9WriterOutputOnlyPhiConfigs 


#####################################################################################################################
## HGcal configuration: write out both inputs and outputs
hgcalWriterConfig_ = cms.PSet(
    partition = cms.string("HGCal"),
    outputRegions = cms.vuint32(*[i+9 for i in range(9)]),
    outputBoard = cms.int32(1),
    nEgammaObjectsOut = cms.uint32(16),
    nInputFramesPerBX = cms.uint32(9),
    nOutputFramesPerBX = cms.uint32(9),
    fileFormat = cms.string("EMP"),
    maxLinesPerInputFile = cms.uint32(1024),
    maxLinesPerOutputFile = cms.uint32(1024),
    eventsPerFile = cms.uint32(eventsPerFile_),
    tfTimeSlices = cms.VPSet(*[cms.PSet(tfSectors = cms.VPSet()) for i in range(3)]),
    hgcTimeSlices = cms.VPSet(*[cms.PSet(hgcSectors = cms.VPSet()) for i in range(3)]),
    gmtTimeSlices = cms.VPSet(cms.PSet(),cms.PSet(),cms.PSet()),
    gmtNumberOfMuons = cms.uint32(12),
    gttLink = cms.int32(-1),
    gttLatency = cms.uint32(gttLatency_),
    gttNumberOfPVs = cms.uint32(gttNumberOfPVs_) 
)
## Ideal configuration: 27 input links from tf, 36 from hgc, 3 from gmt, 1 from gtt, in this order; output 3 puppi + 1 e/gamma
hgcalPosIdealWriterConfig = hgcalWriterConfig_.clone()
for t in range(3):
    hgcalPosIdealWriterConfig.tfTimeSlices[t].tfSectors += [ cms.PSet(tfLink = cms.int32(-1))    for i in range(9) ] # neg
    hgcalPosIdealWriterConfig.tfTimeSlices[t].tfSectors += [ cms.PSet(tfLink = cms.int32(3*i+t)) for i in range(9) ] # pos
    hgcalPosIdealWriterConfig.hgcTimeSlices[t].hgcSectors += [ cms.PSet(hgcLinks = cms.vint32(-1,-1,-1,-1))                        for i in range(3) ] # neg
    hgcalPosIdealWriterConfig.hgcTimeSlices[t].hgcSectors += [ cms.PSet(hgcLinks = cms.vint32(*[27+12*i+4*t+j for j in range(4)])) for i in range(3) ] # pos
    hgcalPosIdealWriterConfig.gmtTimeSlices[t].gmtLink = cms.int32(27+36+t)
hgcalPosIdealWriterConfig.gttLink = 27+36+3
hgcalPosIdealWriterConfig.outputLinksPuppi = cms.vuint32(0,1,2)
hgcalPosIdealWriterConfig.outputLinkEgamma = cms.int32(5)
hgcalPosIdealWriterConfig.inputFileName = cms.string("l1HGCalPos-inputs-ideal") 
hgcalPosIdealWriterConfig.outputFileName = cms.string("l1HGCalPos-outputs-ideal")
hgcalNegIdealWriterConfig = hgcalPosIdealWriterConfig.clone(
    inputFileName = "",
    outputFileName = "l1HGCalNeg-outputs-ideal",
    outputRegions = [i for i in range(9)],
    outputBoard = 0,
)
## Current configuration for VU9P at B904 for layer1 - layer2 tests with puppi and e/gamma outputs on links 56-59
hgcalPosVU9PB904egWriterConfig = hgcalWriterConfig_.clone()
for t in range(3):
    hgcalPosVU9PB904egWriterConfig.tfTimeSlices[t].tfSectors += [ cms.PSet(tfLink = cms.int32(-1))    for i in range(9) ] # neg
    hgcalPosVU9PB904egWriterConfig.tfTimeSlices[t].tfSectors += [ cms.PSet(tfLink = cms.int32(3*i+t+4*2)) for i in range(4) ] # pos, left quads
    hgcalPosVU9PB904egWriterConfig.tfTimeSlices[t].tfSectors += [ cms.PSet(tfLink = cms.int32(3*i+t+4*25)) for i in range(5) ] # pos, right quads
    hgcalPosVU9PB904egWriterConfig.hgcTimeSlices[t].hgcSectors += [ cms.PSet(hgcLinks = cms.vint32(-1,-1,-1,-1))                        for i in range(3) ] # neg
    hgcalPosVU9PB904egWriterConfig.hgcTimeSlices[t].hgcSectors += [ cms.PSet(hgcLinks = cms.vint32(*[4*11+12*i+4*t+j for j in range(4)])) for i in range(3) ] # pos
    hgcalPosVU9PB904egWriterConfig.gmtTimeSlices[t].gmtLink = cms.int32(4+t)
hgcalPosVU9PB904egWriterConfig.gttLink = 4+3
hgcalPosVU9PB904egWriterConfig.outputLinksPuppi = cms.vuint32(56,57,58)
hgcalPosVU9PB904egWriterConfig.outputLinkEgamma = cms.int32(59)
hgcalPosVU9PB904egWriterConfig.inputFileName = cms.string("l1HGCalPos-inputs-vu9p_B904eg") 
hgcalPosVU9PB904egWriterConfig.outputFileName = cms.string("l1HGCalPos-outputs-vu9p_B904eg")
## Current configuration for VU13P 
hgcalPosVU13PWriterConfig = hgcalWriterConfig_.clone()
for t in range(3):
    hgcalPosVU13PWriterConfig.tfTimeSlices[t].tfSectors += [ cms.PSet(tfLink = cms.int32(-1))    for i in range(9) ] # neg
    hgcalPosVU13PWriterConfig.tfTimeSlices[t].tfSectors += [ cms.PSet(tfLink = cms.int32(3*i+t+4*0)) for i in range(5) ] # pos, left quads
    hgcalPosVU13PWriterConfig.tfTimeSlices[t].tfSectors += [ cms.PSet(tfLink = cms.int32(3*i+t+4*28)) for i in range(4) ] # pos, right quads
    hgcalPosVU13PWriterConfig.hgcTimeSlices[t].hgcSectors += [ cms.PSet(hgcLinks = cms.vint32(-1,-1,-1,-1)) for i in range(3) ] # neg
    for isec,q0 in (0,12),(1,17),(2,20):
        hgcalPosVU13PWriterConfig.hgcTimeSlices[t].hgcSectors += [ cms.PSet(hgcLinks = cms.vint32(*[4*q0+4*t+j for j in range(4)])) ] # pos
    hgcalPosVU13PWriterConfig.gmtTimeSlices[t].gmtLink = cms.int32(4*27+t)
hgcalPosVU13PWriterConfig.gttLink = 4*27+3
hgcalPosVU13PWriterConfig.outputLinksPuppi = cms.vuint32(0,1,2)
hgcalPosVU13PWriterConfig.outputLinkEgamma = cms.int32(3)
hgcalPosVU13PWriterConfig.inputFileName = cms.string("l1HGCalPos-inputs-vu13p") 
hgcalPosVU13PWriterConfig.outputFileName = cms.string("l1HGCalPos-outputs-vu13p")
hgcalNegVU13PWriterConfig = hgcalWriterConfig_.clone()
for t in range(3):
    hgcalNegVU13PWriterConfig.tfTimeSlices[t].tfSectors += [ cms.PSet(tfLink = cms.int32(3*i+t+4*0)) for i in range(5) ] # neg, left quads
    hgcalNegVU13PWriterConfig.tfTimeSlices[t].tfSectors += [ cms.PSet(tfLink = cms.int32(3*i+t+4*28)) for i in range(4) ] # neg, right quads
    hgcalNegVU13PWriterConfig.tfTimeSlices[t].tfSectors += [ cms.PSet(tfLink = cms.int32(-1))    for i in range(9) ] # pos
    for isec,q0 in (0,12),(1,17),(2,20):
        hgcalNegVU13PWriterConfig.hgcTimeSlices[t].hgcSectors += [ cms.PSet(hgcLinks = cms.vint32(*[4*q0+4*t+j for j in range(4)])) ] # neg
    hgcalNegVU13PWriterConfig.hgcTimeSlices[t].hgcSectors += [ cms.PSet(hgcLinks = cms.vint32(-1,-1,-1,-1)) for i in range(3) ] # pos
    hgcalNegVU13PWriterConfig.gmtTimeSlices[t].gmtLink = cms.int32(4*27+t)
hgcalNegVU13PWriterConfig.gttLink = 4*27+3
hgcalNegVU13PWriterConfig.outputLinksPuppi = cms.vuint32(0,1,2)
hgcalNegVU13PWriterConfig.outputLinkEgamma = cms.int32(3)
hgcalNegVU13PWriterConfig.inputFileName = cms.string("l1HGCalNeg-inputs-vu13p") 
hgcalNegVU13PWriterConfig.outputFileName = cms.string("l1HGCalNeg-outputs-vu13p")

## Enable both

hgcalWriterConfigs = [ 
    hgcalPosIdealWriterConfig, 
    hgcalNegIdealWriterConfig, 
    hgcalPosVU9PB904egWriterConfig,
    hgcalPosVU13PWriterConfig,
    hgcalNegVU13PWriterConfig
]

#####################################################################################################################
## Forward HGCal configuration: only outputs for now, 18 regions, 12 candidates x region, 4 output fibers
hgcalNoTKWriterOutputOnlyConfig = cms.PSet(
    partition = cms.string("HGCalNoTk"),
    outputRegions = cms.vuint32(*range(18)),
    nOutputFramesPerBX = cms.uint32(9),
    fileFormat = cms.string("EMP"),
    maxLinesPerOutputFile = cms.uint32(1024),
    eventsPerFile = cms.uint32(eventsPerFile_),
    outputLinksPuppi = cms.vuint32(0,1,2,4),
    outputFileName = cms.string("l1HGCalNoTk-outputs-ideal")
)

hgcalNoTKWriterConfigs = [ 
    hgcalNoTKWriterOutputOnlyConfig
]

#####################################################################################################################
## HF configuration: not enabled for the moment
#####################################################################################################################
## HF configuration not realistic, 3 links per endcap, write out the layer 1 outputs for now
hfWriterOutputOnly_ = cms.PSet(
    partition = cms.string("HF"),
    outputLinksPuppi = cms.vuint32(*range(3)),
    nOutputFramesPerBX = cms.uint32(9),
    fileFormat = cms.string("EMP"),
    maxLinesPerOutputFile = cms.uint32(1024),
    eventsPerFile = cms.uint32(eventsPerFile_),
)
hfWriterConfigs = [
    hfWriterOutputOnly_.clone(
        outputRegions = cms.vuint32(*[9*ie+i for i in range(9)]),
        outputFileName = cms.string("l1HF%s-outputs-ideal" % ("Pos" if ie else "Neg")),
    ) for ie in range(2)
]


