from __future__ import print_function
import FWCore.ParameterSet.Config as cms
import FWCore.ParameterSet.VarParsing as VarParsing

process = cms.Process("Test")
options = VarParsing.VarParsing()
options.register('isPhase2',
                 False, # default value
                 VarParsing.VarParsing.multiplicity.singleton, # singleton or list
                 VarParsing.VarParsing.varType.bool, # string, int, or float
                 "change for phase2")
options.register('isBPix',
                 False, # default value
                 VarParsing.VarParsing.multiplicity.singleton, # singleton or list
                 VarParsing.VarParsing.varType.bool, # string, int, or float
                 "switch for BPix")
options.register('isFPix',
                 False, # default value
                 VarParsing.VarParsing.multiplicity.singleton, # singleton or list
                 VarParsing.VarParsing.varType.bool, # string, int, or float
                 "switch for FPix")
options.register('isByModule',
                 False, # default value
                 VarParsing.VarParsing.multiplicity.singleton, # singleton or list
                 VarParsing.VarParsing.varType.bool, # string, int, or float
                 "switch for by Module")
options.parseArguments()

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)
process.source = cms.Source("EmptySource",
                            firstRun = cms.untracked.uint32(1)
                            )

process.TFileService = cms.Service("TFileService",
                                   fileName = cms.string("siPixelLorentzAngle_histo.root")
                                   )

process.MessageLogger = cms.Service("MessageLogger",
    cerr = cms.untracked.PSet(
        enable = cms.untracked.bool(False)
    ),
    cout = cms.untracked.PSet(
        enable = cms.untracked.bool(True),
        threshold = cms.untracked.string('WARNING')
    )
)

process.Timing = cms.Service("Timing")


## import the ESSource
from CalibTracker.SiPixelESProducers.siPixelFakeLorentzAngleESSource_cfi import siPixelFakeLorentzAngleESSource

BPIX_LAYER1=0.0595
BPIX_LAYER_2_MODULE_1_4 = 0.0765
BPIX_LAYER_2_MODULE_5_8 = 0.0805
BPIX_LAYER_3_MODULE_1_4 = 0.0864
BPIX_LAYER_3_MODULE_5_8 = 0.0929
BPIX_LAYER_4_MODULE_1_4 = 0.0961
BPIX_LAYER_4_MODULE_5_8 = 0.1036

FPix_300V_RNG1_PNL1 = 0.0805
FPix_300V_RNG1_PNL2 = 0.0788
FPix_300V_RNG2_PNL1 = 0.0756
FPix_300V_RNG2_PNL2 = 0.0736

process.SiPixelFakeLorentzAngleESSource = siPixelFakeLorentzAngleESSource.clone()

if(options.isPhase2):
    print(" ========> Testing Phase-2")
    process.SiPixelFakeLorentzAngleESSource.appendToDataLabel = cms.string("forPhase2")
    process.SiPixelFakeLorentzAngleESSource.bPixLorentzAnglePerTesla = cms.untracked.double(0.106)
    process.SiPixelFakeLorentzAngleESSource.fPixLorentzAnglePerTesla = cms.untracked.double(0.106)
    process.SiPixelFakeLorentzAngleESSource.file = 'SLHCUpgradeSimulations/Geometry/data/PhaseII/Tilted/PixelSkimmedGeometryT14.txt'
    process.SiPixelFakeLorentzAngleESSource.topologyInput = 'Geometry/TrackerCommonData/data/PhaseII/trackerParameters.xml'
else:
    if(options.isBPix):
        print("  ========> Testing BPix parameters")
        process.SiPixelFakeLorentzAngleESSource.BPixParameters = cms.VPSet(
            cms.PSet(layer = cms.int32(1), angle = cms.double(BPIX_LAYER1)),
            cms.PSet(layer = cms.int32(2), module = cms.int32(1), angle = cms.double(BPIX_LAYER_2_MODULE_1_4)),
            cms.PSet(layer = cms.int32(2), module = cms.int32(2), angle = cms.double(BPIX_LAYER_2_MODULE_1_4)),
            cms.PSet(layer = cms.int32(2), module = cms.int32(3), angle = cms.double(BPIX_LAYER_2_MODULE_1_4)),
            cms.PSet(layer = cms.int32(2), module = cms.int32(4), angle = cms.double(BPIX_LAYER_2_MODULE_1_4)),
            cms.PSet(layer = cms.int32(2), module = cms.int32(5), angle = cms.double(BPIX_LAYER_2_MODULE_5_8)),
            cms.PSet(layer = cms.int32(2), module = cms.int32(6), angle = cms.double(BPIX_LAYER_2_MODULE_5_8)),
            cms.PSet(layer = cms.int32(2), module = cms.int32(7), angle = cms.double(BPIX_LAYER_2_MODULE_5_8)),
            cms.PSet(layer = cms.int32(2), module = cms.int32(8), angle = cms.double(BPIX_LAYER_2_MODULE_5_8)),
            cms.PSet(layer = cms.int32(3), module = cms.int32(1), angle = cms.double(BPIX_LAYER_3_MODULE_1_4)),
            cms.PSet(layer = cms.int32(3), module = cms.int32(2), angle = cms.double(BPIX_LAYER_3_MODULE_1_4)),
            cms.PSet(layer = cms.int32(3), module = cms.int32(3), angle = cms.double(BPIX_LAYER_3_MODULE_1_4)),
            cms.PSet(layer = cms.int32(3), module = cms.int32(4), angle = cms.double(BPIX_LAYER_3_MODULE_1_4)),
            cms.PSet(layer = cms.int32(3), module = cms.int32(5), angle = cms.double(BPIX_LAYER_3_MODULE_5_8)),
            cms.PSet(layer = cms.int32(3), module = cms.int32(6), angle = cms.double(BPIX_LAYER_3_MODULE_5_8)),
            cms.PSet(layer = cms.int32(3), module = cms.int32(7), angle = cms.double(BPIX_LAYER_3_MODULE_5_8)),
            cms.PSet(layer = cms.int32(3), module = cms.int32(8), angle = cms.double(BPIX_LAYER_3_MODULE_5_8)),
            cms.PSet(layer = cms.int32(4), module = cms.int32(1), angle = cms.double(BPIX_LAYER_4_MODULE_1_4)),
            cms.PSet(layer = cms.int32(4), module = cms.int32(2), angle = cms.double(BPIX_LAYER_4_MODULE_1_4)),
            cms.PSet(layer = cms.int32(4), module = cms.int32(3), angle = cms.double(BPIX_LAYER_4_MODULE_1_4)),
            cms.PSet(layer = cms.int32(4), module = cms.int32(4), angle = cms.double(BPIX_LAYER_4_MODULE_1_4)),
            cms.PSet(layer = cms.int32(4), module = cms.int32(5), angle = cms.double(BPIX_LAYER_4_MODULE_5_8)),
            cms.PSet(layer = cms.int32(4), module = cms.int32(6), angle = cms.double(BPIX_LAYER_4_MODULE_5_8)),
            cms.PSet(layer = cms.int32(4), module = cms.int32(7), angle = cms.double(BPIX_LAYER_4_MODULE_5_8)),
            cms.PSet(layer = cms.int32(4), module = cms.int32(8), angle = cms.double(BPIX_LAYER_4_MODULE_5_8)),        
        )
    elif(options.isFPix):
        print("  ========> Testing FPix parameters")
        process.SiPixelFakeLorentzAngleESSource.FPixParameters = cms.VPSet(
            cms.PSet(
                ring = cms.int32(1),
                panel = cms.int32(1),
                angle = cms.double(0.0805)
            ),
            cms.PSet(
                ring = cms.int32(1),
                panel = cms.int32(2),
                angle = cms.double(0.0788)
            ),
            cms.PSet(
                ring = cms.int32(2),
                panel = cms.int32(1),
                angle = cms.double(0.0756)
            ),
            cms.PSet(
                ring = cms.int32(2),
                panel = cms.int32(2),
                angle = cms.double(0.0736)
            )
        )       
    elif(options.isByModule):
        print(" ========> Testing byModule")
        process.SiPixelFakeLorentzAngleESSource.ModuleParameters = cms.VPSet(
            cms.PSet( rawid=cms.uint32(352588804), angle=cms.double(FPix_300V_RNG1_PNL1) ),
            cms.PSet( rawid=cms.uint32(352592900), angle=cms.double(FPix_300V_RNG1_PNL1) ),
            cms.PSet( rawid=cms.uint32(352596996), angle=cms.double(FPix_300V_RNG1_PNL1) ),
            cms.PSet( rawid=cms.uint32(352601092), angle=cms.double(FPix_300V_RNG1_PNL1) ),
            cms.PSet( rawid=cms.uint32(352605188), angle=cms.double(FPix_300V_RNG1_PNL1) ),
            cms.PSet( rawid=cms.uint32(352609284), angle=cms.double(FPix_300V_RNG1_PNL1) ),
            cms.PSet( rawid=cms.uint32(352658436), angle=cms.double(FPix_300V_RNG1_PNL1) ),
            cms.PSet( rawid=cms.uint32(352662532), angle=cms.double(FPix_300V_RNG1_PNL1) ),
            cms.PSet( rawid=cms.uint32(352666628), angle=cms.double(FPix_300V_RNG1_PNL1) ),
            cms.PSet( rawid=cms.uint32(352670724), angle=cms.double(FPix_300V_RNG1_PNL1) ),
            cms.PSet( rawid=cms.uint32(352674820), angle=cms.double(FPix_300V_RNG1_PNL1) ),
            cms.PSet( rawid=cms.uint32(344749060), angle=cms.double(FPix_300V_RNG1_PNL1) ),
            cms.PSet( rawid=cms.uint32(344753156), angle=cms.double(FPix_300V_RNG1_PNL1) ),
            cms.PSet( rawid=cms.uint32(344757252), angle=cms.double(FPix_300V_RNG1_PNL1) ),
            cms.PSet( rawid=cms.uint32(344781828), angle=cms.double(FPix_300V_RNG1_PNL1) ),
            cms.PSet( rawid=cms.uint32(344785924), angle=cms.double(FPix_300V_RNG1_PNL1) ),
            cms.PSet( rawid=cms.uint32(344790020), angle=cms.double(FPix_300V_RNG1_PNL1) ),
            cms.PSet( rawid=cms.uint32(352589828), angle=cms.double(FPix_300V_RNG1_PNL2) ),
            cms.PSet( rawid=cms.uint32(352593924), angle=cms.double(FPix_300V_RNG1_PNL2) ),
            cms.PSet( rawid=cms.uint32(352598020), angle=cms.double(FPix_300V_RNG1_PNL2) ),
            cms.PSet( rawid=cms.uint32(352602116), angle=cms.double(FPix_300V_RNG1_PNL2) ),
            cms.PSet( rawid=cms.uint32(352606212), angle=cms.double(FPix_300V_RNG1_PNL2) ),
            cms.PSet( rawid=cms.uint32(352610308), angle=cms.double(FPix_300V_RNG1_PNL2) ),
            cms.PSet( rawid=cms.uint32(352659460), angle=cms.double(FPix_300V_RNG1_PNL2) ),
            cms.PSet( rawid=cms.uint32(352663556), angle=cms.double(FPix_300V_RNG1_PNL2) ),
            cms.PSet( rawid=cms.uint32(352667652), angle=cms.double(FPix_300V_RNG1_PNL2) ),
            cms.PSet( rawid=cms.uint32(352671748), angle=cms.double(FPix_300V_RNG1_PNL2) ),
            cms.PSet( rawid=cms.uint32(352675844), angle=cms.double(FPix_300V_RNG1_PNL2) ),
            cms.PSet( rawid=cms.uint32(344750084), angle=cms.double(FPix_300V_RNG1_PNL2) ),
            cms.PSet( rawid=cms.uint32(344754180), angle=cms.double(FPix_300V_RNG1_PNL2) ),
            cms.PSet( rawid=cms.uint32(344758276), angle=cms.double(FPix_300V_RNG1_PNL2) ),
            cms.PSet( rawid=cms.uint32(344782852), angle=cms.double(FPix_300V_RNG1_PNL2) ),
            cms.PSet( rawid=cms.uint32(344786948), angle=cms.double(FPix_300V_RNG1_PNL2) ),
            cms.PSet( rawid=cms.uint32(344791044), angle=cms.double(FPix_300V_RNG1_PNL2) ),
            cms.PSet( rawid=cms.uint32(344851460), angle=cms.double(FPix_300V_RNG2_PNL1) ),
            cms.PSet( rawid=cms.uint32(344855556), angle=cms.double(FPix_300V_RNG2_PNL1) ),
            cms.PSet( rawid=cms.uint32(344859652), angle=cms.double(FPix_300V_RNG2_PNL1) ),
            cms.PSet( rawid=cms.uint32(344863748), angle=cms.double(FPix_300V_RNG2_PNL1) ),
            cms.PSet( rawid=cms.uint32(344852484), angle=cms.double(FPix_300V_RNG2_PNL2) ),
            cms.PSet( rawid=cms.uint32(344856580), angle=cms.double(FPix_300V_RNG2_PNL2) ),
            cms.PSet( rawid=cms.uint32(344860676), angle=cms.double(FPix_300V_RNG2_PNL2) )
        )
    
##
## SiPixelLorentzAngleReader (from Event Setup)
##
process.LorentzAngleReader = cms.EDAnalyzer("SiPixelLorentzAngleReader",
                                            printDebug = cms.untracked.uint32(10),
                                            useSimRcd = cms.bool(False),
                                            recoLabel = cms.string("forPhase2" if options.isPhase2 else "") # test the label
                                            )

process.p = cms.Path(process.LorentzAngleReader)
