import FWCore.ParameterSet.Config as cms

from Configuration.StandardSequences.Eras import eras
import FWCore.ParameterSet.VarParsing as VarParsing

import sys

process = cms.Process('PLOTTER', eras.Run3)

#SETUP PARAMETERS
process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool(True),
    )
options = VarParsing.VarParsing ('analysis')
options.register('outputFileName',
                'PPS_ALCARECO_plots',
                VarParsing.VarParsing.multiplicity.singleton,
                VarParsing.VarParsing.varType.string,
                "output ROOT file name")
options.register('jsonFileName',
                '',
                VarParsing.VarParsing.multiplicity.singleton,
                VarParsing.VarParsing.varType.string,
                "JSON file list name")
options.register('globalTag',
                '',
                VarParsing.VarParsing.multiplicity.singleton,
                VarParsing.VarParsing.varType.string,
                "GT to use")
options.parseArguments()

#CONFIGURE PROCESS
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )

#SETUP GLOBAL TAG
from Configuration.AlCa.GlobalTag import GlobalTag
if options.globalTag != '':
    gt = options.globalTag
else:
    gt = 'auto:run3_data_prompt'

# Load geometry from DB
process.load('Geometry.VeryForwardGeometry.geometryRPFromDB_cfi')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
print('Using GT:',gt)
process.GlobalTag = GlobalTag(process.GlobalTag, gt)

alcarecoSuffix = ''
# Uncomment the line below to run on ALCARECO files
alcarecoSuffix += 'AlCaRecoProducer'

process.load
process.load('Validation.CTPPS.ctppsLHCInfoPlotter_cfi')
process.ctppsLHCInfoPlotter.outputFile = options.outputFileName + '_lhcInfo.root'

process.ctppsTrackDistributionPlotter = cms.EDAnalyzer("CTPPSTrackDistributionPlotter",
    tagTracks = cms.InputTag("ctppsLocalTrackLiteProducer"+alcarecoSuffix),
    outputFile = cms.string(options.outputFileName + '_trackDistribution.root'),
    rpId_45_N = cms.uint32(3),
    rpId_45_F = cms.uint32(23),
    rpId_56_N = cms.uint32(103),
    rpId_56_F = cms.uint32(123),
)

process.ctppsProtonReconstructionPlotter = cms.EDAnalyzer("CTPPSProtonReconstructionPlotter",
    tagTracks = cms.InputTag("ctppsLocalTrackLiteProducer"+alcarecoSuffix),
    tagRecoProtonsSingleRP = cms.InputTag("ctppsProtons"+alcarecoSuffix, "singleRP"),
    tagRecoProtonsMultiRP = cms.InputTag("ctppsProtons"+alcarecoSuffix, "multiRP"),
    outputFile = cms.string(options.outputFileName + '_protonReconstruction.root'),
    rpId_45_N = cms.uint32(3),
    rpId_45_F = cms.uint32(23),
    rpId_56_N = cms.uint32(103),
    rpId_56_F = cms.uint32(123),
)

if len(options.inputFiles) != 0:
    inputFiles = cms.untracked.vstring(options.inputFiles)
else:
    # Example input file
    inputFiles = cms.untracked.vstring(
        [
'/eos/project-c/ctpps/subsystems/Pixel/RPixTracking/EfficiencyCalculation2023/v1/ReferenceSample/PPS_ALCARECO_366933_0.root',
'/eos/project-c/ctpps/subsystems/Pixel/RPixTracking/EfficiencyCalculation2023/v1/ReferenceSample/PPS_ALCARECO_366933_1.root',
'/eos/project-c/ctpps/subsystems/Pixel/RPixTracking/EfficiencyCalculation2023/v1/ReferenceSample/PPS_ALCARECO_366933_10.root',
'/eos/project-c/ctpps/subsystems/Pixel/RPixTracking/EfficiencyCalculation2023/v1/ReferenceSample/PPS_ALCARECO_366933_100.root',
'/eos/project-c/ctpps/subsystems/Pixel/RPixTracking/EfficiencyCalculation2023/v1/ReferenceSample/PPS_ALCARECO_366933_101.root',
'/eos/project-c/ctpps/subsystems/Pixel/RPixTracking/EfficiencyCalculation2023/v1/ReferenceSample/PPS_ALCARECO_366933_102.root',
'/eos/project-c/ctpps/subsystems/Pixel/RPixTracking/EfficiencyCalculation2023/v1/ReferenceSample/PPS_ALCARECO_366933_103.root',
'/eos/project-c/ctpps/subsystems/Pixel/RPixTracking/EfficiencyCalculation2023/v1/ReferenceSample/PPS_ALCARECO_366933_104.root',
'/eos/project-c/ctpps/subsystems/Pixel/RPixTracking/EfficiencyCalculation2023/v1/ReferenceSample/PPS_ALCARECO_366933_105.root',
'/eos/project-c/ctpps/subsystems/Pixel/RPixTracking/EfficiencyCalculation2023/v1/ReferenceSample/PPS_ALCARECO_366933_106.root',
'/eos/project-c/ctpps/subsystems/Pixel/RPixTracking/EfficiencyCalculation2023/v1/ReferenceSample/PPS_ALCARECO_366933_107.root',
'/eos/project-c/ctpps/subsystems/Pixel/RPixTracking/EfficiencyCalculation2023/v1/ReferenceSample/PPS_ALCARECO_366933_108.root',
'/eos/project-c/ctpps/subsystems/Pixel/RPixTracking/EfficiencyCalculation2023/v1/ReferenceSample/PPS_ALCARECO_366933_109.root',
'/eos/project-c/ctpps/subsystems/Pixel/RPixTracking/EfficiencyCalculation2023/v1/ReferenceSample/PPS_ALCARECO_366933_11.root',
'/eos/project-c/ctpps/subsystems/Pixel/RPixTracking/EfficiencyCalculation2023/v1/ReferenceSample/PPS_ALCARECO_366933_110.root',
'/eos/project-c/ctpps/subsystems/Pixel/RPixTracking/EfficiencyCalculation2023/v1/ReferenceSample/PPS_ALCARECO_366933_111.root',
'/eos/project-c/ctpps/subsystems/Pixel/RPixTracking/EfficiencyCalculation2023/v1/ReferenceSample/PPS_ALCARECO_366933_112.root',
'/eos/project-c/ctpps/subsystems/Pixel/RPixTracking/EfficiencyCalculation2023/v1/ReferenceSample/PPS_ALCARECO_366933_113.root',
'/eos/project-c/ctpps/subsystems/Pixel/RPixTracking/EfficiencyCalculation2023/v1/ReferenceSample/PPS_ALCARECO_366933_114.root',
'/eos/project-c/ctpps/subsystems/Pixel/RPixTracking/EfficiencyCalculation2023/v1/ReferenceSample/PPS_ALCARECO_366933_115.root',
'/eos/project-c/ctpps/subsystems/Pixel/RPixTracking/EfficiencyCalculation2023/v1/ReferenceSample/PPS_ALCARECO_366933_116.root',
'/eos/project-c/ctpps/subsystems/Pixel/RPixTracking/EfficiencyCalculation2023/v1/ReferenceSample/PPS_ALCARECO_366933_117.root',
'/eos/project-c/ctpps/subsystems/Pixel/RPixTracking/EfficiencyCalculation2023/v1/ReferenceSample/PPS_ALCARECO_366933_118.root',
'/eos/project-c/ctpps/subsystems/Pixel/RPixTracking/EfficiencyCalculation2023/v1/ReferenceSample/PPS_ALCARECO_366933_119.root',
'/eos/project-c/ctpps/subsystems/Pixel/RPixTracking/EfficiencyCalculation2023/v1/ReferenceSample/PPS_ALCARECO_366933_12.root',
'/eos/project-c/ctpps/subsystems/Pixel/RPixTracking/EfficiencyCalculation2023/v1/ReferenceSample/PPS_ALCARECO_366933_120.root',
'/eos/project-c/ctpps/subsystems/Pixel/RPixTracking/EfficiencyCalculation2023/v1/ReferenceSample/PPS_ALCARECO_366933_121.root',
'/eos/project-c/ctpps/subsystems/Pixel/RPixTracking/EfficiencyCalculation2023/v1/ReferenceSample/PPS_ALCARECO_366933_122.root',
'/eos/project-c/ctpps/subsystems/Pixel/RPixTracking/EfficiencyCalculation2023/v1/ReferenceSample/PPS_ALCARECO_366933_123.root',
'/eos/project-c/ctpps/subsystems/Pixel/RPixTracking/EfficiencyCalculation2023/v1/ReferenceSample/PPS_ALCARECO_366933_124.root',
'/eos/project-c/ctpps/subsystems/Pixel/RPixTracking/EfficiencyCalculation2023/v1/ReferenceSample/PPS_ALCARECO_366933_125.root',
'/eos/project-c/ctpps/subsystems/Pixel/RPixTracking/EfficiencyCalculation2023/v1/ReferenceSample/PPS_ALCARECO_366933_126.root',
'/eos/project-c/ctpps/subsystems/Pixel/RPixTracking/EfficiencyCalculation2023/v1/ReferenceSample/PPS_ALCARECO_366933_127.root',
'/eos/project-c/ctpps/subsystems/Pixel/RPixTracking/EfficiencyCalculation2023/v1/ReferenceSample/PPS_ALCARECO_366933_128.root',
'/eos/project-c/ctpps/subsystems/Pixel/RPixTracking/EfficiencyCalculation2023/v1/ReferenceSample/PPS_ALCARECO_366933_129.root',
'/eos/project-c/ctpps/subsystems/Pixel/RPixTracking/EfficiencyCalculation2023/v1/ReferenceSample/PPS_ALCARECO_366933_13.root',
'/eos/project-c/ctpps/subsystems/Pixel/RPixTracking/EfficiencyCalculation2023/v1/ReferenceSample/PPS_ALCARECO_366933_130.root',
'/eos/project-c/ctpps/subsystems/Pixel/RPixTracking/EfficiencyCalculation2023/v1/ReferenceSample/PPS_ALCARECO_366933_131.root',
'/eos/project-c/ctpps/subsystems/Pixel/RPixTracking/EfficiencyCalculation2023/v1/ReferenceSample/PPS_ALCARECO_366933_132.root',
'/eos/project-c/ctpps/subsystems/Pixel/RPixTracking/EfficiencyCalculation2023/v1/ReferenceSample/PPS_ALCARECO_366933_133.root',
'/eos/project-c/ctpps/subsystems/Pixel/RPixTracking/EfficiencyCalculation2023/v1/ReferenceSample/PPS_ALCARECO_366933_134.root',
'/eos/project-c/ctpps/subsystems/Pixel/RPixTracking/EfficiencyCalculation2023/v1/ReferenceSample/PPS_ALCARECO_366933_135.root',
'/eos/project-c/ctpps/subsystems/Pixel/RPixTracking/EfficiencyCalculation2023/v1/ReferenceSample/PPS_ALCARECO_366933_136.root',
'/eos/project-c/ctpps/subsystems/Pixel/RPixTracking/EfficiencyCalculation2023/v1/ReferenceSample/PPS_ALCARECO_366933_137.root',
'/eos/project-c/ctpps/subsystems/Pixel/RPixTracking/EfficiencyCalculation2023/v1/ReferenceSample/PPS_ALCARECO_366933_138.root',
'/eos/project-c/ctpps/subsystems/Pixel/RPixTracking/EfficiencyCalculation2023/v1/ReferenceSample/PPS_ALCARECO_366933_139.root',
'/eos/project-c/ctpps/subsystems/Pixel/RPixTracking/EfficiencyCalculation2023/v1/ReferenceSample/PPS_ALCARECO_366933_14.root',
'/eos/project-c/ctpps/subsystems/Pixel/RPixTracking/EfficiencyCalculation2023/v1/ReferenceSample/PPS_ALCARECO_366933_140.root',
'/eos/project-c/ctpps/subsystems/Pixel/RPixTracking/EfficiencyCalculation2023/v1/ReferenceSample/PPS_ALCARECO_366933_141.root',
'/eos/project-c/ctpps/subsystems/Pixel/RPixTracking/EfficiencyCalculation2023/v1/ReferenceSample/PPS_ALCARECO_366933_142.root',
'/eos/project-c/ctpps/subsystems/Pixel/RPixTracking/EfficiencyCalculation2023/v1/ReferenceSample/PPS_ALCARECO_366933_143.root',
'/eos/project-c/ctpps/subsystems/Pixel/RPixTracking/EfficiencyCalculation2023/v1/ReferenceSample/PPS_ALCARECO_366933_144.root',
'/eos/project-c/ctpps/subsystems/Pixel/RPixTracking/EfficiencyCalculation2023/v1/ReferenceSample/PPS_ALCARECO_366933_145.root',
'/eos/project-c/ctpps/subsystems/Pixel/RPixTracking/EfficiencyCalculation2023/v1/ReferenceSample/PPS_ALCARECO_366933_146.root',
'/eos/project-c/ctpps/subsystems/Pixel/RPixTracking/EfficiencyCalculation2023/v1/ReferenceSample/PPS_ALCARECO_366933_147.root',
'/eos/project-c/ctpps/subsystems/Pixel/RPixTracking/EfficiencyCalculation2023/v1/ReferenceSample/PPS_ALCARECO_366933_148.root',
'/eos/project-c/ctpps/subsystems/Pixel/RPixTracking/EfficiencyCalculation2023/v1/ReferenceSample/PPS_ALCARECO_366933_149.root',
'/eos/project-c/ctpps/subsystems/Pixel/RPixTracking/EfficiencyCalculation2023/v1/ReferenceSample/PPS_ALCARECO_366933_15.root',
'/eos/project-c/ctpps/subsystems/Pixel/RPixTracking/EfficiencyCalculation2023/v1/ReferenceSample/PPS_ALCARECO_366933_150.root',
'/eos/project-c/ctpps/subsystems/Pixel/RPixTracking/EfficiencyCalculation2023/v1/ReferenceSample/PPS_ALCARECO_366933_151.root',
'/eos/project-c/ctpps/subsystems/Pixel/RPixTracking/EfficiencyCalculation2023/v1/ReferenceSample/PPS_ALCARECO_366933_152.root',
'/eos/project-c/ctpps/subsystems/Pixel/RPixTracking/EfficiencyCalculation2023/v1/ReferenceSample/PPS_ALCARECO_366933_153.root',
'/eos/project-c/ctpps/subsystems/Pixel/RPixTracking/EfficiencyCalculation2023/v1/ReferenceSample/PPS_ALCARECO_366933_154.root',
'/eos/project-c/ctpps/subsystems/Pixel/RPixTracking/EfficiencyCalculation2023/v1/ReferenceSample/PPS_ALCARECO_366933_16.root',
'/eos/project-c/ctpps/subsystems/Pixel/RPixTracking/EfficiencyCalculation2023/v1/ReferenceSample/PPS_ALCARECO_366933_17.root',
'/eos/project-c/ctpps/subsystems/Pixel/RPixTracking/EfficiencyCalculation2023/v1/ReferenceSample/PPS_ALCARECO_366933_18.root',
'/eos/project-c/ctpps/subsystems/Pixel/RPixTracking/EfficiencyCalculation2023/v1/ReferenceSample/PPS_ALCARECO_366933_19.root',
'/eos/project-c/ctpps/subsystems/Pixel/RPixTracking/EfficiencyCalculation2023/v1/ReferenceSample/PPS_ALCARECO_366933_2.root',
'/eos/project-c/ctpps/subsystems/Pixel/RPixTracking/EfficiencyCalculation2023/v1/ReferenceSample/PPS_ALCARECO_366933_20.root',
'/eos/project-c/ctpps/subsystems/Pixel/RPixTracking/EfficiencyCalculation2023/v1/ReferenceSample/PPS_ALCARECO_366933_21.root',
'/eos/project-c/ctpps/subsystems/Pixel/RPixTracking/EfficiencyCalculation2023/v1/ReferenceSample/PPS_ALCARECO_366933_22.root',
'/eos/project-c/ctpps/subsystems/Pixel/RPixTracking/EfficiencyCalculation2023/v1/ReferenceSample/PPS_ALCARECO_366933_23.root',
'/eos/project-c/ctpps/subsystems/Pixel/RPixTracking/EfficiencyCalculation2023/v1/ReferenceSample/PPS_ALCARECO_366933_24.root',
'/eos/project-c/ctpps/subsystems/Pixel/RPixTracking/EfficiencyCalculation2023/v1/ReferenceSample/PPS_ALCARECO_366933_25.root',
'/eos/project-c/ctpps/subsystems/Pixel/RPixTracking/EfficiencyCalculation2023/v1/ReferenceSample/PPS_ALCARECO_366933_26.root',
'/eos/project-c/ctpps/subsystems/Pixel/RPixTracking/EfficiencyCalculation2023/v1/ReferenceSample/PPS_ALCARECO_366933_27.root',
'/eos/project-c/ctpps/subsystems/Pixel/RPixTracking/EfficiencyCalculation2023/v1/ReferenceSample/PPS_ALCARECO_366933_28.root',
'/eos/project-c/ctpps/subsystems/Pixel/RPixTracking/EfficiencyCalculation2023/v1/ReferenceSample/PPS_ALCARECO_366933_29.root',
'/eos/project-c/ctpps/subsystems/Pixel/RPixTracking/EfficiencyCalculation2023/v1/ReferenceSample/PPS_ALCARECO_366933_3.root',
'/eos/project-c/ctpps/subsystems/Pixel/RPixTracking/EfficiencyCalculation2023/v1/ReferenceSample/PPS_ALCARECO_366933_30.root',
'/eos/project-c/ctpps/subsystems/Pixel/RPixTracking/EfficiencyCalculation2023/v1/ReferenceSample/PPS_ALCARECO_366933_31.root',
'/eos/project-c/ctpps/subsystems/Pixel/RPixTracking/EfficiencyCalculation2023/v1/ReferenceSample/PPS_ALCARECO_366933_32.root',
'/eos/project-c/ctpps/subsystems/Pixel/RPixTracking/EfficiencyCalculation2023/v1/ReferenceSample/PPS_ALCARECO_366933_33.root',
'/eos/project-c/ctpps/subsystems/Pixel/RPixTracking/EfficiencyCalculation2023/v1/ReferenceSample/PPS_ALCARECO_366933_34.root',
'/eos/project-c/ctpps/subsystems/Pixel/RPixTracking/EfficiencyCalculation2023/v1/ReferenceSample/PPS_ALCARECO_366933_35.root',
'/eos/project-c/ctpps/subsystems/Pixel/RPixTracking/EfficiencyCalculation2023/v1/ReferenceSample/PPS_ALCARECO_366933_36.root',
'/eos/project-c/ctpps/subsystems/Pixel/RPixTracking/EfficiencyCalculation2023/v1/ReferenceSample/PPS_ALCARECO_366933_37.root',
'/eos/project-c/ctpps/subsystems/Pixel/RPixTracking/EfficiencyCalculation2023/v1/ReferenceSample/PPS_ALCARECO_366933_38.root',
'/eos/project-c/ctpps/subsystems/Pixel/RPixTracking/EfficiencyCalculation2023/v1/ReferenceSample/PPS_ALCARECO_366933_39.root',
'/eos/project-c/ctpps/subsystems/Pixel/RPixTracking/EfficiencyCalculation2023/v1/ReferenceSample/PPS_ALCARECO_366933_4.root',
'/eos/project-c/ctpps/subsystems/Pixel/RPixTracking/EfficiencyCalculation2023/v1/ReferenceSample/PPS_ALCARECO_366933_40.root',
'/eos/project-c/ctpps/subsystems/Pixel/RPixTracking/EfficiencyCalculation2023/v1/ReferenceSample/PPS_ALCARECO_366933_41.root',
'/eos/project-c/ctpps/subsystems/Pixel/RPixTracking/EfficiencyCalculation2023/v1/ReferenceSample/PPS_ALCARECO_366933_42.root',
'/eos/project-c/ctpps/subsystems/Pixel/RPixTracking/EfficiencyCalculation2023/v1/ReferenceSample/PPS_ALCARECO_366933_43.root',
'/eos/project-c/ctpps/subsystems/Pixel/RPixTracking/EfficiencyCalculation2023/v1/ReferenceSample/PPS_ALCARECO_366933_44.root',
'/eos/project-c/ctpps/subsystems/Pixel/RPixTracking/EfficiencyCalculation2023/v1/ReferenceSample/PPS_ALCARECO_366933_45.root',
'/eos/project-c/ctpps/subsystems/Pixel/RPixTracking/EfficiencyCalculation2023/v1/ReferenceSample/PPS_ALCARECO_366933_46.root',
'/eos/project-c/ctpps/subsystems/Pixel/RPixTracking/EfficiencyCalculation2023/v1/ReferenceSample/PPS_ALCARECO_366933_47.root',
'/eos/project-c/ctpps/subsystems/Pixel/RPixTracking/EfficiencyCalculation2023/v1/ReferenceSample/PPS_ALCARECO_366933_48.root',
'/eos/project-c/ctpps/subsystems/Pixel/RPixTracking/EfficiencyCalculation2023/v1/ReferenceSample/PPS_ALCARECO_366933_49.root',
'/eos/project-c/ctpps/subsystems/Pixel/RPixTracking/EfficiencyCalculation2023/v1/ReferenceSample/PPS_ALCARECO_366933_5.root',
'/eos/project-c/ctpps/subsystems/Pixel/RPixTracking/EfficiencyCalculation2023/v1/ReferenceSample/PPS_ALCARECO_366933_50.root',
'/eos/project-c/ctpps/subsystems/Pixel/RPixTracking/EfficiencyCalculation2023/v1/ReferenceSample/PPS_ALCARECO_366933_51.root',
'/eos/project-c/ctpps/subsystems/Pixel/RPixTracking/EfficiencyCalculation2023/v1/ReferenceSample/PPS_ALCARECO_366933_52.root',
'/eos/project-c/ctpps/subsystems/Pixel/RPixTracking/EfficiencyCalculation2023/v1/ReferenceSample/PPS_ALCARECO_366933_53.root',
'/eos/project-c/ctpps/subsystems/Pixel/RPixTracking/EfficiencyCalculation2023/v1/ReferenceSample/PPS_ALCARECO_366933_54.root',
'/eos/project-c/ctpps/subsystems/Pixel/RPixTracking/EfficiencyCalculation2023/v1/ReferenceSample/PPS_ALCARECO_366933_55.root',
'/eos/project-c/ctpps/subsystems/Pixel/RPixTracking/EfficiencyCalculation2023/v1/ReferenceSample/PPS_ALCARECO_366933_56.root',
'/eos/project-c/ctpps/subsystems/Pixel/RPixTracking/EfficiencyCalculation2023/v1/ReferenceSample/PPS_ALCARECO_366933_57.root',
'/eos/project-c/ctpps/subsystems/Pixel/RPixTracking/EfficiencyCalculation2023/v1/ReferenceSample/PPS_ALCARECO_366933_58.root',
'/eos/project-c/ctpps/subsystems/Pixel/RPixTracking/EfficiencyCalculation2023/v1/ReferenceSample/PPS_ALCARECO_366933_59.root',
'/eos/project-c/ctpps/subsystems/Pixel/RPixTracking/EfficiencyCalculation2023/v1/ReferenceSample/PPS_ALCARECO_366933_6.root',
'/eos/project-c/ctpps/subsystems/Pixel/RPixTracking/EfficiencyCalculation2023/v1/ReferenceSample/PPS_ALCARECO_366933_60.root',
'/eos/project-c/ctpps/subsystems/Pixel/RPixTracking/EfficiencyCalculation2023/v1/ReferenceSample/PPS_ALCARECO_366933_61.root',
'/eos/project-c/ctpps/subsystems/Pixel/RPixTracking/EfficiencyCalculation2023/v1/ReferenceSample/PPS_ALCARECO_366933_62.root',
'/eos/project-c/ctpps/subsystems/Pixel/RPixTracking/EfficiencyCalculation2023/v1/ReferenceSample/PPS_ALCARECO_366933_63.root',
'/eos/project-c/ctpps/subsystems/Pixel/RPixTracking/EfficiencyCalculation2023/v1/ReferenceSample/PPS_ALCARECO_366933_64.root',
'/eos/project-c/ctpps/subsystems/Pixel/RPixTracking/EfficiencyCalculation2023/v1/ReferenceSample/PPS_ALCARECO_366933_65.root',
'/eos/project-c/ctpps/subsystems/Pixel/RPixTracking/EfficiencyCalculation2023/v1/ReferenceSample/PPS_ALCARECO_366933_66.root',
'/eos/project-c/ctpps/subsystems/Pixel/RPixTracking/EfficiencyCalculation2023/v1/ReferenceSample/PPS_ALCARECO_366933_67.root',
'/eos/project-c/ctpps/subsystems/Pixel/RPixTracking/EfficiencyCalculation2023/v1/ReferenceSample/PPS_ALCARECO_366933_68.root',
'/eos/project-c/ctpps/subsystems/Pixel/RPixTracking/EfficiencyCalculation2023/v1/ReferenceSample/PPS_ALCARECO_366933_69.root',
'/eos/project-c/ctpps/subsystems/Pixel/RPixTracking/EfficiencyCalculation2023/v1/ReferenceSample/PPS_ALCARECO_366933_7.root',
'/eos/project-c/ctpps/subsystems/Pixel/RPixTracking/EfficiencyCalculation2023/v1/ReferenceSample/PPS_ALCARECO_366933_70.root',
'/eos/project-c/ctpps/subsystems/Pixel/RPixTracking/EfficiencyCalculation2023/v1/ReferenceSample/PPS_ALCARECO_366933_71.root',
'/eos/project-c/ctpps/subsystems/Pixel/RPixTracking/EfficiencyCalculation2023/v1/ReferenceSample/PPS_ALCARECO_366933_72.root',
'/eos/project-c/ctpps/subsystems/Pixel/RPixTracking/EfficiencyCalculation2023/v1/ReferenceSample/PPS_ALCARECO_366933_73.root',
'/eos/project-c/ctpps/subsystems/Pixel/RPixTracking/EfficiencyCalculation2023/v1/ReferenceSample/PPS_ALCARECO_366933_74.root',
'/eos/project-c/ctpps/subsystems/Pixel/RPixTracking/EfficiencyCalculation2023/v1/ReferenceSample/PPS_ALCARECO_366933_75.root',
'/eos/project-c/ctpps/subsystems/Pixel/RPixTracking/EfficiencyCalculation2023/v1/ReferenceSample/PPS_ALCARECO_366933_76.root',
'/eos/project-c/ctpps/subsystems/Pixel/RPixTracking/EfficiencyCalculation2023/v1/ReferenceSample/PPS_ALCARECO_366933_77.root',
'/eos/project-c/ctpps/subsystems/Pixel/RPixTracking/EfficiencyCalculation2023/v1/ReferenceSample/PPS_ALCARECO_366933_78.root',
'/eos/project-c/ctpps/subsystems/Pixel/RPixTracking/EfficiencyCalculation2023/v1/ReferenceSample/PPS_ALCARECO_366933_79.root',
'/eos/project-c/ctpps/subsystems/Pixel/RPixTracking/EfficiencyCalculation2023/v1/ReferenceSample/PPS_ALCARECO_366933_8.root',
'/eos/project-c/ctpps/subsystems/Pixel/RPixTracking/EfficiencyCalculation2023/v1/ReferenceSample/PPS_ALCARECO_366933_80.root',
'/eos/project-c/ctpps/subsystems/Pixel/RPixTracking/EfficiencyCalculation2023/v1/ReferenceSample/PPS_ALCARECO_366933_81.root',
'/eos/project-c/ctpps/subsystems/Pixel/RPixTracking/EfficiencyCalculation2023/v1/ReferenceSample/PPS_ALCARECO_366933_82.root',
'/eos/project-c/ctpps/subsystems/Pixel/RPixTracking/EfficiencyCalculation2023/v1/ReferenceSample/PPS_ALCARECO_366933_83.root',
'/eos/project-c/ctpps/subsystems/Pixel/RPixTracking/EfficiencyCalculation2023/v1/ReferenceSample/PPS_ALCARECO_366933_84.root',
'/eos/project-c/ctpps/subsystems/Pixel/RPixTracking/EfficiencyCalculation2023/v1/ReferenceSample/PPS_ALCARECO_366933_85.root',
'/eos/project-c/ctpps/subsystems/Pixel/RPixTracking/EfficiencyCalculation2023/v1/ReferenceSample/PPS_ALCARECO_366933_86.root',
'/eos/project-c/ctpps/subsystems/Pixel/RPixTracking/EfficiencyCalculation2023/v1/ReferenceSample/PPS_ALCARECO_366933_87.root',
'/eos/project-c/ctpps/subsystems/Pixel/RPixTracking/EfficiencyCalculation2023/v1/ReferenceSample/PPS_ALCARECO_366933_88.root',
'/eos/project-c/ctpps/subsystems/Pixel/RPixTracking/EfficiencyCalculation2023/v1/ReferenceSample/PPS_ALCARECO_366933_89.root',
'/eos/project-c/ctpps/subsystems/Pixel/RPixTracking/EfficiencyCalculation2023/v1/ReferenceSample/PPS_ALCARECO_366933_9.root',
'/eos/project-c/ctpps/subsystems/Pixel/RPixTracking/EfficiencyCalculation2023/v1/ReferenceSample/PPS_ALCARECO_366933_90.root',
'/eos/project-c/ctpps/subsystems/Pixel/RPixTracking/EfficiencyCalculation2023/v1/ReferenceSample/PPS_ALCARECO_366933_91.root',
'/eos/project-c/ctpps/subsystems/Pixel/RPixTracking/EfficiencyCalculation2023/v1/ReferenceSample/PPS_ALCARECO_366933_92.root',
'/eos/project-c/ctpps/subsystems/Pixel/RPixTracking/EfficiencyCalculation2023/v1/ReferenceSample/PPS_ALCARECO_366933_93.root',
'/eos/project-c/ctpps/subsystems/Pixel/RPixTracking/EfficiencyCalculation2023/v1/ReferenceSample/PPS_ALCARECO_366933_94.root',
'/eos/project-c/ctpps/subsystems/Pixel/RPixTracking/EfficiencyCalculation2023/v1/ReferenceSample/PPS_ALCARECO_366933_95.root',
'/eos/project-c/ctpps/subsystems/Pixel/RPixTracking/EfficiencyCalculation2023/v1/ReferenceSample/PPS_ALCARECO_366933_96.root',
'/eos/project-c/ctpps/subsystems/Pixel/RPixTracking/EfficiencyCalculation2023/v1/ReferenceSample/PPS_ALCARECO_366933_97.root',
'/eos/project-c/ctpps/subsystems/Pixel/RPixTracking/EfficiencyCalculation2023/v1/ReferenceSample/PPS_ALCARECO_366933_98.root',
'/eos/project-c/ctpps/subsystems/Pixel/RPixTracking/EfficiencyCalculation2023/v1/ReferenceSample/PPS_ALCARECO_366933_99.root',
        ]
    )
if len(inputFiles) != 0:
    fileList = [f'file:{f}' if not (f.startswith('/store/') or f.startswith('file:') or f.startswith('root:')) else f for f in inputFiles]
    inputFiles = cms.untracked.vstring(fileList)
    print('Input files:')
    print(inputFiles)

process.source = cms.Source("PoolSource",
    fileNames = inputFiles,
    # Drop everything from the prompt alcareco besides the digis at input
    inputCommands = cms.untracked.vstring(
        'keep *'
    )
)

if options.jsonFileName != '':
    import FWCore.PythonUtilities.LumiList as LumiList
    jsonFileName = options.jsonFileName
    print("Using JSON file:",jsonFileName)
    process.source.lumisToProcess = LumiList.LumiList(filename = jsonFileName).getVLuminosityBlockRange()


# processing sequences
process.path = cms.Path(
    process.ctppsLHCInfoPlotter *
    process.ctppsTrackDistributionPlotter *
    process.ctppsProtonReconstructionPlotter
)
