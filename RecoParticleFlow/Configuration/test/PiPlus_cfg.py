import FWCore.ParameterSet.Config as cms

process = cms.Process("REPROD")
process.load("Configuration.StandardSequences.Reconstruction_cff")
process.load("Configuration.StandardSequences.MagneticField_38T_cff")
#process.load("Configuration.StandardSequences.MagneticField_4T_cff")
process.load("Configuration.StandardSequences.Geometry_cff")
process.load('Configuration/StandardSequences/FrontierConditions_GlobalTag_cff')
from Configuration.PyReleaseValidation.autoCond import autoCond
process.GlobalTag.globaltag = autoCond['startup']

#process.Timing =cms.Service("Timing")
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

process.source = cms.Source(
    "PoolSource",
    fileNames = cms.untracked.vstring(
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/pi+/pi+_1.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/pi+/pi+_2.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/pi+/pi+_3.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/pi+/pi+_4.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/pi+/pi+_5.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/pi+/pi+_6.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/pi+/pi+_7.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/pi+/pi+_8.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/pi+/pi+_9.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/pi+/pi+_10.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/pi+/pi+_11.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/pi+/pi+_12.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/pi+/pi+_13.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/pi+/pi+_14.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/pi+/pi+_15.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/pi+/pi+_16.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/pi+/pi+_17.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/pi+/pi+_18.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/pi+/pi+_19.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/pi+/pi+_20.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/pi+/pi+_21.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/pi+/pi+_22.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/pi+/pi+_23.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/pi+/pi+_24.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/pi+/pi+_25.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/pi+/pi+_26.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/pi+/pi+_27.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/pi+/pi+_28.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/pi+/pi+_29.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/pi+/pi+_30.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/pi+/pi+_41.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/pi+/pi+_42.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/pi+/pi+_43.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/pi+/pi+_44.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/pi+/pi+_45.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/pi+/pi+_46.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/pi+/pi+_47.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/pi+/pi+_48.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/pi+/pi+_49.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/pi+/pi+_50.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/pi+/pi+_51.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/pi+/pi+_52.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/pi+/pi+_53.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/pi+/pi+_54.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/pi+/pi+_55.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/pi+/pi+_56.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/pi+/pi+_57.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/pi+/pi+_58.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/pi+/pi+_59.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/pi+/pi+_60.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/pi+/pi+_61.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/pi+/pi+_62.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/pi+/pi+_63.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/pi+/pi+_64.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/pi+/pi+_65.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/pi+/pi+_66.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/pi+/pi+_67.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/pi+/pi+_68.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/pi+/pi+_69.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/pi+/pi+_70.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/pi+/pi+_71.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/pi+/pi+_72.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/pi+/pi+_73.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/pi+/pi+_74.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/pi+/pi+_75.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/pi+/pi+_76.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/pi+/pi+_77.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/pi+/pi+_78.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/pi+/pi+_79.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/pi+/pi+_80.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/pi+/pi+_81.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/pi+/pi+_82.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/pi+/pi+_83.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/pi+/pi+_84.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/pi+/pi+_85.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/pi+/pi+_86.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/pi+/pi+_87.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/pi+/pi+_88.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/pi+/pi+_89.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/pi+/pi+_90.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/pi+/pi+_91.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/pi+/pi+_92.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/pi+/pi+_93.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/pi+/pi+_94.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/pi+/pi+_95.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/pi+/pi+_96.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/pi+/pi+_97.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/pi+/pi+_98.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/pi+/pi+_99.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/pi+/pi+_100.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/pi+/pi+_101.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/pi+/pi+_102.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/pi+/pi+_103.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/pi+/pi+_104.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/pi+/pi+_105.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/pi+/pi+_106.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/pi+/pi+_107.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/pi+/pi+_108.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/pi+/pi+_109.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/pi+/pi+_110.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/pi+/pi+_111.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/pi+/pi+_112.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/pi+/pi+_113.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/pi+/pi+_114.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/pi+/pi+_115.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/pi+/pi+_116.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/pi+/pi+_117.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/pi+/pi+_118.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/pi+/pi+_119.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/pi+/pi+_120.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/pi+/pi+_121.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/pi+/pi+_122.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/pi+/pi+_123.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/pi+/pi+_124.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/pi+/pi+_125.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/pi+/pi+_126.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/pi+/pi+_127.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/pi+/pi+_128.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/pi+/pi+_129.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/pi+/pi+_130.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/pi+/pi+_141.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/pi+/pi+_142.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/pi+/pi+_143.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/pi+/pi+_144.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/pi+/pi+_145.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/pi+/pi+_146.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/pi+/pi+_147.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/pi+/pi+_148.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/pi+/pi+_149.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/pi+/pi+_150.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/pi+/pi+_151.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/pi+/pi+_152.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/pi+/pi+_153.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/pi+/pi+_154.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/pi+/pi+_155.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/pi+/pi+_156.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/pi+/pi+_157.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/pi+/pi+_158.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/pi+/pi+_159.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/pi+/pi+_160.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/pi+/pi+_161.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/pi+/pi+_162.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/pi+/pi+_163.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/pi+/pi+_164.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/pi+/pi+_165.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/pi+/pi+_166.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/pi+/pi+_167.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/pi+/pi+_168.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/pi+/pi+_169.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/pi+/pi+_170.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/pi+/pi+_171.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/pi+/pi+_172.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/pi+/pi+_173.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/pi+/pi+_174.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/pi+/pi+_175.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/pi+/pi+_176.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/pi+/pi+_177.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/pi+/pi+_178.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/pi+/pi+_179.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/pi+/pi+_180.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/pi+/pi+_181.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/pi+/pi+_182.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/pi+/pi+_183.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/pi+/pi+_184.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/pi+/pi+_185.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/pi+/pi+_186.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/pi+/pi+_187.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/pi+/pi+_188.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/pi+/pi+_189.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/pi+/pi+_190.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/pi+/pi+_191.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/pi+/pi+_192.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/pi+/pi+_193.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/pi+/pi+_194.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/pi+/pi+_195.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/pi+/pi+_196.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/pi+/pi+_197.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/pi+/pi+_198.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/pi+/pi+_199.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/pi+/pi+_200.root",
    ),
    eventsToProcess = cms.untracked.VEventRange(),
    #eventsToProcess = cms.untracked.VEventRange('1:1217421-1:1217421'),
    #                                             '1:1220344-1:1220344',
    #                                             '1:1655912-1:1655912',
    #                                             '1:415027-1:415027',
    #                                             '1:460640-1:460640',
    #                                             '1:2054772-1:2054772'),
    secondaryFileNames = cms.untracked.vstring(),
    noEventSort = cms.untracked.bool(True),
    duplicateCheckMode = cms.untracked.string('noDuplicateCheck')
)

process.pfChargedHadronAnalyzer = cms.EDAnalyzer(
    "PFChargedHadronAnalyzer",
    PFCandidates = cms.InputTag("particleFlow"),
    PFSimParticles = cms.InputTag("particleFlowSimParticle"),
    ptMin = cms.double(1.),                     # Minimum pt
    pMin = cms.double(3.),                      # Minimum p
    nPixMin = cms.int32(2),                     # Nb of pixel hits
    nHitMin = cms.vint32(14,17,20,17,10),       # Nb of track hits
    nEtaMin = cms.vdouble(1.4,1.6,2.0,2.4,2.6), # in these eta ranges
    hcalMin = cms.double(1.),                   # Minimum hcal energy
    ecalMax = cms.double(1E9),                  # Maximum ecal energy 
    verbose = cms.untracked.bool(True),         # not used.
    rootOutputFile = cms.string("pfcalib_piplus.root"),# the root tree
)

#from RecoParticleFlow.Configuration.reco_QCDForPF_cff import fileNames
#process.source.fileNames = fileNames

process.dump = cms.EDAnalyzer("EventContentAnalyzer")


process.load("RecoParticleFlow.Configuration.ReDisplay_EventContent_NoTracking_cff")
process.display = cms.OutputModule("PoolOutputModule",
    process.DisplayEventContent,
    #outputCommands = cms.untracked.vstring('keep *'),
    #process.RECOSIMEventContent,
    fileName = cms.untracked.string('piplus_1.root'),
    SelectEvents = cms.untracked.PSet(SelectEvents = cms.vstring('p'))
)

# modify reconstruction sequence
#process.hbhereflag = process.hbhereco.clone()
#process.hbhereflag.hbheInput = 'hbhereco'
#process.towerMakerPF.hbheInput = 'hbhereflag'
#process.particleFlowRecHitHCAL.hcalRecHitsHBHE = cms.InputTag("hbhereflag")

# Local re-reco: Produce tracker rechits, pf rechits and pf clusters
process.localReReco = cms.Sequence(process.particleFlowCluster)


# Particle Flow re-processing
process.pfReReco = cms.Sequence(process.particleFlowReco+
                                process.recoPFJets+
                                process.recoPFMET+
                                process.PFTau)
                                
# Gen Info re-processing
process.load("PhysicsTools.HepMCCandAlgos.genParticles_cfi")
process.load("RecoJets.Configuration.GenJetParticles_cff")
process.load("RecoJets.Configuration.RecoGenJets_cff")
process.load("RecoMET.Configuration.GenMETParticles_cff")
process.load("RecoMET.Configuration.RecoGenMET_cff")
process.load("RecoParticleFlow.PFProducer.particleFlowSimParticle_cff")
process.load("RecoParticleFlow.Configuration.HepMCCopy_cfi")
process.genReReco = cms.Sequence(process.generator+
                                 process.genParticles+
                                 process.genJetParticles+
                                 process.recoGenJets+
                                 process.genMETParticles+
                                 process.recoGenMET+
                                 process.particleFlowSimParticle)

# The complete reprocessing
process.p = cms.Path(#process.localReReco+
                     #process.pfReReco+
                     process.genReReco+
                     process.pfChargedHadronAnalyzer
                     )

# And the output.
#process.outpath = cms.EndPath(process.display)

# And the logger
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.options = cms.untracked.PSet(
    makeTriggerResults = cms.untracked.bool(True),
    wantSummary = cms.untracked.bool(True),
    Rethrow = cms.untracked.vstring('Unknown', 
        'ProductNotFound', 
        'DictionaryNotFound', 
        'InsertFailure', 
        'Configuration', 
        'LogicError', 
        'UnimplementedFeature', 
        'InvalidReference', 
        'NullPointerError', 
        'NoProductSpecified', 
        'EventTimeout', 
        'EventCorruption', 
        'ModuleFailure', 
        'ScheduleExecutionFailure', 
        'EventProcessorFailure', 
        'FileInPathError', 
        'FatalRootError', 
        'NotFound')
)

process.MessageLogger.cerr.FwkReport.reportEvery = 1000


