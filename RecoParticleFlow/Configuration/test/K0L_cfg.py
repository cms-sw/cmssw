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
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/K0L/k0l_1.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/K0L/k0l_2.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/K0L/k0l_3.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/K0L/k0l_4.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/K0L/k0l_5.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/K0L/k0l_6.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/K0L/k0l_7.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/K0L/k0l_8.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/K0L/k0l_9.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/K0L/k0l_10.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/K0L/k0l_11.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/K0L/k0l_12.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/K0L/k0l_13.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/K0L/k0l_14.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/K0L/k0l_15.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/K0L/k0l_16.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/K0L/k0l_17.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/K0L/k0l_18.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/K0L/k0l_19.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/K0L/k0l_20.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/K0L/k0l_21.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/K0L/k0l_22.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/K0L/k0l_23.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/K0L/k0l_24.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/K0L/k0l_25.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/K0L/k0l_26.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/K0L/k0l_27.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/K0L/k0l_28.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/K0L/k0l_29.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/K0L/k0l_30.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/K0L/k0l_41.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/K0L/k0l_42.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/K0L/k0l_43.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/K0L/k0l_44.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/K0L/k0l_45.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/K0L/k0l_46.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/K0L/k0l_47.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/K0L/k0l_48.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/K0L/k0l_49.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/K0L/k0l_50.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/K0L/k0l_51.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/K0L/k0l_52.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/K0L/k0l_53.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/K0L/k0l_54.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/K0L/k0l_55.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/K0L/k0l_56.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/K0L/k0l_57.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/K0L/k0l_58.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/K0L/k0l_59.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/K0L/k0l_60.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/K0L/k0l_61.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/K0L/k0l_62.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/K0L/k0l_63.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/K0L/k0l_64.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/K0L/k0l_65.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/K0L/k0l_66.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/K0L/k0l_67.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/K0L/k0l_68.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/K0L/k0l_69.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/K0L/k0l_70.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/K0L/k0l_71.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/K0L/k0l_72.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/K0L/k0l_73.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/K0L/k0l_74.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/K0L/k0l_75.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/K0L/k0l_76.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/K0L/k0l_77.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/K0L/k0l_78.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/K0L/k0l_79.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/K0L/k0l_80.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/K0L/k0l_81.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/K0L/k0l_82.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/K0L/k0l_83.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/K0L/k0l_84.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/K0L/k0l_85.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/K0L/k0l_86.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/K0L/k0l_87.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/K0L/k0l_88.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/K0L/k0l_89.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/K0L/k0l_90.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/K0L/k0l_91.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/K0L/k0l_92.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/K0L/k0l_93.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/K0L/k0l_94.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/K0L/k0l_95.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/K0L/k0l_96.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/K0L/k0l_97.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/K0L/k0l_98.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/K0L/k0l_99.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/K0L/k0l_100.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/K0L/k0l_101.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/K0L/k0l_102.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/K0L/k0l_103.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/K0L/k0l_104.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/K0L/k0l_105.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/K0L/k0l_106.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/K0L/k0l_107.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/K0L/k0l_108.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/K0L/k0l_109.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/K0L/k0l_110.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/K0L/k0l_111.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/K0L/k0l_112.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/K0L/k0l_113.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/K0L/k0l_114.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/K0L/k0l_115.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/K0L/k0l_116.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/K0L/k0l_117.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/K0L/k0l_118.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/K0L/k0l_119.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/K0L/k0l_120.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/K0L/k0l_121.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/K0L/k0l_122.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/K0L/k0l_123.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/K0L/k0l_124.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/K0L/k0l_125.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/K0L/k0l_126.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/K0L/k0l_127.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/K0L/k0l_128.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/K0L/k0l_129.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/K0L/k0l_130.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/K0L/k0l_141.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/K0L/k0l_142.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/K0L/k0l_143.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/K0L/k0l_144.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/K0L/k0l_145.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/K0L/k0l_146.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/K0L/k0l_147.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/K0L/k0l_148.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/K0L/k0l_149.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/K0L/k0l_150.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/K0L/k0l_151.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/K0L/k0l_152.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/K0L/k0l_153.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/K0L/k0l_154.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/K0L/k0l_155.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/K0L/k0l_156.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/K0L/k0l_157.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/K0L/k0l_158.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/K0L/k0l_159.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/K0L/k0l_160.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/K0L/k0l_161.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/K0L/k0l_162.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/K0L/k0l_163.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/K0L/k0l_164.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/K0L/k0l_165.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/K0L/k0l_166.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/K0L/k0l_167.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/K0L/k0l_168.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/K0L/k0l_169.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/K0L/k0l_170.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/K0L/k0l_171.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/K0L/k0l_172.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/K0L/k0l_173.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/K0L/k0l_174.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/K0L/k0l_175.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/K0L/k0l_176.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/K0L/k0l_177.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/K0L/k0l_178.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/K0L/k0l_179.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/K0L/k0l_180.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/K0L/k0l_181.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/K0L/k0l_182.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/K0L/k0l_183.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/K0L/k0l_184.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/K0L/k0l_185.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/K0L/k0l_186.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/K0L/k0l_187.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/K0L/k0l_188.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/K0L/k0l_189.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/K0L/k0l_190.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/K0L/k0l_191.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/K0L/k0l_192.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/K0L/k0l_193.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/K0L/k0l_194.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/K0L/k0l_195.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/K0L/k0l_196.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/K0L/k0l_197.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/K0L/k0l_198.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/K0L/k0l_199.root",
    "rfio:/castor/cern.ch/user/a/abdullin/single_hadrons_production_PF_3110+SF+C/K0L/k0l_200.root",
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
    rootOutputFile = cms.string("pfcalib_k0L.root"),# the root tree
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


