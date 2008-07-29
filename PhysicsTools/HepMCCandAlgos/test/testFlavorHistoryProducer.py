from FWCore.ParameterSet.Config import *

process = Process("testGenParticles")


# request a summary at the end of the file
process.options = untracked.PSet(
    wantSummary = untracked.bool(True)
)

process.load( "FWCore.MessageLogger.MessageLogger_cfi" )
process.load( "SimGeneral.HepPDTESSource.pythiapdt_cfi")
process.load( "PhysicsTools.HepMCCandAlgos.genParticles_cfi")
process.load( "PhysicsTools.HepMCCandAlgos.genEventWeight_cfi")
process.load( "PhysicsTools.HepMCCandAlgos.genEventScale_cfi")

process.load("PhysicsTools.HepMCCandAlgos.flavorHistoryProducer_cfi")
process.load("PhysicsTools.HepMCCandAlgos.flavorHistoryFilter_cfi")

process.printList = EDAnalyzer( "ParticleListDrawer",
                                src = InputTag( "genParticles" ),
                                maxEventsToPrint = untracked.int32( 10 )
)


process.printTree = EDAnalyzer( "ParticleTreeDrawer",
  src = InputTag( "genParticles" ),
#    printP4 = cms.untracked.bool( True ),
#    printPtEtaPhi = cms.untracked.bool( True ),
#    printStatus = cms.untracked.bool( True ),
  status = untracked.vint32( 2, 3 ),
  printIndex = untracked.bool(True )
)


process.printDecay = EDAnalyzer( "ParticleDecayDrawer",
  src = InputTag( "genParticles" ),
#    untracked bool printP4 = true
#    untracked bool printPtEtaPhi = true
  status = untracked.vint32( 2, 3 )
)


process.add_( Service("RandomNumberGeneratorService",
              sourceSeed= untracked.uint32( 123456789 ) ) )

process.maxEvents = untracked.PSet( input = untracked.int32(-1) )



process.source = Source("PoolSource",
                        debugVerbosity = untracked.uint32(200),
                        debugFlag = untracked.bool(True),
                        
                        fileNames = untracked.vstring(
    'dcache:/pnfs/cms/WAX/resilient/rappocc/wjets/wbb_10TeV_100.root',
    'dcache:/pnfs/cms/WAX/resilient/rappocc/wjets/wbb_10TeV_101.root',
    'dcache:/pnfs/cms/WAX/resilient/rappocc/wjets/wbb_10TeV_102.root',
    'dcache:/pnfs/cms/WAX/resilient/rappocc/wjets/wbb_10TeV_103.root',
    'dcache:/pnfs/cms/WAX/resilient/rappocc/wjets/wbb_10TeV_104.root',
    'dcache:/pnfs/cms/WAX/resilient/rappocc/wjets/wbb_10TeV_105.root',
    'dcache:/pnfs/cms/WAX/resilient/rappocc/wjets/wbb_10TeV_106.root',
    'dcache:/pnfs/cms/WAX/resilient/rappocc/wjets/wbb_10TeV_107.root',
    'dcache:/pnfs/cms/WAX/resilient/rappocc/wjets/wbb_10TeV_108.root',
    'dcache:/pnfs/cms/WAX/resilient/rappocc/wjets/wbb_10TeV_109.root',
    'dcache:/pnfs/cms/WAX/resilient/rappocc/wjets/wbb_10TeV_10.root',
    'dcache:/pnfs/cms/WAX/resilient/rappocc/wjets/wbb_10TeV_110.root',
    'dcache:/pnfs/cms/WAX/resilient/rappocc/wjets/wbb_10TeV_111.root',
    'dcache:/pnfs/cms/WAX/resilient/rappocc/wjets/wbb_10TeV_112.root',
    'dcache:/pnfs/cms/WAX/resilient/rappocc/wjets/wbb_10TeV_113.root',
    'dcache:/pnfs/cms/WAX/resilient/rappocc/wjets/wbb_10TeV_114.root',
    'dcache:/pnfs/cms/WAX/resilient/rappocc/wjets/wbb_10TeV_115.root',
    'dcache:/pnfs/cms/WAX/resilient/rappocc/wjets/wbb_10TeV_116.root',
    'dcache:/pnfs/cms/WAX/resilient/rappocc/wjets/wbb_10TeV_117.root',
    'dcache:/pnfs/cms/WAX/resilient/rappocc/wjets/wbb_10TeV_118.root',
    'dcache:/pnfs/cms/WAX/resilient/rappocc/wjets/wbb_10TeV_119.root',
    'dcache:/pnfs/cms/WAX/resilient/rappocc/wjets/wbb_10TeV_11.root',
    'dcache:/pnfs/cms/WAX/resilient/rappocc/wjets/wbb_10TeV_120.root',
    'dcache:/pnfs/cms/WAX/resilient/rappocc/wjets/wbb_10TeV_121.root',
    'dcache:/pnfs/cms/WAX/resilient/rappocc/wjets/wbb_10TeV_122.root',
    'dcache:/pnfs/cms/WAX/resilient/rappocc/wjets/wbb_10TeV_123.root',
    'dcache:/pnfs/cms/WAX/resilient/rappocc/wjets/wbb_10TeV_124.root',
    'dcache:/pnfs/cms/WAX/resilient/rappocc/wjets/wbb_10TeV_125.root',
    'dcache:/pnfs/cms/WAX/resilient/rappocc/wjets/wbb_10TeV_126.root',
    'dcache:/pnfs/cms/WAX/resilient/rappocc/wjets/wbb_10TeV_128.root',
    'dcache:/pnfs/cms/WAX/resilient/rappocc/wjets/wbb_10TeV_129.root',
    'dcache:/pnfs/cms/WAX/resilient/rappocc/wjets/wbb_10TeV_12.root',
    'dcache:/pnfs/cms/WAX/resilient/rappocc/wjets/wbb_10TeV_130.root',
    'dcache:/pnfs/cms/WAX/resilient/rappocc/wjets/wbb_10TeV_131.root',
    'dcache:/pnfs/cms/WAX/resilient/rappocc/wjets/wbb_10TeV_132.root',
    'dcache:/pnfs/cms/WAX/resilient/rappocc/wjets/wbb_10TeV_133.root',
    'dcache:/pnfs/cms/WAX/resilient/rappocc/wjets/wbb_10TeV_134.root',
    'dcache:/pnfs/cms/WAX/resilient/rappocc/wjets/wbb_10TeV_135.root',
    'dcache:/pnfs/cms/WAX/resilient/rappocc/wjets/wbb_10TeV_136.root',
    'dcache:/pnfs/cms/WAX/resilient/rappocc/wjets/wbb_10TeV_137.root',
    'dcache:/pnfs/cms/WAX/resilient/rappocc/wjets/wbb_10TeV_138.root',
    'dcache:/pnfs/cms/WAX/resilient/rappocc/wjets/wbb_10TeV_139.root',
    'dcache:/pnfs/cms/WAX/resilient/rappocc/wjets/wbb_10TeV_13.root',
    'dcache:/pnfs/cms/WAX/resilient/rappocc/wjets/wbb_10TeV_140.root',
    'dcache:/pnfs/cms/WAX/resilient/rappocc/wjets/wbb_10TeV_141.root',
    'dcache:/pnfs/cms/WAX/resilient/rappocc/wjets/wbb_10TeV_142.root',
    'dcache:/pnfs/cms/WAX/resilient/rappocc/wjets/wbb_10TeV_143.root',
    'dcache:/pnfs/cms/WAX/resilient/rappocc/wjets/wbb_10TeV_144.root',
    'dcache:/pnfs/cms/WAX/resilient/rappocc/wjets/wbb_10TeV_145.root',
    'dcache:/pnfs/cms/WAX/resilient/rappocc/wjets/wbb_10TeV_146.root',
    'dcache:/pnfs/cms/WAX/resilient/rappocc/wjets/wbb_10TeV_147.root',
    'dcache:/pnfs/cms/WAX/resilient/rappocc/wjets/wbb_10TeV_148.root',
    'dcache:/pnfs/cms/WAX/resilient/rappocc/wjets/wbb_10TeV_149.root',
    'dcache:/pnfs/cms/WAX/resilient/rappocc/wjets/wbb_10TeV_14.root',
    'dcache:/pnfs/cms/WAX/resilient/rappocc/wjets/wbb_10TeV_150.root',
    'dcache:/pnfs/cms/WAX/resilient/rappocc/wjets/wbb_10TeV_151.root',
    'dcache:/pnfs/cms/WAX/resilient/rappocc/wjets/wbb_10TeV_152.root',
    'dcache:/pnfs/cms/WAX/resilient/rappocc/wjets/wbb_10TeV_153.root',
    'dcache:/pnfs/cms/WAX/resilient/rappocc/wjets/wbb_10TeV_154.root',
#    'dcache:/pnfs/cms/WAX/resilient/rappocc/wjets/wbb_10TeV_155.root',
    'dcache:/pnfs/cms/WAX/resilient/rappocc/wjets/wbb_10TeV_156.root',
    'dcache:/pnfs/cms/WAX/resilient/rappocc/wjets/wbb_10TeV_157.root',
    'dcache:/pnfs/cms/WAX/resilient/rappocc/wjets/wbb_10TeV_158.root',
    'dcache:/pnfs/cms/WAX/resilient/rappocc/wjets/wbb_10TeV_159.root',
    'dcache:/pnfs/cms/WAX/resilient/rappocc/wjets/wbb_10TeV_15.root',
    'dcache:/pnfs/cms/WAX/resilient/rappocc/wjets/wbb_10TeV_160.root',
    'dcache:/pnfs/cms/WAX/resilient/rappocc/wjets/wbb_10TeV_161.root',
    'dcache:/pnfs/cms/WAX/resilient/rappocc/wjets/wbb_10TeV_162.root',
    'dcache:/pnfs/cms/WAX/resilient/rappocc/wjets/wbb_10TeV_163.root',
    'dcache:/pnfs/cms/WAX/resilient/rappocc/wjets/wbb_10TeV_164.root',
    'dcache:/pnfs/cms/WAX/resilient/rappocc/wjets/wbb_10TeV_165.root',
    'dcache:/pnfs/cms/WAX/resilient/rappocc/wjets/wbb_10TeV_166.root',
    'dcache:/pnfs/cms/WAX/resilient/rappocc/wjets/wbb_10TeV_167.root',
    'dcache:/pnfs/cms/WAX/resilient/rappocc/wjets/wbb_10TeV_168.root',
    'dcache:/pnfs/cms/WAX/resilient/rappocc/wjets/wbb_10TeV_169.root',
    'dcache:/pnfs/cms/WAX/resilient/rappocc/wjets/wbb_10TeV_16.root',
    'dcache:/pnfs/cms/WAX/resilient/rappocc/wjets/wbb_10TeV_170.root',
    'dcache:/pnfs/cms/WAX/resilient/rappocc/wjets/wbb_10TeV_171.root',
    'dcache:/pnfs/cms/WAX/resilient/rappocc/wjets/wbb_10TeV_172.root',
    'dcache:/pnfs/cms/WAX/resilient/rappocc/wjets/wbb_10TeV_173.root',
    'dcache:/pnfs/cms/WAX/resilient/rappocc/wjets/wbb_10TeV_174.root',
    'dcache:/pnfs/cms/WAX/resilient/rappocc/wjets/wbb_10TeV_175.root',
    'dcache:/pnfs/cms/WAX/resilient/rappocc/wjets/wbb_10TeV_176.root',
    'dcache:/pnfs/cms/WAX/resilient/rappocc/wjets/wbb_10TeV_177.root',
    'dcache:/pnfs/cms/WAX/resilient/rappocc/wjets/wbb_10TeV_178.root',
    'dcache:/pnfs/cms/WAX/resilient/rappocc/wjets/wbb_10TeV_179.root',
    'dcache:/pnfs/cms/WAX/resilient/rappocc/wjets/wbb_10TeV_17.root',
    'dcache:/pnfs/cms/WAX/resilient/rappocc/wjets/wbb_10TeV_180.root',
    'dcache:/pnfs/cms/WAX/resilient/rappocc/wjets/wbb_10TeV_181.root',
    'dcache:/pnfs/cms/WAX/resilient/rappocc/wjets/wbb_10TeV_182.root',
    'dcache:/pnfs/cms/WAX/resilient/rappocc/wjets/wbb_10TeV_183.root',
    'dcache:/pnfs/cms/WAX/resilient/rappocc/wjets/wbb_10TeV_184.root',
    'dcache:/pnfs/cms/WAX/resilient/rappocc/wjets/wbb_10TeV_185.root',
    'dcache:/pnfs/cms/WAX/resilient/rappocc/wjets/wbb_10TeV_186.root',
    'dcache:/pnfs/cms/WAX/resilient/rappocc/wjets/wbb_10TeV_187.root',
    'dcache:/pnfs/cms/WAX/resilient/rappocc/wjets/wbb_10TeV_188.root',
    'dcache:/pnfs/cms/WAX/resilient/rappocc/wjets/wbb_10TeV_189.root',
    'dcache:/pnfs/cms/WAX/resilient/rappocc/wjets/wbb_10TeV_18.root',
    'dcache:/pnfs/cms/WAX/resilient/rappocc/wjets/wbb_10TeV_190.root',
    'dcache:/pnfs/cms/WAX/resilient/rappocc/wjets/wbb_10TeV_191.root',
    'dcache:/pnfs/cms/WAX/resilient/rappocc/wjets/wbb_10TeV_192.root',
    'dcache:/pnfs/cms/WAX/resilient/rappocc/wjets/wbb_10TeV_193.root',
    'dcache:/pnfs/cms/WAX/resilient/rappocc/wjets/wbb_10TeV_194.root',
    'dcache:/pnfs/cms/WAX/resilient/rappocc/wjets/wbb_10TeV_195.root',
    'dcache:/pnfs/cms/WAX/resilient/rappocc/wjets/wbb_10TeV_196.root',
    'dcache:/pnfs/cms/WAX/resilient/rappocc/wjets/wbb_10TeV_197.root',
    'dcache:/pnfs/cms/WAX/resilient/rappocc/wjets/wbb_10TeV_198.root',
    'dcache:/pnfs/cms/WAX/resilient/rappocc/wjets/wbb_10TeV_199.root',
    'dcache:/pnfs/cms/WAX/resilient/rappocc/wjets/wbb_10TeV_19.root',
    'dcache:/pnfs/cms/WAX/resilient/rappocc/wjets/wbb_10TeV_1.root',
    'dcache:/pnfs/cms/WAX/resilient/rappocc/wjets/wbb_10TeV_200.root',
    'dcache:/pnfs/cms/WAX/resilient/rappocc/wjets/wbb_10TeV_201.root',
    'dcache:/pnfs/cms/WAX/resilient/rappocc/wjets/wbb_10TeV_202.root',
    'dcache:/pnfs/cms/WAX/resilient/rappocc/wjets/wbb_10TeV_203.root',
    'dcache:/pnfs/cms/WAX/resilient/rappocc/wjets/wbb_10TeV_204.root',
    'dcache:/pnfs/cms/WAX/resilient/rappocc/wjets/wbb_10TeV_205.root',
    'dcache:/pnfs/cms/WAX/resilient/rappocc/wjets/wbb_10TeV_206.root',
    'dcache:/pnfs/cms/WAX/resilient/rappocc/wjets/wbb_10TeV_207.root',
    'dcache:/pnfs/cms/WAX/resilient/rappocc/wjets/wbb_10TeV_208.root',
    'dcache:/pnfs/cms/WAX/resilient/rappocc/wjets/wbb_10TeV_209.root',
    'dcache:/pnfs/cms/WAX/resilient/rappocc/wjets/wbb_10TeV_210.root',
    'dcache:/pnfs/cms/WAX/resilient/rappocc/wjets/wbb_10TeV_211.root',
    'dcache:/pnfs/cms/WAX/resilient/rappocc/wjets/wbb_10TeV_212.root',
    'dcache:/pnfs/cms/WAX/resilient/rappocc/wjets/wbb_10TeV_213.root',
    'dcache:/pnfs/cms/WAX/resilient/rappocc/wjets/wbb_10TeV_214.root',
    'dcache:/pnfs/cms/WAX/resilient/rappocc/wjets/wbb_10TeV_215.root',
    'dcache:/pnfs/cms/WAX/resilient/rappocc/wjets/wbb_10TeV_216.root',
    'dcache:/pnfs/cms/WAX/resilient/rappocc/wjets/wbb_10TeV_217.root',
    'dcache:/pnfs/cms/WAX/resilient/rappocc/wjets/wbb_10TeV_218.root',
    'dcache:/pnfs/cms/WAX/resilient/rappocc/wjets/wbb_10TeV_21.root',
    'dcache:/pnfs/cms/WAX/resilient/rappocc/wjets/wbb_10TeV_220.root',
    'dcache:/pnfs/cms/WAX/resilient/rappocc/wjets/wbb_10TeV_221.root',
    'dcache:/pnfs/cms/WAX/resilient/rappocc/wjets/wbb_10TeV_222.root',
    'dcache:/pnfs/cms/WAX/resilient/rappocc/wjets/wbb_10TeV_223.root',
    'dcache:/pnfs/cms/WAX/resilient/rappocc/wjets/wbb_10TeV_224.root',
    'dcache:/pnfs/cms/WAX/resilient/rappocc/wjets/wbb_10TeV_225.root',
    'dcache:/pnfs/cms/WAX/resilient/rappocc/wjets/wbb_10TeV_226.root',
    'dcache:/pnfs/cms/WAX/resilient/rappocc/wjets/wbb_10TeV_227.root',
    'dcache:/pnfs/cms/WAX/resilient/rappocc/wjets/wbb_10TeV_228.root',
    'dcache:/pnfs/cms/WAX/resilient/rappocc/wjets/wbb_10TeV_229.root',
    'dcache:/pnfs/cms/WAX/resilient/rappocc/wjets/wbb_10TeV_22.root',
    'dcache:/pnfs/cms/WAX/resilient/rappocc/wjets/wbb_10TeV_230.root',
    'dcache:/pnfs/cms/WAX/resilient/rappocc/wjets/wbb_10TeV_231.root',
    'dcache:/pnfs/cms/WAX/resilient/rappocc/wjets/wbb_10TeV_232.root',
    'dcache:/pnfs/cms/WAX/resilient/rappocc/wjets/wbb_10TeV_233.root',
    'dcache:/pnfs/cms/WAX/resilient/rappocc/wjets/wbb_10TeV_234.root',
    'dcache:/pnfs/cms/WAX/resilient/rappocc/wjets/wbb_10TeV_235.root',
    'dcache:/pnfs/cms/WAX/resilient/rappocc/wjets/wbb_10TeV_236.root',
    'dcache:/pnfs/cms/WAX/resilient/rappocc/wjets/wbb_10TeV_237.root',
    'dcache:/pnfs/cms/WAX/resilient/rappocc/wjets/wbb_10TeV_238.root',
    'dcache:/pnfs/cms/WAX/resilient/rappocc/wjets/wbb_10TeV_239.root',
    'dcache:/pnfs/cms/WAX/resilient/rappocc/wjets/wbb_10TeV_23.root',
    'dcache:/pnfs/cms/WAX/resilient/rappocc/wjets/wbb_10TeV_240.root',
    'dcache:/pnfs/cms/WAX/resilient/rappocc/wjets/wbb_10TeV_241.root',
    'dcache:/pnfs/cms/WAX/resilient/rappocc/wjets/wbb_10TeV_242.root',
    'dcache:/pnfs/cms/WAX/resilient/rappocc/wjets/wbb_10TeV_243.root',
    'dcache:/pnfs/cms/WAX/resilient/rappocc/wjets/wbb_10TeV_244.root',
    'dcache:/pnfs/cms/WAX/resilient/rappocc/wjets/wbb_10TeV_245.root',
    'dcache:/pnfs/cms/WAX/resilient/rappocc/wjets/wbb_10TeV_246.root',
    'dcache:/pnfs/cms/WAX/resilient/rappocc/wjets/wbb_10TeV_247.root',
    'dcache:/pnfs/cms/WAX/resilient/rappocc/wjets/wbb_10TeV_248.root',
    'dcache:/pnfs/cms/WAX/resilient/rappocc/wjets/wbb_10TeV_249.root',
    'dcache:/pnfs/cms/WAX/resilient/rappocc/wjets/wbb_10TeV_24.root',
    'dcache:/pnfs/cms/WAX/resilient/rappocc/wjets/wbb_10TeV_250.root',
    'dcache:/pnfs/cms/WAX/resilient/rappocc/wjets/wbb_10TeV_25.root',
    'dcache:/pnfs/cms/WAX/resilient/rappocc/wjets/wbb_10TeV_26.root',
    'dcache:/pnfs/cms/WAX/resilient/rappocc/wjets/wbb_10TeV_27.root',
    'dcache:/pnfs/cms/WAX/resilient/rappocc/wjets/wbb_10TeV_28.root',
    'dcache:/pnfs/cms/WAX/resilient/rappocc/wjets/wbb_10TeV_29.root',
    'dcache:/pnfs/cms/WAX/resilient/rappocc/wjets/wbb_10TeV_2.root',
    'dcache:/pnfs/cms/WAX/resilient/rappocc/wjets/wbb_10TeV_30.root',
    'dcache:/pnfs/cms/WAX/resilient/rappocc/wjets/wbb_10TeV_31.root',
    'dcache:/pnfs/cms/WAX/resilient/rappocc/wjets/wbb_10TeV_32.root',
    'dcache:/pnfs/cms/WAX/resilient/rappocc/wjets/wbb_10TeV_33.root',
    'dcache:/pnfs/cms/WAX/resilient/rappocc/wjets/wbb_10TeV_34.root',
    'dcache:/pnfs/cms/WAX/resilient/rappocc/wjets/wbb_10TeV_36.root',
    'dcache:/pnfs/cms/WAX/resilient/rappocc/wjets/wbb_10TeV_37.root',
    'dcache:/pnfs/cms/WAX/resilient/rappocc/wjets/wbb_10TeV_38.root',
    'dcache:/pnfs/cms/WAX/resilient/rappocc/wjets/wbb_10TeV_39.root',
    'dcache:/pnfs/cms/WAX/resilient/rappocc/wjets/wbb_10TeV_3.root',
    'dcache:/pnfs/cms/WAX/resilient/rappocc/wjets/wbb_10TeV_40.root',
    'dcache:/pnfs/cms/WAX/resilient/rappocc/wjets/wbb_10TeV_41.root',
    'dcache:/pnfs/cms/WAX/resilient/rappocc/wjets/wbb_10TeV_42.root',
    'dcache:/pnfs/cms/WAX/resilient/rappocc/wjets/wbb_10TeV_43.root',
    'dcache:/pnfs/cms/WAX/resilient/rappocc/wjets/wbb_10TeV_44.root',
    'dcache:/pnfs/cms/WAX/resilient/rappocc/wjets/wbb_10TeV_45.root',
    'dcache:/pnfs/cms/WAX/resilient/rappocc/wjets/wbb_10TeV_46.root',
    'dcache:/pnfs/cms/WAX/resilient/rappocc/wjets/wbb_10TeV_47.root',
    'dcache:/pnfs/cms/WAX/resilient/rappocc/wjets/wbb_10TeV_48.root',
    'dcache:/pnfs/cms/WAX/resilient/rappocc/wjets/wbb_10TeV_49.root',
    'dcache:/pnfs/cms/WAX/resilient/rappocc/wjets/wbb_10TeV_4.root',
    'dcache:/pnfs/cms/WAX/resilient/rappocc/wjets/wbb_10TeV_50.root',
    'dcache:/pnfs/cms/WAX/resilient/rappocc/wjets/wbb_10TeV_51.root',
    'dcache:/pnfs/cms/WAX/resilient/rappocc/wjets/wbb_10TeV_52.root',
    'dcache:/pnfs/cms/WAX/resilient/rappocc/wjets/wbb_10TeV_53.root',
    'dcache:/pnfs/cms/WAX/resilient/rappocc/wjets/wbb_10TeV_54.root',
    'dcache:/pnfs/cms/WAX/resilient/rappocc/wjets/wbb_10TeV_55.root',
    'dcache:/pnfs/cms/WAX/resilient/rappocc/wjets/wbb_10TeV_56.root',
    'dcache:/pnfs/cms/WAX/resilient/rappocc/wjets/wbb_10TeV_57.root',
    'dcache:/pnfs/cms/WAX/resilient/rappocc/wjets/wbb_10TeV_58.root',
    'dcache:/pnfs/cms/WAX/resilient/rappocc/wjets/wbb_10TeV_59.root',
    'dcache:/pnfs/cms/WAX/resilient/rappocc/wjets/wbb_10TeV_5.root',
    'dcache:/pnfs/cms/WAX/resilient/rappocc/wjets/wbb_10TeV_60.root',
    'dcache:/pnfs/cms/WAX/resilient/rappocc/wjets/wbb_10TeV_61.root',
    'dcache:/pnfs/cms/WAX/resilient/rappocc/wjets/wbb_10TeV_62.root',
    'dcache:/pnfs/cms/WAX/resilient/rappocc/wjets/wbb_10TeV_63.root',
    'dcache:/pnfs/cms/WAX/resilient/rappocc/wjets/wbb_10TeV_64.root',
    'dcache:/pnfs/cms/WAX/resilient/rappocc/wjets/wbb_10TeV_65.root',
    'dcache:/pnfs/cms/WAX/resilient/rappocc/wjets/wbb_10TeV_66.root',
    'dcache:/pnfs/cms/WAX/resilient/rappocc/wjets/wbb_10TeV_67.root',
    'dcache:/pnfs/cms/WAX/resilient/rappocc/wjets/wbb_10TeV_68.root',
    'dcache:/pnfs/cms/WAX/resilient/rappocc/wjets/wbb_10TeV_69.root',
    'dcache:/pnfs/cms/WAX/resilient/rappocc/wjets/wbb_10TeV_6.root',
    'dcache:/pnfs/cms/WAX/resilient/rappocc/wjets/wbb_10TeV_70.root',
    'dcache:/pnfs/cms/WAX/resilient/rappocc/wjets/wbb_10TeV_71.root',
    'dcache:/pnfs/cms/WAX/resilient/rappocc/wjets/wbb_10TeV_72.root',
    'dcache:/pnfs/cms/WAX/resilient/rappocc/wjets/wbb_10TeV_73.root',
    'dcache:/pnfs/cms/WAX/resilient/rappocc/wjets/wbb_10TeV_74.root',
    'dcache:/pnfs/cms/WAX/resilient/rappocc/wjets/wbb_10TeV_75.root',
    'dcache:/pnfs/cms/WAX/resilient/rappocc/wjets/wbb_10TeV_76.root',
    'dcache:/pnfs/cms/WAX/resilient/rappocc/wjets/wbb_10TeV_77.root',
    'dcache:/pnfs/cms/WAX/resilient/rappocc/wjets/wbb_10TeV_78.root',
    'dcache:/pnfs/cms/WAX/resilient/rappocc/wjets/wbb_10TeV_79.root',
    'dcache:/pnfs/cms/WAX/resilient/rappocc/wjets/wbb_10TeV_7.root',
    'dcache:/pnfs/cms/WAX/resilient/rappocc/wjets/wbb_10TeV_80.root',
    'dcache:/pnfs/cms/WAX/resilient/rappocc/wjets/wbb_10TeV_81.root',
    'dcache:/pnfs/cms/WAX/resilient/rappocc/wjets/wbb_10TeV_82.root',
    'dcache:/pnfs/cms/WAX/resilient/rappocc/wjets/wbb_10TeV_83.root',
    'dcache:/pnfs/cms/WAX/resilient/rappocc/wjets/wbb_10TeV_84.root',
    'dcache:/pnfs/cms/WAX/resilient/rappocc/wjets/wbb_10TeV_85.root',
    'dcache:/pnfs/cms/WAX/resilient/rappocc/wjets/wbb_10TeV_86.root',
    'dcache:/pnfs/cms/WAX/resilient/rappocc/wjets/wbb_10TeV_87.root',
    'dcache:/pnfs/cms/WAX/resilient/rappocc/wjets/wbb_10TeV_88.root',
    'dcache:/pnfs/cms/WAX/resilient/rappocc/wjets/wbb_10TeV_89.root',
    'dcache:/pnfs/cms/WAX/resilient/rappocc/wjets/wbb_10TeV_8.root',
    'dcache:/pnfs/cms/WAX/resilient/rappocc/wjets/wbb_10TeV_90.root',
    'dcache:/pnfs/cms/WAX/resilient/rappocc/wjets/wbb_10TeV_91.root',
    'dcache:/pnfs/cms/WAX/resilient/rappocc/wjets/wbb_10TeV_92.root',
    'dcache:/pnfs/cms/WAX/resilient/rappocc/wjets/wbb_10TeV_93.root',
    'dcache:/pnfs/cms/WAX/resilient/rappocc/wjets/wbb_10TeV_94.root',
    'dcache:/pnfs/cms/WAX/resilient/rappocc/wjets/wbb_10TeV_95.root',
    'dcache:/pnfs/cms/WAX/resilient/rappocc/wjets/wbb_10TeV_96.root',
    'dcache:/pnfs/cms/WAX/resilient/rappocc/wjets/wbb_10TeV_97.root',
    'dcache:/pnfs/cms/WAX/resilient/rappocc/wjets/wbb_10TeV_98.root',
    'dcache:/pnfs/cms/WAX/resilient/rappocc/wjets/wbb_10TeV_99.root',
    'dcache:/pnfs/cms/WAX/resilient/rappocc/wjets/wbb_10TeV_9.root'
    
    )
                        )


#from PhysicsTools.HepMCCandAlgos.data.h4l_cff import pythiaSource
#process.source = pythiaSource

  
process.out = OutputModule( "PoolOutputModule",
  fileName = untracked.string( "/uscms_data/d1/rappocc/wbb_hffilterstudies_genevents.root" ),
  outputCommands= untracked.vstring(
    "drop *",
    "keep *_sisCone5GenJets_*_*",
    "keep *_genParticles_*_*",
    "keep *_genEventWeight_*_*",
    "keep *_flavorHistoryProducer_*_*"
  )
)
  
process.p = Path( 
  process.genParticles *
  process.genEventWeight *
#  process.printDecay *
#  process.printList *
#  process.printTree *
  process.flavorHistoryProducer*
  process.flavorHistoryFilter
)

process.o = EndPath( 
  process.out 
)
