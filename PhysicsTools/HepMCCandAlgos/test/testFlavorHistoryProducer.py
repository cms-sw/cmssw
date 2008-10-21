from FWCore.ParameterSet.Config import *

process = Process("testGenParticles")

process.include( "FWCore/MessageLogger/data/MessageLogger.cfi" )
process.include( "SimGeneral/HepPDTESSource/data/pythiapdt.cfi")
process.include( "PhysicsTools/HepMCCandAlgos/data/genParticles.cfi")
process.include( "PhysicsTools/HepMCCandAlgos/data/genEventWeight.cfi")
process.include( "PhysicsTools/HepMCCandAlgos/data/genEventScale.cfi")

process.load("PhysicsTools.HepMCCandAlgos.flavorHistoryProducer_cfi")

process.printList = EDAnalyzer( "ParticleListDrawer",
                                src = InputTag( "genParticles" ),
                                maxEventsToPrint = untracked.int32( 100 )
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

process.maxEvents = untracked.PSet( input = untracked.int32(100) )

#from PhysicsTools.HepMCCandAlgos.data.RecoInput_WMRelVal_cfi import *
#from PhysicsTools.HepMCCandAlgos.data.RecoInput_WjetsMadgraph_cfi import *

#process.source = Source("PoolSource",
#                        debugVerbosity = untracked.uint32(200),
#                        debugFlag = untracked.bool(True),
#                        
#                        fileNames = untracked.vstring(
#    'file:/uscms/home/mrenna/scratch/devel/CMSSW_1_8_4/src/GeneratorInterface/AlpgenInterface/test/wbbj__0.root'
#    )
#                        )


#process.source = Source("PoolSource",
#                        debugVerbosity = untracked.uint32(200),
#                        debugFlag = untracked.bool(True),
#                        
#                        fileNames = untracked.vstring(
#    'file:/uscms/home/mrenna/scratch/devel/CMSSW_1_8_4/src/GeneratorInterface/AlpgenInterface/test/tt1j__0.root'
#    )
#                        )


process.source = Source("PoolSource",
                        debugVerbosity = untracked.uint32(200),
                        debugFlag = untracked.bool(True),
                        
                        fileNames = untracked.vstring(
    '/store/relval/2008/6/20/RelVal-RelValWM-1213921089-STARTUP_V1-2nd/0000/12727E22-B83E-DD11-952C-000423D999CA.root'
    )
                        )



#process.source = RecoInput()


#process.source = Source("PoolSource",
#                   debugVerbosity = untracked.uint32(200),
#                   debugFlag = untracked.bool(True),
#                   fileNames = untracked.vstring(
#'/store/relval/2008/6/20/RelVal-RelValWM-1213921089-STARTUP_V1-2nd/0000/12727E22-B83E-DD11-952C-000423D999CA.root',
#'/store/relval/2008/6/20/RelVal-RelValWM-1213921089-STARTUP_V1-2nd/0000/167E5DD9-B73E-DD11-A8BE-001617E30CA4.root',
#'/store/relval/2008/6/20/RelVal-RelValWM-1213921089-STARTUP_V1-2nd/0000/22CD15F1-B93E-DD11-BF1A-0016177CA7A0.root',
#'/store/relval/2008/6/20/RelVal-RelValWM-1213921089-STARTUP_V1-2nd/0000/448FB527-B93E-DD11-9CCB-001617DBD472.root',
#'/store/relval/2008/6/20/RelVal-RelValWM-1213921089-STARTUP_V1-2nd/0000/4A1FDAB8-B83E-DD11-A265-000423D998BA.root',
#'/store/relval/2008/6/20/RelVal-RelValWM-1213921089-STARTUP_V1-2nd/0000/78E156C6-B63E-DD11-8C84-001617C3B5E4.root',
#'/store/relval/2008/6/20/RelVal-RelValWM-1213921089-STARTUP_V1-2nd/0000/7CD75819-BD3E-DD11-A313-001617E30F48.root',
#'/store/relval/2008/6/20/RelVal-RelValWM-1213921089-STARTUP_V1-2nd/0000/864FEACF-BB3E-DD11-99FA-001617C3B66C.root',
#'/store/relval/2008/6/20/RelVal-RelValWM-1213921089-STARTUP_V1-2nd/0000/ACD691FE-BA3E-DD11-82E5-000423D6CA72.root',
#'/store/relval/2008/6/20/RelVal-RelValWM-1213921089-STARTUP_V1-2nd/0000/B4EA03FD-BA3E-DD11-91CF-001617E30F48.root',
#'/store/relval/2008/6/20/RelVal-RelValWM-1213921089-STARTUP_V1-2nd/0000/BA438F3C-B93E-DD11-8DF5-000423D98800.root',
#'/store/relval/2008/6/20/RelVal-RelValWM-1213921089-STARTUP_V1-2nd/0000/CA9CE8D1-B63E-DD11-A51B-001617DBD224.root',
#'/store/relval/2008/6/20/RelVal-RelValWM-1213921089-STARTUP_V1-2nd/0000/CC213510-B93E-DD11-8892-000423D94A04.root',
#'/store/relval/2008/6/20/RelVal-RelValWM-1213921089-STARTUP_V1-2nd/0000/DA568E19-B93E-DD11-9B04-000423D99AAE.root',
#'/store/relval/2008/6/20/RelVal-RelValWM-1213921089-STARTUP_V1-2nd/0000/E8FDE51C-B83E-DD11-944D-000423D99AAE.root',
#'/store/relval/2008/6/20/RelVal-RelValWM-1213921089-STARTUP_V1-2nd/0000/F61FC68A-B73E-DD11-BDE6-000423D9997E.root',
#'/store/relval/2008/6/20/RelVal-RelValWM-1213921089-STARTUP_V1-2nd/0000/F6B6877C-B83E-DD11-ABBE-000423D99AAA.root',
#'/store/relval/2008/6/20/RelVal-RelValWM-1213921089-STARTUP_V1-2nd/0000/FA55E030-B83E-DD11-936E-000423D99CEE.root'
#)
#)

#from PhysicsTools.HepMCCandAlgos.data.h4l_cff import pythiaSource
#process.source = pythiaSource

  
process.out = OutputModule( "PoolOutputModule",
  fileName = untracked.string( "genevents.root" ),
  outputCommands= untracked.vstring(
    "drop *",
    "keep *_genParticles_*_*",
    "keep *_genEventWeight_*_*",
    "keep *_flavorHistoryProducer_*_*"
  )
)
  
process.p = Path( 
  process.genParticles *
  process.genEventWeight *
#  process.printDecay *
  process.printList *
#  process.printTree *
  process.flavorHistoryProducer
)

process.o = EndPath( 
  process.out 
)
