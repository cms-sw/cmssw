import FWCore.ParameterSet.Config as cms

process = cms.Process("ANA")

process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")
process.load("Configuration.StandardSequences.Services_cff")
process.load("GeneratorInterface.HydjetInterface.hydjetDefault_cfi")

process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(10000)
                                       )

process.source = cms.Source("EmptySource")

process.generator = cms.EDFilter("HijingGeneratorFilter",
                                     frame = cms.string('CMS     '),
                                     targ = cms.string('P       '),
                                     izp = cms.int32(82),
                                     bMin = cms.double(0),
                                     izt = cms.int32(1),
                                     proj = cms.string('A       '),
                                     comEnergy = cms.double(5020.0),
                                     iat = cms.int32(1),
                                     bMax = cms.double(15),
                                     iap = cms.int32(208),
                                     rotateEventPlane = cms.bool(True)
                                 )


process.RandomNumberGeneratorService.generator.initialSeed = 6

process.SimpleMemoryCheck = cms.Service('SimpleMemoryCheck',
                                        ignoreTotal=cms.untracked.int32(0),
                                        oncePerEventMode = cms.untracked.bool(False)
                                        )

process.ana = cms.EDAnalyzer('HydjetAnalyzer'
                             )

process.TFileService = cms.Service('TFileService',
                                   fileName = cms.string('treefileR6.root')
                                   )

process.p1 = cms.Path(process.generator)
process.p2 = cms.Path(process.ana)




