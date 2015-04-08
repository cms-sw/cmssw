import FWCore.ParameterSet.VarParsing as VarParsing

ivars = VarParsing.VarParsing('standard')
ivars.files = ''
ivars.output = 'HIPAT_output_full13.root'

ivars.register ('randomNumber',
                                mult=ivars.multiplicity.singleton,
                                info="for testing")

ivars.register ('initialEvent',
                                mult=ivars.multiplicity.singleton,
                                info="for testing")

ivars.randomNumber=13
ivars.initialEvent=1
ivars.parseArguments()

import FWCore.ParameterSet.Config as cms

process = cms.Process("GEN")

process.load('Configuration.StandardSequences.GeometryDB_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
process.GlobalTag.globaltag = 'STARTHI53_LV1::All'

process.load('Configuration/StandardSequences/Services_cff')
process.load('Configuration/StandardSequences/Generator_cff')
process.load('IOMC.EventVertexGenerators.VtxSmearedRealistic7TeV2011Collision_cfi')


process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")
process.load('Configuration.GenProduction.HI.PyquenTUNE_Dijet100_NN_Quenched_TuneD6T_2760GeV_cfi')
process.generator = process.hiSignal.clone(embeddingMode = False,
                                           cFlag = 1,
                                           bMin = 0,
                                           bMax = 15
                                           )

process.source = cms.Source('EmptySource')


process.maxEvents = cms.untracked.PSet(
        input = cms.untracked.int32(100)
        )

process.RandomNumberGeneratorService.generator.initialSeed = ivars.randomNumber

process.TFileService = cms.Service('TFileService',
                                   fileName = cms.string(ivars.output)
                                   )


#process.load('RecoHI.HiJetAlgos.HiGenJets_cff')
process.load('HeavyIonsAnalysis.JetAnalysis.HiGenAnalyzer_cfi')
process.load('HeavyIonsAnalysis.JetAnalysis.jets.HiGenJetAnalyzers_cff')
process.load('HeavyIonsAnalysis.JetAnalysis.PatAna_cff')

process.p = cms.Path(process.generator
                     *process.VtxSmeared
                     *process.hiGenParticles
                     *process.hiGenParticlesForJets
                     *process.ak3HiGenJets
                     *process.ak4HiGenJets
                     *process.ak5HiGenJets
                     *process.ak3GenJetAnalyzer
                     *process.ak4GenJetAnalyzer
                     *process.ak5GenJetAnalyzer                     
                     *process.heavyIon
                     *process.HiGenParticleAna
                     )


