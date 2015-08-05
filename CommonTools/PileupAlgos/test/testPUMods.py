import FWCore.ParameterSet.Config as cms

process = cms.Process('TestPUMods')
process.load('Configuration/StandardSequences/Services_cff')
process.load('FWCore/MessageService/MessageLogger_cfi')
process.load('Configuration/StandardSequences/FrontierConditions_GlobalTag_cff')
process.MessageLogger.cerr.FwkReport.reportEvery = 10
process.GlobalTag.globaltag = 'START53_V7G::All'

process.load('CommonTools/PileupAlgos/Puppi_cff')
process.load('CommonTools/PileupAlgos/softKiller_cfi')

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(2) )
process.source = cms.Source("PoolSource",
	fileNames  = cms.untracked.vstring(
		# '/store/mc/RunIISpring15DR74/QCD_Pt-15to7000_TuneCUETP8M1_Flat_13TeV_pythia8/GEN-SIM-RECO/AsymptFlat0to50bx25Reco_MCRUN2_74_V9-v3/10000/0009D30B-0207-E511-B581-0026182FD753.root'
		'/store/relval/CMSSW_7_4_1/RelValZMM_13/GEN-SIM-RECO/PU50ns_MCRUN2_74_V8_gensim_740pre7-v1/00000/32FD5AA2-41EC-E411-94B1-0025905B8572.root'
		)
)
process.source.inputCommands = cms.untracked.vstring("keep *",
                                                     "drop *_MEtoEDMConverter_*_*")

process.options = cms.untracked.PSet(
  wantSummary = cms.untracked.bool(True),
  Rethrow     = cms.untracked.vstring('ProductNotFound'),
  fileMode    = cms.untracked.string('NOMERGE')
)


process.puSequence = cms.Sequence(process.puppi)
process.p = cms.Path(process.puSequence)
process.output = cms.OutputModule("PoolOutputModule",
                                  outputCommands = cms.untracked.vstring('drop *',
                                                                         'keep *_particleFlow_*_*',
                                                                         'keep *_*_*_TestPUMods'),
                                  fileName       = cms.untracked.string ("Output.root")
)
# schedule definition                                                                                                       
process.outpath  = cms.EndPath(process.output) 
