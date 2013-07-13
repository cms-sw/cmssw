import FWCore.ParameterSet.Config as cms

process = cms.Process("OWNPARTICLES")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")#https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideFrontierConditions#Global_Tags_for_Monte_Carlo_Prod
process.load("FWCore.MessageService.MessageLogger_cfi")
process.load("GeneratorInterface.ExternalDecays.TauSpinner_cfi")

process.MessageLogger.cerr = cms.untracked.PSet(
    threshold = cms.untracked.string('INFO'),
    FwkReport = cms.untracked.PSet(limit = cms.untracked.int32(0)),
    DEBUG = cms.untracked.PSet(limit = cms.untracked.int32(-1))
    )

process.GlobalTag.globaltag = 'MC_50_V13::All'

numberOfEvents = 100

process.source = cms.Source("PoolSource",fileNames = cms.untracked.vstring('file:Hadronizer_Et20ExclTuneZ2_7TeV_alpgen_tauola_cff_py_GEN_SIM_DIGI_L1_DIGI2RAW_RAW2DIGI_RECO.root'))

process.debugOutput = cms.OutputModule("PoolOutputModule",
                                       outputCommands = cms.untracked.vstring('keep *'),
                                       fileName = cms.untracked.string('TauSpinerRecoTest.root'),
                                       )
process.out_step = cms.EndPath(process.debugOutput)

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(numberOfEvents) )
process.p1 = cms.Path( process.TauSpinnerReco )
process.schedule = cms.Schedule(process.p1)
process.schedule.append(process.out_step)
