import FWCore.ParameterSet.VarParsing as VarParsing

ivars = VarParsing.VarParsing('standard')
ivars.files = 'dcache:/pnfs/cmsaf.mit.edu/t2bat/cms/store/user/yetkin/sim/CMSSW_3_3_5/Pythia_MinBias_D6T_900GeV_d20091208/Vertex1207/Pythia_MinBias_D6T_900GeV_d20091208_000005.root'

ivars.output = 'bambu.root'
ivars.maxEvents = -1

ivars.parseArguments()

import FWCore.ParameterSet.Config as cms

process = cms.Process('DUMMY')

process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
process.GlobalTag.globaltag = 'MC_38Y_V13::All'

process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(13))
process.source = cms.Source("EmptyIOVSource",
                            timetype = cms.string("runnumber"),
                            firstValue = cms.uint64(1),
                            lastValue = cms.uint64(3),
                            interval = cms.uint64(3)
                            )

process.TFileService = cms.Service('TFileService',
                                   fileName = cms.string('centralityfile.root')
                                   )

process.HeavyIonGlobalParameters = cms.PSet(
       centralityVariable = cms.string("HFhits"),
          nonDefaultGlauberModel = cms.string("AMPT_2760GeV"),
          centralitySrc = cms.InputTag("hiCentrality")
          )

process.makeCentralityTableTFile = cms.EDAnalyzer('CentralityTableProducer',
                                                  isMC = cms.untracked.bool(True),
                                                  makeDBFromTFile = cms.untracked.bool(False),
                                                  makeTFileFromDB = cms.untracked.bool(True)
                                                  )

process.step  = cms.Path(process.makeCentralityTableTFile)
    




