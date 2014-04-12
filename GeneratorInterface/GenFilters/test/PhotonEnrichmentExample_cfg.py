import FWCore.ParameterSet.Config as cms

process = cms.Process("GenFilter")

## Load Gen Filter
process.load("GeneratorInterface.GenFilters.PhotonEnrichmentFilter_cfi")

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(200) )
process.MessageLogger.cerr.FwkReport.reportEvery = 100
process.options = cms.untracked.PSet(wantSummary = cms.untracked.bool(True))

process.source = cms.Source("PoolSource",
							fileNames = cms.untracked.vstring(
        '/store/relval/CMSSW_3_11_0_pre3/RelValQCD_FlatPt_15_3000/GEN-SIM-RECO/MC_310_V3-v1/0059/9EF1C99B-771A-E011-9D7C-002618943908.root',
        '/store/relval/CMSSW_3_11_0_pre3/RelValQCD_FlatPt_15_3000/GEN-SIM-RECO/MC_310_V3-v1/0059/8E8672F8-AB1A-E011-BA71-002618943810.root',
        '/store/relval/CMSSW_3_11_0_pre3/RelValQCD_FlatPt_15_3000/GEN-SIM-RECO/MC_310_V3-v1/0058/DAD54B79-751A-E011-B557-00261894393A.root',
        '/store/relval/CMSSW_3_11_0_pre3/RelValQCD_FlatPt_15_3000/GEN-SIM-RECO/MC_310_V3-v1/0058/84C6B8FF-691A-E011-BA85-002354EF3BDD.root',
        '/store/relval/CMSSW_3_11_0_pre3/RelValQCD_FlatPt_15_3000/GEN-SIM-RECO/MC_310_V3-v1/0058/7CB92E70-6B1A-E011-8193-002618943875.root',
        '/store/relval/CMSSW_3_11_0_pre3/RelValQCD_FlatPt_15_3000/GEN-SIM-RECO/MC_310_V3-v1/0058/1E00347B-6D1A-E011-9B19-00248C55CC4D.root'
	)
)

# Output definition
process.output = cms.OutputModule("PoolOutputModule",
								  #fastCloning = cms.untracked.bool(False),
								  outputCommands = process.RECOSIMEventContent.outputCommands,
								  fileName = cms.untracked.string('PhotonEnrichedSample.root')
								  )

process.p = cms.Path(process.PhotonEnrichmentFilter)
process.outpath = cms.EndPath(process.output)
