import FWCore.ParameterSet.Config as cms

process = cms.Process("EWKHighPtDiMuonsSkim")

process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.options   = cms.untracked.PSet( wantSummary = cms.untracked.bool(True) )

process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")

# source
process.source = cms.Source("PoolSource", 
     fileNames = cms.untracked.vstring(
##"file:/data1/degrutto/CMSSW_3_6_1_patch2/src/ElectroWeakAnalysis/Skimming/test/138/testZMuMuSubskim138_737_748.root",
#"file:/data1/degrutto/CMSSW_3_6_1_patch4/src/ElectroWeakAnalysis/Skimming/test/139/testZMuMuSubskim_362_365.root",
#"file:/data1/degrutto/CMSSW_3_6_1_patch4/src/ElectroWeakAnalysis/Skimming/test/139/testZMuMuSubskim_368_370.root",
#"file:/data1/degrutto/CMSSW_3_6_1_patch4/src/ElectroWeakAnalysis/Skimming/test/139/testZMuMuSubskim139_195_239.root",
#"file:/data1/degrutto/CMSSW_3_6_1_patch4/src/ElectroWeakAnalysis/Skimming/test/139/testZMuMuSubskim_356_360.root",

"file:/data1/degrutto/CMSSW_3_6_1_patch4/src/ElectroWeakAnalysis/Skimming/test/139/testZMuMuSubskim_372_375.root",
"file:/data1/degrutto/CMSSW_3_6_1_patch4/src/ElectroWeakAnalysis/Skimming/test/139/testZMuMuSubskim139_347.root",

"file:/data1/degrutto/CMSSW_3_6_1_patch4/src/ElectroWeakAnalysis/Skimming/test/139/testZMuMuSubskim_399_411.root",


"file:/data1/degrutto/CMSSW_3_6_1_patch4/src/ElectroWeakAnalysis/Skimming/test/139/testZMuMuSubskim_457_459.root",



    )
)
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(1000) )

process.load("Configuration.StandardSequences.Geometry_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = cms.string('GR10_P_V6::All')
process.load("Configuration.StandardSequences.MagneticField_cff")

# Muon filter
process.goodMuons = cms.EDFilter("MuonSelector",
  src = cms.InputTag("muons"),
  cut = cms.string('pt > 20.0 && ( isGlobalMuon=1 || isTrackerMuon=1) && isolationR03().sumPt<4.0'),
  filter = cms.bool(True)
)

# dxy filter on good muons
process.dxyFilteredMuons = cms.EDFilter("MuonSelector",
  src = cms.InputTag("goodMuons"),
  cut = cms.string('abs(innerTrack().dxy)<1.0'),
  filter = cms.bool(True)                                
)


process.dimuons = cms.EDFilter("CandViewShallowCloneCombiner",
                                    checkCharge = cms.bool(True),
                                    cut = cms.string('mass > 60 & charge=0'),
                                    decay = cms.string("goodMuons@+ goodMuons@-")
                                )


# dimuon filter
process.dimuonsFilter = cms.EDFilter("CandViewCountFilter",
                                 src = cms.InputTag("dimuons"),
                                 minNumber = cms.uint32(1)
                             )




# Skim path
process.EWK_HighPtDiMuonSkimPath = cms.Path(
  process.goodMuons *
  process.dxyFilteredMuons *
  process.dimuons *
  process.dimuonsFilter
)

# Output module configuration
from Configuration.EventContent.EventContent_cff import *
EWK_MuSkimEventContent = cms.PSet(
    outputCommands = cms.untracked.vstring()
)
EWK_MuSkimEventContent.outputCommands.extend(RECOEventContent.outputCommands)

EWK_MuSkimEventSelection = cms.PSet(
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring(
           'EWK_HighPtDiMuonSkimPath')
    )
)

process.EWK_MuSkimOutputModule = cms.OutputModule("PoolOutputModule",
    EWK_MuSkimEventContent,
    EWK_MuSkimEventSelection,
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string('EWKHighPtDiMuonSkim'),
        dataTier = cms.untracked.string('USER')
   ),
   fileName = cms.untracked.string('EWK_HighPtDiMuonSkim_SD_Mu_139_372_459.root')
)

process.outpath = cms.EndPath(process.EWK_MuSkimOutputModule)


