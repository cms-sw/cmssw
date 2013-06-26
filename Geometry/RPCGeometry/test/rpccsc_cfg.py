import FWCore.ParameterSet.Config as cms

process = cms.Process("Demo")

process.load("FWCore.MessageService.MessageLogger_cfi")
process.load("Geometry.MuonCommonData.muonIdealGeometryXML_cfi")
process.load("Geometry.RPCGeometry.rpcGeometry_cfi")
process.load("Geometry.CSCGeometry.cscGeometry_cfi")
process.load("Geometry.DTGeometry.dtGeometry_cfi")
process.load("Geometry.MuonNumbering.muonNumberingInitialization_cfi")
process.load("DQMServices.Components.MEtoEDMConverter_cfi")

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
#process.GlobalTag.connect = "frontier://PromptProd/CMS_COND_21X_GLOBALTAG"
#process.GlobalTag.globaltag = "CRUZET4_V5P::All"
#process.GlobalTag.globaltag = "IDEAL_V9::All"
#process.GlobalTag.globaltag = "COSMMC_21X::All"
#process.GlobalTag.globaltag = "CRAFT_ALL_V4::All"


process.GlobalTag.globaltag = 'MC_31X_V1::All'

process.prefer("GlobalTag")


process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(1) )

process.source = cms.Source("EmptySource"
    # replace 'myfile.root' with the source file you want to use
)

process.demo = cms.EDAnalyzer('RPCCSC'
)


process.p = cms.Path(process.demo)
