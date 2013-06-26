import FWCore.ParameterSet.Config as cms

pfClient = cms.EDAnalyzer("PFClient",
    FolderNames = cms.vstring("PFJet/CompWithGenJet","PFJet/CompWithCaloJet"),
    HistogramNames = cms.vstring( "delta_et_Over_et_VS_et_"),
    CreateEfficiencyPlots = cms.bool(False),
    HistogramNamesForEfficiencyPlots = cms.vstring( " " ) 
)
