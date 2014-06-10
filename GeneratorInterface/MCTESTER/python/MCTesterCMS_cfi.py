import FWCore.ParameterSet.Config as cms

MCTesterCMS = cms.EDAnalyzer( "MCTesterCMS" ,
    hepmcCollection = cms.InputTag("generator","")
    )
