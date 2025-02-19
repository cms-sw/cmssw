import FWCore.ParameterSet.Config as cms

l1GtTriggerMenuTester = cms.EDAnalyzer( "L1GtTriggerMenuTester",  
                      OverwriteHtmlFile = cms.bool(False),
                      HtmlFile = cms.string(""),                               
                      UseHltMenu = cms.bool(False),                   
                      HltProcessName = cms.string("L1GtTriggerMenuTester"),
                      NoThrowIncompatibleMenu = cms.bool(True),  
                      PrintPfsRates = cms.bool(False),  
                      IndexPfSet = cms.int32(0)                                       
                    )
