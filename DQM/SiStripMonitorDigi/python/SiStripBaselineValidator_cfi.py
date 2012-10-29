import FWCore.ParameterSet.Config as cms

SiStripBaselineValidator = cms.EDAnalyzer("SiStripBaselineValidator",
	outputFile = cms.string("test.root"),
        srcProcessedRawDigi =  cms.InputTag('siStripZeroSuppression','VirginRaw'),

					  	
                                          )

