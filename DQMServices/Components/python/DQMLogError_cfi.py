
import FWCore.ParameterSet.Config as cms

logErrorDQM = cms.EDAnalyzer( "DQMLogError",
              Categories = cms.vstring (
                      'PFTrackTransformer', 
		      'RPCHitAssociator', 
		      'GaussianSumUtilities'
                  ),
              Directory = cms.string("LogMsg")
              )


