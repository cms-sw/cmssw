
import FWCore.ParameterSet.Config as cms

preIdAnalyzer = cms.EDAnalyzer("PreIdAnalyzer",
                               PreIdMap=cms.InputTag("trackerDrivenElectronSeeds:preid"),
                               TrackCollection=cms.InputTag("generalTracks"),                               
                               )
# foo bar baz
# DkEZp2oBv4u1I
