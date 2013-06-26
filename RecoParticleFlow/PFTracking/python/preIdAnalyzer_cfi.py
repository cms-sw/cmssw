
import FWCore.ParameterSet.Config as cms

preIdAnalyzer = cms.EDAnalyzer("PreIdAnalyzer",
                               PreIdMap=cms.InputTag("trackerDrivenElectronSeeds:preid"),
                               TrackCollection=cms.InputTag("generalTracks"),                               
                               )
