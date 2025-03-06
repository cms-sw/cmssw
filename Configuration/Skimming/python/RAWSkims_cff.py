import FWCore.ParameterSet.Config as cms
import HLTrigger.HLTfilters.hltHighLevel_cfi as hlt

ReserveDMu = hlt.hltHighLevel.clone()
ReserveDMu.TriggerResultsTag = cms.InputTag( "TriggerResults", "", "HLT" )
ReserveDMu.eventSetupPathsLabel = cms.string('SecondaryDatasetTrigger')
ReserveDMu.eventSetupPathsKey = cms.string('ReserveDMu')
ReserveDMu.andOr = cms.bool( True )
ReserveDMu.throw = cms.bool( False )
