import FWCore.ParameterSet.Config as cms

#Trigger bit requirement
import HLTrigger.HLTfilters.hltHighLevel_cfi as hlt
ReserveDMu = hlt.hltHighLevel.clone()
ReserveDMu.TriggerResultsTag = cms.InputTag( "TriggerResults", "", "HLT" )
# Read list of paths from the SecondaryDataset Triggerbit tag in the GT
ReserveDMu.eventSetupPathsLabel = 'SecondaryDatasetTrigger' # TriggerBits tag label
ReserveDMu.eventSetupPathsKey = 'ReserveDMu'                # Dataset-specific key
ReserveDMu.andOr = cms.bool( True )
# we want to intentionally throw and exception
# in case it does not match one of the HLT Paths
# set to False now, switch to on once matrix is updated
ReserveDMu.throw = cms.bool( True )
