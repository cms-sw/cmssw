import FWCore.ParameterSet.Config as cms
import HLTrigger.HLTfilters.hltHighLevel_cfi as hlt

ReserveDMu = hlt.hltHighLevel.clone(
   TriggerResultsTag = ("TriggerResults", "", "HLT" ),
   eventSetupPathsLabel = 'SecondaryDatasetTrigger',
   eventSetupPathsKey = 'ReserveDMu',
   andOr = True,
   throw = False,
)
