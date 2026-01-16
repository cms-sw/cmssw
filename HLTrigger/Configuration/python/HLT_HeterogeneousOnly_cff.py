import FWCore.ParameterSet.Config as cms

from .HLT_75e33_cff import fragment

for p in dir(fragment):
    att = getattr(fragment, p)
    if isinstance(att, cms.Path) and p not in [ "HLTriggerFinalPath", "HLTAnalyzerEndpath"]:
        delattr(fragment, p)
    del att

fragment.load("HLTrigger/Configuration/HLT_75e33/paths/DST_HeterogeneousReco_cfi")    
fragment.schedule = cms.Schedule(*[
    fragment.DST_HeterogeneousReco,
    fragment.HLTriggerFinalPath,
    fragment.HLTAnalyzerEndpath,
])
