import FWCore.ParameterSet.Config as cms

from .HLT_75e33_cff import fragment


for p in dir(fragment):
    att = getattr(fragment, p)
    if isinstance(att, cms.Path) and p not in ["MC_TRK", "HLTriggerFinalPath", "HLTAnalyzerEndpath"]:
        delattr(fragment, p)
    del att

fragment.schedule = cms.Schedule(*[
    fragment.MC_TRK,
    fragment.HLTriggerFinalPath,
    fragment.HLTAnalyzerEndpath,
])
