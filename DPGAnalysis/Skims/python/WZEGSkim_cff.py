import FWCore.ParameterSet.Config as cms

hltFilter = cms.EDFilter("HLTHighLevel",
                                 TriggerResultsTag = cms.InputTag("TriggerResults","","HLT"),
                                 HLTPaths = cms.vstring(
    #    "HLT_Photon15_L1R",
    #    "HLT_Photon15_Cleaned_L1R",
    #    "HLT_Photon20_Cleaned_L1R",
    "HLT_Ele15_LW_L1R",
    "HLT_Ele15_SW_L1R",
    "HLT_Ele15_SW_CaloEleId_L1R",
    "HLT_Ele17_SW_CaloEleId_L1R",
    "HLT_Ele17_SW_L1R",
    "HLT_Ele17_SW_TightEleId_L1R",
    "HLT_Ele17_SW_TightCaloEleId_SC8HE_L1R"
    ),
                         eventSetupPathsKey = cms.string(''),
                         andOr = cms.bool(True),
                         throw = cms.bool(False),
                         saveTags = cms.bool(False)
                         )

from DPGAnalysis.Skims.WZinterestingEventFilter_cfi import *

WZfilterSkim = cms.Sequence(WZInterestingEventSelector)

