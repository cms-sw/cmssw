import FWCore.ParameterSet.Config as cms

from Configuration.EventContent.EventContent_cff import AODSIMEventContent
EXODisappTrkSkimContent = AODSIMEventContent.clone()

EXODisappTrkSkimContent.outputCommands.append('drop *')
#EXODisappTrkSkimContent.outputCommands.append('keep *_ecalRecHit_EcalRecHitsEB_*')
#EXODisappTrkSkimContent.outputCommands.append('keep *_ecalRecHit_EcalRecHitsEE_*')
#EXODisappTrkSkimContent.outputCommands.append('keep *_hbhereco_*_*')
EXODisappTrkSkimContent.outputCommands.append('keep *_reducedHcalRecHits_*_*')
EXODisappTrkSkimContent.outputCommands.append('keep *_reducedEcalRecHits*_*_*')
#EXODisappTrkSkimContent.outputCommands.append('keep *_dedxHitInfo_*_*')
#EXODisappTrkSkimContent.outputCommands.append('keep *_dedx*Harmonic2_*_*')
#EXODisappTrkSkimContent.outputCommands.append('keep *_generalTracks_*_*')

import copy
from HLTrigger.HLTfilters.hltHighLevel_cfi import *

hltDisappTrk = copy.deepcopy(hltHighLevel)
hltDisappTrk.throw = cms.bool(False)

hltDisappTrk.HLTPaths = [

    #2017 and 2018
    #MET

    #"HLT_MET105_IsoTrk50_v*",
    #"HLT_PFMET120_PFMHT120_IDTight_v*",
    #"HLT_PFMET130_PFMHT130_IDTight_v*",
    #"HLT_PFMET140_PFMHT140_IDTight_v*", 
    #"HLT_PFMETNoMu120_PFMHTNoMu120_IDTight_v*", 
    #"HLT_PFMETNoMu130_PFMHTNoMu130_IDTight_v*",
    #"HLT_PFMETNoMu140_PFMHTNoMu140_IDTight_v*",
    #"HLT_PFMET250_HBHECleaned_v*",
    #"HLT_PFMET300_HBHECleaned_v*", 
    #"HLT_PFMETTypePne140_PFMHT140_IDTight_v*",
    #"HLT_PFMET200_HBHE_BeamHaloCleaned_v*",
    #"HLT_PFMetTypeOne200_HBHE_BEAMHaloCleaned_v*",
    #EGamma
    #"HLT_Ele35_WPTight_Gsf_v*", 
    #"HLT_Ele32_WPTight_Gsf_v*", 
    #SingleMuon
    #"HLT_IsoMu27_v*", 
    #"HLT_IsoMu24_v*", 
    #Tau
    #"HLT_MediumChargedIsoPFTau50_Trk30_eta2p1_1pr_v*",

    #2016
    #"HLT_Photon175_v*",
    #"HLT_DoublePhoton60_v*",
    #"HLT_PFMET300_v*",
    #"HLT_PFMET170_HBHE_BeamHaloCleaned_v*",
    #2017 and 2018
    #"HLT_Photon200_v*",
    #"HLT_Photon300_NoHE_v*",
    #"HLT_DoublePhoton70_v*",
    #"HLT_PFMET140_PFMHT140_IDTight_v*",
    #"HLT_PFMET250_HBHECleaned_v*",
    #"HLT_PFMET300_HBHECleaned_v*",

    #test
    #"HLT_Mu20_Mu10_v*",
    #"HLT_Random_v*",
    #"HLT_ZeroBias_v*,"
    #"MC_CaloMHT_v*",
    "MC_PFMET_v17"
]

hltDisappTrk.throw = False
hltDisappTrk.andOr = True

disappTrkSelection=cms.EDFilter("TrackSelector", 
    src = cms.InputTag("generalTracks"),
    cut = cms.string('pt > 25 && abs(eta()) < 2.1'),
    filter = cms.bool(True)
)

# disappTrk skim sequence
EXODisappTrkSkimSequence = cms.Sequence(
    hltDisappTrk * disappTrkSelection
    )
