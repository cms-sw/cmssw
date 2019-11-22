import FWCore.ParameterSet.Config as cms

#                                __                 __ 
#    ____  ____   ____   _______/  |______    _____/  |_  ______
#  _/ ___\/  _ \ /    \ /  ___/\   __\__  \  /    \   __\/  ___/
#  \  \__(  <_> )   |  \\___ \  |  |  / __ \|   |  \  |  \___ \
#   \___  >____/|___|  /____  > |__| (____  /___|  /__| /____  >
#        \/           \/     \/            \/     \/          \/
        
PHOTON_CALOIDISO_ET_HIGH_CUT_MIN = 26.
PHOTON_CALOIDISO_ET_LOW_CUT_MIN = 18.
MASS_DIPHOTON_CALOIDISO_CUT_MIN = 70.

PHOTON_R9ID_ET_HIGH_CUT_MIN = 26.
PHOTON_R9ID_ET_LOW_CUT_MIN = 18.
MASS_DIPHOTON_R9ID_CUT_MIN = 70.

MASS_DIPHOTON_MIXEDID_CUT_MIN = 70.

#  _____ __  _____            _         _   _
# |  |  |  ||_   _|   ___ ___| |___ ___| |_|_|___ ___
# |     |  |__| |    |_ -| -_| | -_|  _|  _| | . |   |
# |__|__|_____|_|    |___|___|_|___|___|_| |_|___|_|_|
                                                     
import copy
from HLTrigger.HLTfilters.hltHighLevel_cfi import *
DiPhotonHltFilter = copy.deepcopy(hltHighLevel)
DiPhotonHltFilter.throw = cms.bool(False)
DiPhotonHltFilter.HLTPaths = ["HLT_Photon*_Photon*"]

#  _____     _     _____   _ _____
# |     |___| |___|     |_| |     |___ ___
# |   --| .'| | . |-   -| . |-   -|_ -| . |
# |_____|__,|_|___|_____|___|_____|___|___|
                                          
hltDiPhotonCaloIdIsoObjectProducer = cms.EDProducer("CandidateTriggerObjectProducer",
                                             triggerName = cms.string("HLT_Photon.*_CaloId.*_Iso.*_Photon.*_CaloId.*_Iso.*_.*"),
                                             triggerResults = cms.InputTag("TriggerResults","","HLT"),
                                             triggerEvent   = cms.InputTag("hltTriggerSummaryAOD","","HLT"),
                                             stageL1Trigger = cms.uint32(1)
                                             )
from Configuration.Eras.Modifier_stage2L1Trigger_cff import stage2L1Trigger
stage2L1Trigger.toModify(hltDiPhotonCaloIdIsoObjectProducer, stageL1Trigger = 2)

TrailingPtCaloIdIsoPhotons = cms.EDFilter("CandViewRefSelector",
    filter = cms.bool(True),
    src = cms.InputTag("hltDiPhotonCaloIdIsoObjectProducer"),
    cut = cms.string('pt > '+str(PHOTON_CALOIDISO_ET_LOW_CUT_MIN))
)

LeadingPtCaloIdIsoPhotons = cms.EDFilter("CandViewRefSelector",
    filter = cms.bool(True),
    src = cms.InputTag("hltDiPhotonCaloIdIsoObjectProducer"),
    cut = cms.string('pt > '+str(PHOTON_CALOIDISO_ET_HIGH_CUT_MIN))
)

CaloIdIsoPhotonPairs = cms.EDProducer("CandViewShallowCloneCombiner",
    decay = cms.string("LeadingPtCaloIdIsoPhotons TrailingPtCaloIdIsoPhotons"), # charge coniugate states are implied
    checkCharge = cms.bool(False),                           
    cut   = cms.string("mass > " + str(MASS_DIPHOTON_CALOIDISO_CUT_MIN))
)

CaloIdIsoPhotonPairsCounter = cms.EDFilter("CandViewCountFilter",
                                    src = cms.InputTag("CaloIdIsoPhotonPairs"),
                                    minNumber = cms.uint32(1)
                                    )

CaloIdIsoPhotonPairsFilter = cms.Sequence(DiPhotonHltFilter*hltDiPhotonCaloIdIsoObjectProducer*TrailingPtCaloIdIsoPhotons*LeadingPtCaloIdIsoPhotons*CaloIdIsoPhotonPairs*CaloIdIsoPhotonPairsCounter)

#  _____ ___ _____   _
# | __  | . |     |_| |
# |    -|_  |-   -| . |
# |__|__|___|_____|___|
 
hltDiPhotonR9IdObjectProducer = hltDiPhotonCaloIdIsoObjectProducer.clone(
                                             triggerName = cms.string("HLT_Photon.*_R9Id.*_Photon.*_R9Id.*_.*"),
                                             )

TrailingPtR9IdPhotons = TrailingPtCaloIdIsoPhotons.clone(
    src = cms.InputTag("hltDiPhotonR9IdObjectProducer"),
    cut = cms.string('pt > '+str(PHOTON_R9ID_ET_LOW_CUT_MIN))
)

LeadingPtR9IdPhotons = LeadingPtCaloIdIsoPhotons.clone(
    src = cms.InputTag("hltDiPhotonR9IdObjectProducer"),
    cut = cms.string('pt > '+str(PHOTON_R9ID_ET_HIGH_CUT_MIN))
)

R9IdPhotonPairs = CaloIdIsoPhotonPairs.clone( 
    decay = cms.string("LeadingPtR9IdPhotons TrailingPtR9IdPhotons"), # charge coniugate states are implied
    cut   = cms.string("mass > " + str(MASS_DIPHOTON_R9ID_CUT_MIN))
)

R9IdPhotonPairsCounter = CaloIdIsoPhotonPairsCounter.clone(
                                    src = cms.InputTag("R9IdPhotonPairs"),
                                    )

R9IdPhotonPairsFilter = cms.Sequence(DiPhotonHltFilter*hltDiPhotonR9IdObjectProducer*TrailingPtR9IdPhotons*LeadingPtR9IdPhotons*R9IdPhotonPairs*R9IdPhotonPairsCounter)

#  __   __ _ _   _ ____   _    _____     _     _____   _ _____           _____ ___ _____   _ 
# |  \ /  | | \_/ |  __|_| |  |     |___| |___|     |_| |     |___ ___  | __  | . |     |_| |
# | | v | | |\ _ /|  __| . |  |   --| .'| | . |-   -| . |-   -|_ -| . | |    -|_  |-   -| . |
# |_|   |_|_|_/ \_|____|___|  |_____|__,|_|___|_____|___|_____|___|___| |__|__|___|_____|___|
  

hltDiPhotonMixedCaloR9IdObjectProducer = hltDiPhotonCaloIdIsoObjectProducer.clone(
                                                    triggerName = cms.string("HLT_Photon.*_CaloId.*_Iso.*_Photon.*_R9Id.*_.*"),
                                             )

TrailingPtMixedCaloR9IdPhotons = TrailingPtCaloIdIsoPhotons.clone(
   src = cms.InputTag("hltDiPhotonMixedCaloR9IdObjectProducer"),
   cut = cms.string('pt > '+str(PHOTON_R9ID_ET_LOW_CUT_MIN))
)

LeadingPtMixedCaloR9IdPhotons = LeadingPtCaloIdIsoPhotons.clone(
    src = cms.InputTag("hltDiPhotonMixedCaloR9IdObjectProducer"),
    cut = cms.string('pt > '+str(PHOTON_CALOIDISO_ET_HIGH_CUT_MIN))
)

MixedCaloR9IdPhotonPairs = CaloIdIsoPhotonPairs.clone( 
    decay = cms.string("LeadingPtMixedCaloR9IdPhotons TrailingPtMixedCaloR9IdPhotons"), # charge coniugate states are implied
    cut   = cms.string("mass > " + str(MASS_DIPHOTON_MIXEDID_CUT_MIN))
)

MixedCaloR9IdPhotonPairsCounter = CaloIdIsoPhotonPairsCounter.clone(
                                    src = cms.InputTag("MixedCaloR9IdPhotonPairs"),
                                    )

MixedCaloR9IdPhotonPairsFilter = cms.Sequence(DiPhotonHltFilter*hltDiPhotonMixedCaloR9IdObjectProducer*TrailingPtMixedCaloR9IdPhotons*LeadingPtMixedCaloR9IdPhotons*MixedCaloR9IdPhotonPairs*MixedCaloR9IdPhotonPairsCounter)

#  __   __ _ _   _ ____   _    _____ ___ _____   _   _____     _     _____   _ _____          
# |  \ /  | | \_/ |  __|_| |  | __  | . |     |_| | |     |___| |___|     |_| |     |___ ___  
# | | v | | |\ _ /|  __| . |  |    -|_  |-   -| . | |   --| .'| | . |-   -| . |-   -|_ -| . | 
# |_|   |_|_|_/ \_|____|___|  |__|__|___|_____|___| |_____|__,|_|___|_____|___|_____|___|___| 

hltDiPhotonMixedR9CaloIdObjectProducer = hltDiPhotonCaloIdIsoObjectProducer.clone(
                                             triggerName = cms.string("HLT_Photon.*_R9Id.*_Photon.*_CaloId.*_Iso.*_.*"),
                                             )

TrailingPtMixedR9CaloIdPhotons = TrailingPtCaloIdIsoPhotons.clone(
    src = cms.InputTag("hltDiPhotonMixedR9CaloIdObjectProducer"),
    cut = cms.string('pt > '+str(PHOTON_CALOIDISO_ET_LOW_CUT_MIN))
)

LeadingPtMixedR9CaloIdPhotons = LeadingPtCaloIdIsoPhotons.clone(
    src = cms.InputTag("hltDiPhotonMixedR9CaloIdObjectProducer"),
    cut = cms.string('pt > '+str(PHOTON_R9ID_ET_HIGH_CUT_MIN))
)

MixedR9CaloIdPhotonPairs = CaloIdIsoPhotonPairs.clone( 
    decay = cms.string("LeadingPtMixedR9CaloIdPhotons TrailingPtMixedR9CaloIdPhotons"), # charge coniugate states are implied
    cut   = cms.string("mass > " + str(MASS_DIPHOTON_MIXEDID_CUT_MIN))
)

MixedR9CaloIdPhotonPairsCounter = CaloIdIsoPhotonPairsCounter.clone(
                                    src = cms.InputTag("MixedR9CaloIdPhotonPairs"),
                                    )

MixedR9CaloIdPhotonPairsFilter = cms.Sequence(DiPhotonHltFilter*hltDiPhotonMixedR9CaloIdObjectProducer*TrailingPtMixedR9CaloIdPhotons*LeadingPtMixedR9CaloIdPhotons*MixedR9CaloIdPhotonPairs*MixedR9CaloIdPhotonPairsCounter)
