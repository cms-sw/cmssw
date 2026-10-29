import FWCore.ParameterSet.Config as cms

############################################################
# L1 Global Trigger Emulation
############################################################

# Conditions

from L1Trigger.Phase2L1GT.l1tGTProducer_cff import l1tGTProducer

from L1Trigger.Phase2L1GT.l1tGTSingleObjectCond_cfi import l1tGTSingleObjectCond
from L1Trigger.Phase2L1GT.l1tGTDoubleObjectCond_cfi import l1tGTDoubleObjectCond
from L1Trigger.Phase2L1GT.l1tGTTripleObjectCond_cfi import l1tGTTripleObjectCond
from L1Trigger.Phase2L1GT.l1tGTQuadObjectCond_cfi import l1tGTQuadObjectCond

from L1Trigger.Phase2L1GT.l1tGTAlgoBlockProducer_cff import algorithms

from L1Trigger.Configuration.Phase2GTMenus.SeedDefinitions.step1_2024.l1tGTObject_constants import *
from L1Trigger.Configuration.Phase2GTMenus.SeedDefinitions.step1_2024.l1tGTMenuObjects_cff import *

# Step 2 seeds that P2GT can already run, from Phase2-L1MenuTools
# configs/V50nano/rate_table/step1and2_menu_cfg.yml. Not implemented: L1_PFHTT_QuadJet_BTagNNScore
# (no SC4NG jets or b-tag scores in P2GT) and L1_W3pi_GTT (no track triplets in P2GT).

####### SC8 JET SEEDS ###########
# L1_SinglePfJet8_Mass and L1_DoublePFJet8_dEtaMax_Mass are not implemented:
# the P2GT jet word carries no jet mass.

SinglePfJet8 = l1tGTSingleObjectCond.clone(
    l1tGTsc8Jet_er5.clone(),
    regionsMinPt = get_object_thrs(230, "CL2JetsSC8", "default"),
)
pSinglePfJet8 = cms.Path(SinglePfJet8)
algorithms.append(cms.PSet(expression = cms.string("pSinglePfJet8")))

DoublePFJet8dEtaMax = l1tGTDoubleObjectCond.clone(
    collection1 = l1tGTsc8Jet_er5.clone(
        regionsMinPt = get_object_thrs(200, "CL2JetsSC8", "default"),
    ),
    collection2 = l1tGTsc8Jet_er5.clone(
        regionsMinPt = get_object_thrs(200, "CL2JetsSC8", "default"),
    ),
    maxDEta = cms.double(1.6),
)
pDoublePFJet8_dEtaMax = cms.Path(DoublePFJet8dEtaMax)
algorithms.append(cms.PSet(expression = cms.string("pDoublePFJet8_dEtaMax")))

####### DISPLACED MUON SEEDS ###########
# GMT standalone displaced muons have no online-to-offline scalings: online thresholds.

# Quality >= 15 in the endcap: the 4-bit quality word must have all bits set
SingleDispMu = l1tGTSingleObjectCond.clone(
    l1tGTsaDisplacedMuon.clone(
        minEta = cms.double(-2.0),
        maxEta = cms.double(2.0),
        regionsAbsEtaLowerBounds = cms.vdouble(0, 0.83, 1.24),
        regionsQualityFlags = cms.vuint32(0, 0, 0b1111),
    ),
    minPt = cms.double(22),
)
pSingleDispMu = cms.Path(SingleDispMu)
algorithms.append(cms.PSet(expression = cms.string("pSingleDispMu")))

DoubleDispMu = l1tGTDoubleObjectCond.clone(
    collection1 = l1tGTsaDisplacedMuon.clone(
        minPt = cms.double(15),
    ),
    collection2 = l1tGTsaDisplacedMuon.clone(
        minPt = cms.double(7),
    ),
)
pDoubleDispMu = cms.Path(DoubleDispMu)
algorithms.append(cms.PSet(expression = cms.string("pDoubleDispMu")))
