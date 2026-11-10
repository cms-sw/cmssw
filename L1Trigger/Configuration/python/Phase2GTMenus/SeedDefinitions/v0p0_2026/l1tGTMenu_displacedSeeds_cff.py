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
from L1Trigger.Phase2L1GT.l1tGTLUTs_cfi import *

from L1Trigger.Phase2L1GT.l1tGTAlgoBlockProducer_cff import algorithms

from L1Trigger.Configuration.Phase2GTMenus.SeedDefinitions.v0p0_2026.l1tGTObject_constants import *
from L1Trigger.Configuration.Phase2GTMenus.SeedDefinitions.v0p0_2026.l1tGTMenuObjects_cff import *

####### DISPLACED MUON SEEDS ###########
# GMT standalone displaced muons have no online-to-offline scalings: online thresholds.
# Their pT is coarsely binned, so the thresholds sit one pT LSB below the value to select pT >= value.
from L1Trigger.Phase2L1GT.l1tGTScales import scale_parameter
_ptLSB = scale_parameter.pT_lsb.value()

SingleDispMuBMTF = l1tGTSingleObjectCond.clone(
    l1tGTsaDisplacedMuon.clone(
        minEta = cms.double(-0.83),
        maxEta = cms.double(0.83),
    ),
    minPt = cms.double(15 - _ptLSB),
)
pSingleDispMu_BMTF = cms.Path(SingleDispMuBMTF)
algorithms.append(cms.PSet(expression = cms.string("pSingleDispMu_BMTF")))

DoubleDispMu = l1tGTDoubleObjectCond.clone(
    collection1 = l1tGTsaDisplacedMuon.clone(
        minEta = cms.double(-2.0),
        maxEta = cms.double(2.0),
        minPt = cms.double(6 - _ptLSB),
    ),
    collection2 = l1tGTsaDisplacedMuon.clone(
        minEta = cms.double(-2.0),
        maxEta = cms.double(2.0),
        minPt = cms.double(6 - _ptLSB),
    ),
)
pDoubleDispMu = cms.Path(DoubleDispMu)
algorithms.append(cms.PSet(expression = cms.string("pDoubleDispMu")))
