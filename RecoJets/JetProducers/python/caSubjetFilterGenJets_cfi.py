################################################################################
# see: https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSubjetFilterJetProducer
################################################################################

import FWCore.ParameterSet.Config as cms

from RecoJets.JetProducers.GenJetParameters_cfi import *
from RecoJets.JetProducers.AnomalousCellParameters_cfi import *

caSubjetFilterGenJets = cms.EDProducer(
    "SubjetFilterJetProducer",
    GenJetParameters,
    AnomalousCellParameters,
    jetAlgorithm = cms.string("CambridgeAachen"),
    nFatMax      = cms.uint32(0),
    rParam       = cms.double(1.2),
    rFilt        = cms.double(0.3),
    massDropCut  = cms.double(0.67),
    asymmCut     = cms.double(0.3),
    asymmCutLater= cms.bool(True)
    )
caSubjetFilterGenJets.doAreaFastjet= cms.bool(True)
