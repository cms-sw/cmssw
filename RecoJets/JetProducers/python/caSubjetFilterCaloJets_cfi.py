################################################################################
# see: https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideSubjetFilterJetProducer
################################################################################

import FWCore.ParameterSet.Config as cms

from RecoJets.JetProducers.CaloJetParameters_cfi import *
from RecoJets.JetProducers.AnomalousCellParameters_cfi import *

caSubjetFilterCaloJets = cms.EDProducer(
    "SubjetFilterJetProducer",
    CaloJetParameters,
    AnomalousCellParameters,
    jetAlgorithm = cms.string("CambridgeAachen"),
    nFatMax      = cms.uint32(0),
    rParam       = cms.double(1.2),
    rFilt        = cms.double(0.3),
    massDropCut  = cms.double(0.67),
    asymmCut     = cms.double(0.3),
    asymmCutLater= cms.bool(True)
    )
caSubjetFilterCaloJets.doAreaFastjet = cms.bool(True)
