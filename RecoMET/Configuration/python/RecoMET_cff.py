import FWCore.ParameterSet.Config as cms

##____________________________________________________________________________||
from RecoJets.Configuration.CaloTowersES_cfi import *
from RecoMET.METProducers.CaloMET_cfi import *
from RecoMET.METProducers.hcalnoiseinfoproducer_cfi import *
from RecoMET.METProducers.MuonMETValueMapProducer_cff import *
from RecoMET.METProducers.caloMetM_cfi import *
from RecoMET.Configuration.RecoMET_BeamHaloId_cff import *

##____________________________________________________________________________||
metreco = cms.Sequence(
        caloMet+
        caloMetBE+
        caloMetBEFO+
        muonMETValueMapProducer+
        caloMetM +
        BeamHaloId
        )

##____________________________________________________________________________||
metrecoPlusHCALNoise = cms.Sequence( metreco + hcalnoise )

from Configuration.Eras.Modifier_phase2_hcal_cff import phase2_hcal
phase2_hcal.toReplaceWith( metrecoPlusHCALNoise, metreco )

##____________________________________________________________________________||
