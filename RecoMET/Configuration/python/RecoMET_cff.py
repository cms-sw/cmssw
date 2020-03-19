import FWCore.ParameterSet.Config as cms

##____________________________________________________________________________||
from RecoJets.Configuration.CaloTowersES_cfi import *
from RecoMET.METProducers.CaloMET_cfi import *
from RecoMET.METProducers.hcalnoiseinfoproducer_cfi import *
from RecoMET.METProducers.MuonMETValueMapProducer_cff import *
from RecoMET.METProducers.caloMetM_cfi import *
from RecoMET.Configuration.RecoMET_BeamHaloId_cff import *

##____________________________________________________________________________||
metrecoTask = cms.Task(
        caloMet,
        caloMetBE,
        caloMetBEFO,
        muonMETValueMapProducer,
        caloMetM,
        BeamHaloIdTask
        )
metreco = cms.Sequence(metrecoTask)

##____________________________________________________________________________||
metrecoPlusHCALNoiseTask = cms.Task( metrecoTask,  hcalnoise )
metrecoPlusHCALNoise = cms.Sequence(metrecoPlusHCALNoiseTask)

from Configuration.Eras.Modifier_phase2_hcal_cff import phase2_hcal
phase2_hcal.toReplaceWith( metrecoPlusHCALNoiseTask, metrecoTask )

##____________________________________________________________________________||
