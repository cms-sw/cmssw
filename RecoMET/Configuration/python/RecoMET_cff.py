import FWCore.ParameterSet.Config as cms

##____________________________________________________________________________||
from RecoJets.Configuration.CaloTowersES_cfi import *
from RecoMET.METProducers.CaloMET_cfi import *
from RecoMET.METProducers.CaloMETSignif_cfi import *
from RecoMET.METProducers.hcalnoiseinfoproducer_cfi import *
from RecoMET.METProducers.MuonMETValueMapProducer_cff import *
from RecoMET.METProducers.MetMuonCorrections_cff import *
from RecoMET.Configuration.RecoMET_BeamHaloId_cff import *

##____________________________________________________________________________||
metreco = cms.Sequence(
        caloMet+
        caloMetBE+
        caloMetBEFO+
        muonMETValueMapProducer+
        corMetGlobalMuons+
        BeamHaloId
        )

##____________________________________________________________________________||
metrecoPlusHCALNoise = cms.Sequence( metreco + hcalnoise )

##____________________________________________________________________________||
