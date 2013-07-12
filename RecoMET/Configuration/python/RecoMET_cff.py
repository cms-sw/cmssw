import FWCore.ParameterSet.Config as cms
# $Id: RecoMET_cff.py,v 1.18 2012/09/04 21:18:33 sakuma Exp $

##____________________________________________________________________________||
from RecoJets.Configuration.CaloTowersES_cfi import *
from RecoMET.METProducers.CaloMET_cfi import *
from RecoMET.METProducers.CaloMETSignif_cfi import *
from RecoMET.METProducers.hcalnoiseinfoproducer_cfi import *
from RecoMET.METProducers.MuonMETValueMapProducer_cff import *
from RecoMET.METProducers.MuonTCMETValueMapProducer_cff import *
from RecoMET.METProducers.MetMuonCorrections_cff import *
from RecoMET.Configuration.RecoMET_BeamHaloId_cff import *
from RecoMET.Configuration.RecoTCMET_cff import *

##____________________________________________________________________________||
metreco = cms.Sequence(
        met+
        metNoHF+
        metHO+
        muonMETValueMapProducer+
        corMetGlobalMuons+
        muonTCMETValueMapProducer+
        tcMetSequence+
        BeamHaloId
        )

##____________________________________________________________________________||
metrecoPlusHCALNoise = cms.Sequence( metreco + hcalnoise )

##____________________________________________________________________________||
