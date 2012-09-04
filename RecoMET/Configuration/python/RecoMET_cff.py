import FWCore.ParameterSet.Config as cms
# $Id$

##____________________________________________________________________________||
from RecoJets.Configuration.CaloTowersES_cfi import *
from RecoMET.METProducers.CaloTowersOpt_cfi import *
from RecoMET.METProducers.CaloMET_cfi import *
from RecoMET.METProducers.HTMET_cfi import *
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
        metNoHFHO+
        calotoweroptmaker+
        metOpt+
        metOptNoHF+
        calotoweroptmakerWithHO+
        metOptHO+
        metOptNoHFHO+
        htMetKT4+
        htMetKT6+
        htMetIC5+
        htMetAK5+
        htMetAK7+
        muonMETValueMapProducer+
        corMetGlobalMuons+
        muonTCMETValueMapProducer+
        tcMetSequence+
        BeamHaloId
        )

##____________________________________________________________________________||
metrecoPlusHCALNoise = cms.Sequence( metreco + hcalnoise )

##____________________________________________________________________________||
