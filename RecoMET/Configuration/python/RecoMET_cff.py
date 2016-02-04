import FWCore.ParameterSet.Config as cms

# Name:   RecoMET.cff
# Original Author: R.Cavanaugh
# Date:   05.11.2006
# Notes:  CaloMET.cfi assumes that a product with label "caloTowers" is
#         already written into the event.
# Modification by F. Ratnikov and R. Remington
# Date: 10/21/08 
# Addition of MET significance by F.Blekman
# Date: 10/23/08
# Addition of HCAL noise by JP Chou
# Date:  3/26/09

from RecoJets.Configuration.CaloTowersES_cfi import *
from RecoMET.METProducers.CaloTowersOpt_cfi import *
from RecoMET.METProducers.CaloMET_cfi import *
from RecoMET.METProducers.HTMET_cfi import *
from RecoMET.METProducers.CaloMETSignif_cfi import *
#from RecoMET.METProducers.TCMET_cfi import *
from RecoMET.METProducers.hcalnoiseinfoproducer_cfi import *
from RecoMET.METProducers.MuonMETValueMapProducer_cff import *
from RecoMET.METProducers.MuonTCMETValueMapProducer_cff import *
from RecoMET.METProducers.MetMuonCorrections_cff import *
from RecoMET.Configuration.RecoMET_BeamHaloId_cff import *
from RecoMET.Configuration.RecoTCMET_cff import *

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
#        htMetSC5+
#        htMetSC7+
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

metrecoPlusHCALNoise = cms.Sequence( metreco + hcalnoise )



