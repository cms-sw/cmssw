import FWCore.ParameterSet.Config as cms

# Name:   RecoMET.cff
# Author: R.Cavanaugh
# Date:   05.11.2006
# Notes:  CaloMET.cfi assumes that a product with label "caloTowers" is 
#         already written into the event.
from RecoJets.Configuration.CaloTowersES_cfi import *
from RecoMET.METProducers.CaloTowersOpt_cfi import *
from RecoMET.METProducers.CaloMET_cfi import *
from RecoMET.METProducers.HTMET_cfi import *
#sequence metreco = {met, htMetIC5, htMetMC5}
metreco = cms.Sequence(
    met+metNoHF+metHO+metNoHFHO+
    calotoweroptmaker+metOpt+metOptNoHF+calotoweroptmakerWithHO+metOptHO+metOptNoHFHO+
    htMetSC5+htMetSC7+htMetKT4+htMetKT6+htMetIC5
    )

