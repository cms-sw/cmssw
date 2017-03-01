import FWCore.ParameterSet.Config as cms

# Name:   RecoMET.cff
# Author: R.Cavanaugh
# Date:   05.11.2006
# Notes:  CaloMET.cfi assumes that a product with label "caloTowers" is 
#         already written into the event.
from RecoMET.METProducers.genMetCalo_cfi import *
from RecoMET.METProducers.genMetCaloAndNonPrompt_cfi import *
from RecoMET.METProducers.genMetTrue_cfi import *
from RecoMET.METProducers.genMetFromGenJets_cfi import *
#

recoGenMET = cms.Sequence(genMetCalo+genMetTrue)

