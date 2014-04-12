import FWCore.ParameterSet.Config as cms

# Name:   RecoHTMET.cff
# Author: R.Cavanaugh
# Date:   19.03.2007
# Notes:  HTMET.cfi assumes that a product with label "midPointCone5CaloJets" is 
#         already written into the event.
from RecoMET.METProducers.HTMET_cfi import *
htmetreco = cms.Sequence(htMet)

