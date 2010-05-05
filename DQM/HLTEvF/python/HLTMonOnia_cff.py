import FWCore.ParameterSet.Config as cms

# Onia DQM
from DQM.HLTEvF.HLTOniaSource_cfi import *

#Onia Summary
from DQM.HLTEvF.HLTMonOniaBits_cfi import *


hltMonOnia = cms.Sequence(hltOniaSource * hltMonOniaBits)

