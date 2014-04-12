import FWCore.ParameterSet.Config as cms

#
#Egamma skim, very high Et control sample
#
#The skim selects events based on HLT information alone
#
from EgammaAnalysis.CSA07Skims.EgammaVeryHighEtTrigger_cfi import *
egammaVeryHighEt = cms.Path(EgammaVeryHighEtTrigger)

