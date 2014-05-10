import FWCore.ParameterSet.Config as cms

##____________________________________________________________________________||
from RecoMET.METProducers.CaloMET_OldNames_cfi import *
from RecoMET.METProducers.CaloMET_OldNamesOpt_cfi import *

##____________________________________________________________________________||
caloMet = met.clone()
caloMetBEFO = metHO.clone()
caloMetBE = metNoHF.clone()
caloMetBEO = metNoHFHO.clone()

##____________________________________________________________________________||
