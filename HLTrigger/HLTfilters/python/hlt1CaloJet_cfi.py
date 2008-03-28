import FWCore.ParameterSet.Config as cms

from HLTrigger.HLTfilters.hlt1CaloJetDefaults_cff import *
#
hlt1CaloJet = cms.EDFilter("HLT1CaloJet",
    hlt1CaloJetDefaults
)


