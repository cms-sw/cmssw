import FWCore.ParameterSet.Config as cms

from HLTrigger.HLTfilters.hlt1CaloJetRegionalDefaults_cff import *
#
hlt1CaloJetRegional = cms.EDFilter("HLT1CaloJet",
    hlt1CaloJetRegionalDefaults
)


