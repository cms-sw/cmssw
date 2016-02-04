import FWCore.ParameterSet.Config as cms

pythiaDefaultBlock = cms.PSet(
    pythiaDefault = cms.vstring('PMAS(5,1)=4.8 ! b quark mass', 
        'PMAS(6,1)=172.3 ! t quark mass')
)

