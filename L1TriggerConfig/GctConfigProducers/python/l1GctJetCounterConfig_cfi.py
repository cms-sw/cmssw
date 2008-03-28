# The following comments couldn't be translated into the new config version:

# Jet rank threshold cut, rank 1 is 10 GeV
# Jet rank threshold cut combined
#     with |eta|<3
# Jet rank threshold cut, rank 11 is 30 GeV
# Jet rank threshold cut combined
#     with |eta|<3
# Jet rank threshold cut, rank 19 is 60 GeV
# Jet rank threshold cut, rank 1 is 10 GeV
# Jet rank threshold cut combined
#     with |eta|<3
# Jet rank threshold cut, rank 11 is 30 GeV
# Jet rank threshold cut combined
#     with |eta|<3
# Jet rank threshold cut, rank 19 is 60 GeV
import FWCore.ParameterSet.Config as cms

jcSetup1 = cms.PSet(
    jetCountersNegativeWheel = cms.VPSet(cms.PSet(
        cutDescriptionList = cms.vstring('JC_minRank_1')
    ), cms.PSet(
        cutDescriptionList = cms.vstring('JC_minRank_1', 'JC_centralEta_6')
    ), cms.PSet(
        cutDescriptionList = cms.vstring('JC_minRank_11')
    ), cms.PSet(
        cutDescriptionList = cms.vstring('JC_minRank_11', 'JC_centralEta_6')
    ), cms.PSet(
        cutDescriptionList = cms.vstring('JC_minRank_19')
    )),
    jetCountersPositiveWheel = cms.VPSet(cms.PSet(
        cutDescriptionList = cms.vstring('JC_minRank_1')
    ), cms.PSet(
        cutDescriptionList = cms.vstring('JC_minRank_1', 'JC_centralEta_6')
    ), cms.PSet(
        cutDescriptionList = cms.vstring('JC_minRank_11')
    ), cms.PSet(
        cutDescriptionList = cms.vstring('JC_minRank_11', 'JC_centralEta_6')
    ), cms.PSet(
        cutDescriptionList = cms.vstring('JC_minRank_19')
    ))
)

