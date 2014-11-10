import FWCore.ParameterSet.Config as cms

variableJTAPars = cms.PSet(a_dR = cms.double(-0.001053),
                           b_dR = cms.double(0.6263),
                           a_pT = cms.double(0.005263),
                           b_pT = cms.double(0.3684),
                           min_pT = cms.double( 120 ),
                           max_pT = cms.double( 500 ),
                           min_pT_dRcut = cms.double( 0.5 ),
                           max_pT_dRcut = cms.double( 0.1 ),
                           max_pT_trackPTcut = cms.double( 3 )
                           )


