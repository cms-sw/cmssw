import FWCore.ParameterSet.Config as cms
import copy

# Layer: Layer.TIB1 ratio of ratios: 1.02533166915
# Layer: Layer.TIB2 ratio of ratios: 1.01521093183
# Layer: Layer.TIB3 ratio of ratios: 1.01552419364
# Layer: Layer.TIB4 ratio of ratios: 0.95224779507
# Layer: Layer.TOB1 ratio of ratios: 1.01219411074
# Layer: Layer.TOB2 ratio of ratios: 1.00835168635
# Layer: Layer.TOB3 ratio of ratios: 0.996159099354
# Layer: Layer.TOB4 ratio of ratios: 0.997676926445
# Layer: Layer.TOB5 ratio of ratios: 0.993886888572
# Layer: Layer.TOB6 ratio of ratios: 0.997490411188
# Layer: Layer.TIDP1 ratio of ratios: 1.0314881072
# Layer: Layer.TIDP2 ratio of ratios: 1.02853114088
# Layer: Layer.TIDP3 ratio of ratios: 1.0518768914
# Layer: Layer.TIDM1 ratio of ratios: 1.03421675878
# Layer: Layer.TIDM2 ratio of ratios: 1.04546785025
# Layer: Layer.TIDM3 ratio of ratios: 1.0311586591
# Layer: Layer.TECP1 ratio of ratios: 1.04989866792
# Layer: Layer.TECP2 ratio of ratios: 1.03711260343
# Layer: Layer.TECP3 ratio of ratios: 1.04297992451
# Layer: Layer.TECP4 ratio of ratios: 1.04669045804
# Layer: Layer.TECP5 ratio of ratios: 1.03838249025
# Layer: Layer.TECP6 ratio of ratios: 1.04727471357
# Layer: Layer.TECP7 ratio of ratios: 1.03632636024
# Layer: Layer.TECP8 ratio of ratios: 1.04860504406
# Layer: Layer.TECP9 ratio of ratios: 1.03398568113
# Layer: Layer.TECM1 ratio of ratios: 1.04750199121
# Layer: Layer.TECM2 ratio of ratios: 1.03771633506
# Layer: Layer.TECM3 ratio of ratios: 1.0409554129
# Layer: Layer.TECM4 ratio of ratios: 1.03630204118
# Layer: Layer.TECM5 ratio of ratios: 1.0417988699
# Layer: Layer.TECM6 ratio of ratios: 1.03864754217
# Layer: Layer.TECM7 ratio of ratios: 1.03868976393
# Layer: Layer.TECM8 ratio of ratios: 1.03942709841
# Layer: Layer.TECM9 ratio of ratios: 1.03678940814

byLayer = cms.VPSet(

    ################## TIB ##################

    cms.PSet(partition = cms.string("TIB_1"),
             doScale   = cms.bool(True),
             doSmear   = cms.bool(False),
             scaleFactor = cms.double(1.02533166915),
             smearFactor = cms.double(0.0)
             ),
    cms.PSet(partition = cms.string("TIB_2"),
             doScale   = cms.bool(True),
             doSmear   = cms.bool(False),
             scaleFactor = cms.double(1.01521093183),
             smearFactor = cms.double(0.0)
             ),
    cms.PSet(partition = cms.string("TIB_3"),
             doScale   = cms.bool(True),
             doSmear   = cms.bool(False),
             scaleFactor = cms.double(1.01552419364),
             smearFactor = cms.double(0.0)
             ),
    cms.PSet(partition = cms.string("TIB_4"),
             doScale   = cms.bool(True),
             doSmear   = cms.bool(False),
             scaleFactor = cms.double(0.95224779507),
             smearFactor = cms.double(0.0)
             ),

    ################## TOB ##################

    cms.PSet(partition = cms.string("TOB_1"),
             doScale   = cms.bool(True),
             doSmear   = cms.bool(False),
             scaleFactor = cms.double(1.01219411074),
             smearFactor = cms.double(0.0)
             ),
    cms.PSet(partition = cms.string("TOB_2"),
             doScale   = cms.bool(True),
             doSmear   = cms.bool(False),
             scaleFactor = cms.double(1.00835168635),
             smearFactor = cms.double(0.0)
             ),
    cms.PSet(partition = cms.string("TOB_3"),
             doScale   = cms.bool(True),
             doSmear   = cms.bool(False),
             scaleFactor = cms.double(0.996159099354),
             smearFactor = cms.double(0.0)
             ),
    cms.PSet(partition = cms.string("TOB_4"),
             doScale   = cms.bool(True),
             doSmear   = cms.bool(False),
             scaleFactor = cms.double(0.997676926445),
             smearFactor = cms.double(0.0)
             ),
    cms.PSet(partition = cms.string("TOB_5"),
             doScale   = cms.bool(True),
             doSmear   = cms.bool(False),
             scaleFactor = cms.double(0.993886888572),
             smearFactor = cms.double(0.0)
             ),
    cms.PSet(partition = cms.string("TOB_6"),
             doScale   = cms.bool(True),
             doSmear   = cms.bool(False),
             scaleFactor = cms.double(0.997490411188),
             smearFactor = cms.double(0.0)
             ),

    ################## TID Plus ##################

    cms.PSet(partition = cms.string("TIDP_1"),
             doScale   = cms.bool(True),
             doSmear   = cms.bool(False),
             scaleFactor = cms.double(1.0314881072),
             smearFactor = cms.double(0.0)
             ),
    cms.PSet(partition = cms.string("TIDP_2"),
             doScale   = cms.bool(True),
             doSmear   = cms.bool(False),
             scaleFactor = cms.double(1.02853114088),
             smearFactor = cms.double(0.0)
             ),
    cms.PSet(partition = cms.string("TIDP_3"),
             doScale   = cms.bool(True),
             doSmear   = cms.bool(False),
             scaleFactor = cms.double(1.0518768914),
             smearFactor = cms.double(0.0)
             ),

    ################## TID Minus ##################

    cms.PSet(partition = cms.string("TIDM_1"),
             doScale   = cms.bool(True),
             doSmear   = cms.bool(False),
             scaleFactor = cms.double(1.03421675878),
             smearFactor = cms.double(0.0)
             ),
    cms.PSet(partition = cms.string("TIDM_2"),
             doScale   = cms.bool(True),
             doSmear   = cms.bool(False),
             scaleFactor = cms.double(1.04546785025),
             smearFactor = cms.double(0.0)
             ),
    cms.PSet(partition = cms.string("TIDM_3"),
             doScale   = cms.bool(True),
             doSmear   = cms.bool(False),
             scaleFactor = cms.double(1.0311586591),
             smearFactor = cms.double(0.0)
             ),
    
    ################## TEC plus ##################

    cms.PSet(partition = cms.string("TECP_1"),
             doScale   = cms.bool(True),
             doSmear   = cms.bool(False),
             scaleFactor = cms.double(1.04989866792),
             smearFactor = cms.double(0.0)
             ),
    cms.PSet(partition = cms.string("TECP_2"),
             doScale   = cms.bool(True),
             doSmear   = cms.bool(False),
             scaleFactor = cms.double(1.03711260343),
             smearFactor = cms.double(0.0)
             ),
    cms.PSet(partition = cms.string("TECP_3"),
             doScale   = cms.bool(True),
             doSmear   = cms.bool(False),
             scaleFactor = cms.double(1.04297992451),
             smearFactor = cms.double(0.0)
             ),
    cms.PSet(partition = cms.string("TECP_4"),
             doScale   = cms.bool(True),
             doSmear   = cms.bool(False),
             scaleFactor = cms.double(1.04669045804),
             smearFactor = cms.double(0.0)
             ),
    cms.PSet(partition = cms.string("TECP_5"),
             doScale   = cms.bool(True),
             doSmear   = cms.bool(False),
             scaleFactor = cms.double(1.03838249025),
             smearFactor = cms.double(0.0)
             ),
    cms.PSet(partition = cms.string("TECP_6"),
             doScale   = cms.bool(True),
             doSmear   = cms.bool(False),
             scaleFactor = cms.double(1.04727471357),
             smearFactor = cms.double(0.0)
             ),
    cms.PSet(partition = cms.string("TECP_7"),
             doScale   = cms.bool(True),
             doSmear   = cms.bool(False),
             scaleFactor = cms.double(1.03632636024),
             smearFactor = cms.double(0.0)
             ),
    cms.PSet(partition = cms.string("TECP_8"),
             doScale   = cms.bool(True),
             doSmear   = cms.bool(False),
             scaleFactor = cms.double(1.04860504406),
             smearFactor = cms.double(0.0)
             ),
    cms.PSet(partition = cms.string("TECP_9"),
             doScale   = cms.bool(True),
             doSmear   = cms.bool(False),
             scaleFactor = cms.double(1.03398568113),
             smearFactor = cms.double(0.0)
             ),

    ################## TEC Minus ##################
    cms.PSet(partition = cms.string("TECM_1"),
             doScale   = cms.bool(True),
             doSmear   = cms.bool(False),
             scaleFactor = cms.double(1.04750199121),
             smearFactor = cms.double(0.0)
             ),
    cms.PSet(partition = cms.string("TECM_2"),
             doScale   = cms.bool(True),
             doSmear   = cms.bool(False),
             scaleFactor = cms.double(1.03771633506),
             smearFactor = cms.double(0.0)
             ),
    cms.PSet(partition = cms.string("TECM_3"),
             doScale   = cms.bool(True),
             doSmear   = cms.bool(False),
             scaleFactor = cms.double(1.0409554129),
             smearFactor = cms.double(0.0)
             ),
    cms.PSet(partition = cms.string("TECM_4"),
             doScale   = cms.bool(True),
             doSmear   = cms.bool(False),
             scaleFactor = cms.double(1.03630204118),
             smearFactor = cms.double(0.0)
             ),
    cms.PSet(partition = cms.string("TECM_5"),
             doScale   = cms.bool(True),
             doSmear   = cms.bool(False),
             scaleFactor = cms.double(1.0417988699),
             smearFactor = cms.double(0.0)
             ),    
    cms.PSet(partition = cms.string("TECM_6"),
             doScale   = cms.bool(True),
             doSmear   = cms.bool(False),
             scaleFactor = cms.double(1.03864754217),
             smearFactor = cms.double(0.0)
             ),
    cms.PSet(partition = cms.string("TECM_7"),
             doScale   = cms.bool(True),
             doSmear   = cms.bool(False),
             scaleFactor = cms.double(1.0386897639),
             smearFactor = cms.double(0.0)
             ),
    cms.PSet(partition = cms.string("TECM_8"),
             doScale   = cms.bool(True),
             doSmear   = cms.bool(False),
             scaleFactor = cms.double(1.03942709841),
             smearFactor = cms.double(0.0)
             ),
    cms.PSet(partition = cms.string("TECM_9"),
             doScale   = cms.bool(True),
             doSmear   = cms.bool(False),
             scaleFactor = cms.double(1.03678940814),
             smearFactor = cms.double(0.0)
             )
    ) 
