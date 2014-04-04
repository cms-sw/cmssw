import FWCore.ParameterSet.Config as cms

JetIdParams = cms.PSet(
    #4 Eta Categories  0-2.5 2.5-2.75 2.75-3.0 3.0-5.0

    #Tight Id
    Pt010_Tight    = cms.vdouble( 0.5,0.6,0.6,0.9),
    Pt1020_Tight   = cms.vdouble(-0.2,0.2,0.2,0.6),
    Pt2030_Tight   = cms.vdouble( 0.3,0.4,0.7,0.8),
    Pt3050_Tight   = cms.vdouble( 0.5,0.4,0.8,0.9),

    #Medium Id
    Pt010_Medium   = cms.vdouble( 0.2,0.4,0.2,0.6),
    Pt1020_Medium  = cms.vdouble(-0.3,0. ,0. ,0.5),
    Pt2030_Medium  = cms.vdouble( 0.2,0.2,0.5,0.7),
    Pt3050_Medium  = cms.vdouble( 0.3,0.2,0.7,0.8),

    #Loose Id
    Pt010_Loose    = cms.vdouble( 0. , 0. , 0. ,0.2),
    Pt1020_Loose   = cms.vdouble(-0.4,-0.4,-0.4,0.4),
    Pt2030_Loose   = cms.vdouble( 0. , 0. , 0.2,0.6),
    Pt3050_Loose   = cms.vdouble( 0. , 0. , 0.6,0.2)
)

PuJetIdOptMVA_wp = cms.PSet(
    #4 Eta Categories  0-2.5 2.5-2.75 2.75-3.0 3.0-5.0

    #Tight Id
    Pt010_Tight    = cms.vdouble(-0.5,-0.2,-0.83,-0.7),
    Pt1020_Tight   = cms.vdouble(-0.5,-0.2,-0.83,-0.7),
    Pt2030_Tight   = cms.vdouble(-0.2,  0.,    0.,  0.),
    Pt3050_Tight   = cms.vdouble(-0.2,  0.,    0.,  0.),

    #Medium Id
    Pt010_Medium   = cms.vdouble(-0.73,-0.89,-0.89,-0.83),
    Pt1020_Medium  = cms.vdouble(-0.73,-0.89,-0.89,-0.83),
    Pt2030_Medium  = cms.vdouble(0.1,  -0.4, -0.4, -0.45),
    Pt3050_Medium  = cms.vdouble(0.1,  -0.4, -0.4, -0.45),

    #Loose Id
    Pt010_Loose    = cms.vdouble(-0.9,-0.9, -0.9,-0.9),
    Pt1020_Loose   = cms.vdouble(-0.9,-0.9, -0.9,-0.9),
    Pt2030_Loose   = cms.vdouble(-0.4,-0.85,-0.7,-0.6),
    Pt3050_Loose   = cms.vdouble(-0.4,-0.85,-0.7,-0.6)
)

PuJetIdMinMVA_wp = cms.PSet(
    #4 Eta Categories  0-2.5 2.5-2.75 2.75-3.0 3.0-5.0

    #Tight Id
    Pt010_Tight    = cms.vdouble(-0.5,-0.2,-0.83,-0.7),
    Pt1020_Tight   = cms.vdouble(-0.5,-0.2,-0.83,-0.7),
    Pt2030_Tight   = cms.vdouble(-0.2,  0.,    0.,  0.),
    Pt3050_Tight   = cms.vdouble(-0.2,  0.,    0.,  0.),

    #Medium Id
    Pt010_Medium   = cms.vdouble(-0.73,-0.89,-0.89,-0.83),
    Pt1020_Medium  = cms.vdouble(-0.73,-0.89,-0.89,-0.83),
    Pt2030_Medium  = cms.vdouble(0.1,  -0.4, -0.5, -0.45),
    Pt3050_Medium  = cms.vdouble(0.1,  -0.4, -0.5, -0.45),

    #Loose Id
    Pt010_Loose    = cms.vdouble(-0.9,-0.9, -0.94,-0.9),
    Pt1020_Loose   = cms.vdouble(-0.9,-0.9, -0.94,-0.9),
    Pt2030_Loose   = cms.vdouble(-0.4,-0.85,-0.7,-0.6),
    Pt3050_Loose   = cms.vdouble(-0.4,-0.85,-0.7,-0.6)
)

EmptyJetIdParams = cms.PSet(
    #4 Eta Categories  0-2.5 2.5-2.75 2.75-3.0 3.0-5.0

    #Tight Id
    Pt010_Tight    = cms.vdouble(-999.,-999.,-999.,-999.),
    Pt1020_Tight   = cms.vdouble(-999.,-999.,-999.,-999.),
    Pt2030_Tight   = cms.vdouble(-999.,-999.,-999.,-999.),
    Pt3050_Tight   = cms.vdouble(-999.,-999.,-999.,-999.),

    #Medium Id
    Pt010_Medium   = cms.vdouble(-999.,-999.,-999.,-999.),
    Pt1020_Medium  = cms.vdouble(-999.,-999.,-999.,-999.),
    Pt2030_Medium  = cms.vdouble(-999.,-999.,-999.,-999.),
    Pt3050_Medium  = cms.vdouble(-999.,-999.,-999.,-999.),

    #Loose Id
    Pt010_Loose    = cms.vdouble(-999.,-999.,-999.,-999.),
    Pt1020_Loose   = cms.vdouble(-999.,-999.,-999.,-999.),
    Pt2030_Loose   = cms.vdouble(-999.,-999.,-999.,-999.),
    Pt3050_Loose   = cms.vdouble(-999.,-999.,-999.,-999.)
)

