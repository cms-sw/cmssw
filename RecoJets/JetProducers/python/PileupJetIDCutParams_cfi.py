import FWCore.ParameterSet.Config as cms

###########################################################
## Working points for the 81X training (completed in 80X with variable fixes)
###########################################################
full_81x_chs_wp  = cms.PSet(
    #4 Eta Categories  0-2.5 2.5-2.75 2.75-3.0 3.0-5.0

    #Tight Id
    Pt010_Tight    = cms.vdouble( 0.69, -0.35, -0.26, -0.21),
    Pt1020_Tight   = cms.vdouble( 0.69, -0.35, -0.26, -0.21),
    Pt2030_Tight   = cms.vdouble( 0.69, -0.35, -0.26, -0.21),
    Pt3040_Tight   = cms.vdouble( 0.86, -0.10, -0.05, -0.01),
    Pt4050_Tight   = cms.vdouble( 0.86, -0.10, -0.05, -0.01),

    #Medium Id
    Pt010_Medium   = cms.vdouble( 0.18, -0.55, -0.42, -0.36),
    Pt1020_Medium  = cms.vdouble( 0.18, -0.55, -0.42, -0.36),
    Pt2030_Medium  = cms.vdouble( 0.18, -0.55, -0.42, -0.36),
    Pt3040_Medium  = cms.vdouble( 0.61, -0.35, -0.23, -0.17),
    Pt4050_Medium  = cms.vdouble( 0.61, -0.35, -0.23, -0.17),

    #Loose Id
    Pt010_Loose    = cms.vdouble(-0.97, -0.68, -0.53, -0.47),
    Pt1020_Loose   = cms.vdouble(-0.97, -0.68, -0.53, -0.47),
    Pt2030_Loose   = cms.vdouble(-0.97, -0.68, -0.53, -0.47),
    Pt3040_Loose   = cms.vdouble(-0.89, -0.52, -0.38, -0.30),
    Pt4050_Loose   = cms.vdouble(-0.89, -0.52, -0.38, -0.30)
)

###########################################################
## Working points for the 102X training
###########################################################
full_102x_chs_wp = full_81x_chs_wp.clone()

###########################################################
## Working points for the 94X training
###########################################################
full_94x_chs_wp = full_81x_chs_wp.clone()

##########################################################
## Working points for the 106X UL17 training
###########################################################
full_106x_UL17_chs_wp = cms.PSet(
    # 4 Eta Categories  0-2.5 2.5-2.75 2.75-3.0 3.0-5.0
    # 5 Pt Categories   0-10, 10-20, 20-30, 30-40, 40-50

    #Tight Id
    Pt010_Tight  = cms.vdouble( 0.77, 0.38, -0.31, -0.21),
    Pt1020_Tight = cms.vdouble( 0.77, 0.38, -0.31, -0.21),
    Pt2030_Tight = cms.vdouble( 0.90, 0.60, -0.12, -0.13),
    Pt3040_Tight = cms.vdouble( 0.96, 0.82, 0.20, 0.09),
    Pt4050_Tight = cms.vdouble( 0.98, 0.92, 0.47, 0.29),

    #Medium Id
    Pt010_Medium  = cms.vdouble( 0.26, -0.33, -0.54, -0.37),
    Pt1020_Medium = cms.vdouble( 0.26, -0.33, -0.54, -0.37),
    Pt2030_Medium = cms.vdouble( 0.68, -0.04, -0.43, -0.30),
    Pt3040_Medium = cms.vdouble( 0.90, 0.36, -0.16, -0.09),
    Pt4050_Medium = cms.vdouble( 0.96, 0.61, 0.14, 0.12),

    #Loose Id
    Pt010_Loose  = cms.vdouble(-0.95, -0.72, -0.68, -0.47),
    Pt1020_Loose = cms.vdouble(-0.95, -0.72, -0.68, -0.47),
    Pt2030_Loose = cms.vdouble(-0.88, -0.55, -0.60, -0.43),
    Pt3040_Loose = cms.vdouble(-0.63, -0.18, -0.43, -0.24),
    Pt4050_Loose = cms.vdouble(-0.19, 0.22, -0.13, -0.03)
)

##########################################################
## Working points for the 106X UL18 training
###########################################################
full_106x_UL18_chs_wp = full_106x_UL17_chs_wp.clone()

###########################################################
## Working points for the 106X UL16 training
###########################################################
full_106x_UL16_chs_wp = cms.PSet(
    # 4 Eta Categories  0-2.5 2.5-2.75 2.75-3.0 3.0-5.0
    # 5 Pt Categories   0-10, 10-20, 20-30, 30-40, 40-50

    #Tight Id
    Pt010_Tight  = cms.vdouble(-0.95, -0.70, -0.52, -0.49),
    Pt1020_Tight = cms.vdouble(-0.95, -0.70, -0.52, -0.49),
    Pt2030_Tight = cms.vdouble(-0.90, -0.57, -0.43, -0.42),
    Pt3040_Tight = cms.vdouble(-0.71, -0.36, -0.29, -0.23),
    Pt4050_Tight = cms.vdouble(-0.42, -0.09, -0.14, -0.02),

    #Medium Id
    Pt010_Medium  = cms.vdouble(0.20, -0.56, -0.43, -0.38),
    Pt1020_Medium = cms.vdouble(0.20, -0.56, -0.43, -0.38),
    Pt2030_Medium = cms.vdouble(0.62, -0.39, -0.32, -0.29),
    Pt3040_Medium = cms.vdouble(0.86, -0.10, -0.15, -0.08),
    Pt4050_Medium = cms.vdouble(0.93, 0.19, 0.04, 0.12),

    #Loose Id
    Pt010_Loose  = cms.vdouble(0.71, -0.32, -0.30, -0.22),
    Pt1020_Loose = cms.vdouble(0.71, -0.32, -0.30, -0.22),
    Pt2030_Loose = cms.vdouble(0.87, -0.08, -0.16, -0.12),
    Pt3040_Loose = cms.vdouble(0.94, 0.24, 0.05, 0.10),
    Pt4050_Loose = cms.vdouble(0.97, 0.48, 0.26, 0.29)
)

##########################################################
## Working points for the 106X UL16 APV training
###########################################################
full_106x_UL16APV_chs_wp = full_106x_UL16_chs_wp.clone()

#########################################################
## Empty cutbased WP for compatibility
###########################################################
EmptyCutBased_wp = cms.PSet()
#4 Eta Categories  0-2.5 2.5-2.75 2.75-3.0 3.0-5.0
for pt in ["010", "1020", "2030", "3040", "4050"]:
    for tp in ["BetaStar", "RMS"]:
        for wp in ["Loose", "Medium", "Tight"]:
            setattr(EmptyCutBased_wp, "Pt" + pt + "_" + tp + wp, cms.vdouble(-999.,-999.,-999.,-999.))
