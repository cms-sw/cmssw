from Configuration.StandardSequences.CondDBESSource_cff import GlobalTag as essource
connectionString = essource.connect.value()

SiPixelLARecord           =   "SiPixelLorentzAngleRcd"            
SiPixelSimLARecord        =   "SiPixelLorentzAngleSimRcd"
SiPixelGenErrorRecord     =   "SiPixelGenErrorDBObjectRcd"
SiPixelTemplatesRecord    =   "SiPixelTemplateDBObjectRcd"
SiPixel2DTemplatesRecord  =   "SiPixel2DTemplateDBObjectRcd"
TrackerLARecord           =   "SiPhase2OuterTrackerLorentzAngleRcd"
TrackerSimLARecord        =   "SiPhase2OuterTrackerLorentzAngleSimRcd"
TkAlRecord                =   "TrackerAlignmentRcd"
TkAPERecord               =   "TrackerAlignmentErrorExtendedRcd"
TkSurfRecord              =   "TrackerSurfaceDeformationRcd"

##
## Active geometries: https://github.com/cms-sw/cmssw/blob/master/Configuration/Geometry/README.md
##
## T21: Phase2 tilted tracker. Outer Tracker (v8.0.0): TBPS update in Layer 1 (facilitate IT insertion) + In all TEDD, update sensors Z inter-spacing. Inner Tracker: (v6.1.5) from previous T17
## (TFPX: Changed sensors spacing within all double-disks + Increased distance between Disks 6 and 7 + TBPX portcards between Disks 6 and 7.)
## T25: Phase2 tilted tracker. Outer Tracker (v8.0.0): same as T24/T21. Inner Tracker (v7.0.2): Based on (v6.1.5) (T24/T21), but with 3D sensors in TBPX L1.
## T30: Phase2 tilted tracker, exploratory geometry *only to be used in D91 for now*. Outer Tracker (v8.0.1): based on v8.0.0 with updated TB2S spacing. Inner Tracker (v6.4.0): based on v6.1.5 but TFPX with more realistic module positions
## T33: Phase2 tilted tracker. As T25 - OT v8.0.0 (but with BTL overlap removed). IT: v7.1.1 - as in T25, except with more realistic description of 3D sensors.

#combines in a single dict of dict the tags defined below
allTags={}

allTags["LA"] = {
    'T21' : ( ','.join( [ 'SiPixelLorentzAngle_phase2_T15_v5_mc' ,SiPixelLARecord,connectionString, "", "2020-05-05 20:00:00.000"] ), ),  #uH = 0.053/T (TBPX), uH=0.0/T (TEPX+TFPX)
    'T25' : ( ','.join( [ 'SiPixelLorentzAngle_phase2_T25_v0_mc' ,SiPixelLARecord,connectionString, "", "2021-03-16 20:00:00.000"] ), ),  #uH = 0.053/T (TBPX L2,L3,L4), uH=0.0/T (TBPX L1 TEPX+TFPX)
    'T30' : ( ','.join( [ 'SiPixelLorentzAngle_phase2_IT_v6.4.0_25x100_v1_mc' ,SiPixelLARecord,connectionString, "", "2021-11-22 21:00:00.000"] ), ),  #uH = 0.053/T (TBPX), uH=0.0/T (TEPX+TFPX)
    'T33' : ( ','.join( [ 'SiPixelLorentzAngle_phase2_IT_v7.1.1_25x100_v1_mc' ,SiPixelLARecord,connectionString, "", "2023-05-16 20:00:00.000"] ), ),  #uH = 0.053/T (TBPX L2,L3,L4), uH=0.0/T (TBPX L1 TEPX+TFPX)
}

allTags["LAWidth"] = {
    'T21' : ( ','.join( [ 'SiPixelLorentzAngle_phase2_T15_mc_forWidthEmpty' ,SiPixelLARecord,connectionString, "forWidth", "2019-11-05 20:00:00.000"] ), ),  # uH=0.0/T (fall-back to offset)
    'T25' : ( ','.join( [ 'SiPixelLorentzAngle_phase2_T19_mc_forWidthEmpty' ,SiPixelLARecord,connectionString, "forWidth", "2020-02-23 14:00:00.000"] ), ),  # uH=0.0/T (fall-back to offset)
    'T30' : ( ','.join( [ 'SiPixelLorentzAngle_phase2_IT_v6.4.0_25x100_empty_mc' ,SiPixelLARecord,connectionString, "forWidth", "2021-11-29 20:00:00.000"] ), ),  # uH=0.0/T (fall-back to offset)
}

allTags["LAfromAlignment"] = {
    'T21' : ( ','.join( [ 'SiPixelLorentzAngle_phase2_T15_mc_forWidthEmpty' ,SiPixelLARecord,connectionString, "fromAlignment", "2019-11-05 20:00:00.000"] ), ),  # uH=0.0/T (not in use)
    'T25' : ( ','.join( [ 'SiPixelLorentzAngle_phase2_T19_mc_forWidthEmpty' ,SiPixelLARecord,connectionString, "fromAlignment", "2020-02-23 14:00:00.000"] ), ),  # uH=0.0/T (not in use)
    'T30' : ( ','.join( [ 'SiPixelLorentzAngle_phase2_IT_v6.4.0_25x100_empty_mc' ,SiPixelLARecord,connectionString, "fromAlignment", "2021-11-29 20:00:00.000"] ), ),  # uH=0.0/T (fall-back to offset)
}

allTags["SimLA"] = {
    'T21' : ( ','.join( [ 'SiPixelSimLorentzAngle_phase2_T15_v5_mc' ,SiPixelSimLARecord,connectionString, "", "2020-05-05 20:00:00.000"] ), ), #uH = 0.053/T (TBPX), uH=0.0/T (TEPX+TFPX)
    'T25' : ( ','.join( [ 'SiPixelSimLorentzAngle_phase2_T25_v0_mc' ,SiPixelSimLARecord,connectionString, "", "2021-03-16 20:00:00.000"] ), ), #uH = 0.053/T (TBPX L2,L3,L4), uH=0.0/T (TBPX L1 TEPX+TFPX)
    'T30' : ( ','.join( [ 'SiPixelSimLorentzAngle_phase2_IT_v6.4.0_25x100_v1_mc' ,SiPixelSimLARecord,connectionString, "", "2021-12-03 16:00:00.000"] ), ), #uH = 0.053/T (TBPX), uH=0.0/T (TEPX+TFPX)
}

allTags["GenError"] = {
    'T21' : ( ','.join( [ 'SiPixelGenErrorDBObject_phase2_IT_v6.1.5_25x100_v3_mc',SiPixelGenErrorRecord,connectionString, "", "2021-01-27 10:00:00"] ), ),  # cell is 25um (local-x) x 100um (local-y) , VBias=350V
    'T25' : ( ','.join( [ 'SiPixelGenErrorDBObject_phase2_IT_v7.0.2_25x100_v2_mc' ,SiPixelGenErrorRecord,connectionString, "", "2021-04-17 20:00:00"] ), ),  # cell is 25um (local-x) x 100um (local-y), VBias=350V, 3D pixels in TBPX L1
    'T30' : ( ','.join( [ 'SiPixelGenErrorDBObject_phase2_IT_v6.4.0_25x100_v1_mc',SiPixelGenErrorRecord,connectionString, "", "2021-11-22 21:00:00"] ), ),  # cell is 25um (local-x) x 100um (local-y) , VBias=350V
    'T33' : ( ','.join( [ 'SiPixelGenErrorDBObject_phase2_IT_v7.1.1_25x100_v1_mc' ,SiPixelGenErrorRecord,connectionString, "", "2023-05-16 20:00:00"] ), ),  # cell is 25um (local-x) x 100um (local-y) , VBias=350V, 3D pixels in TBPX L1
}

allTags["Template"] = {
    'T21' : ( ','.join( [ 'SiPixelTemplateDBObject_phase2_IT_v6.1.5_25x100_v3_mc',SiPixelTemplatesRecord,connectionString, "", "2021-01-27 10:00:00"] ), ),  # cell is 25um (local-x) x 100um (local-y) , VBias=350V
    'T25' : ( ','.join( [ 'SiPixelTemplateDBObject_phase2_IT_v7.0.2_25x100_v2_mc' ,SiPixelTemplatesRecord,connectionString, "", "2021-04-17 20:00:00"] ), ),  # cell is 25um (local-x) x 100um (local-y), VBias=350V, 3D pixels in TBPX L1
    'T30' : ( ','.join( [ 'SiPixelTemplateDBObject_phase2_IT_v6.4.0_25x100_v1_mc',SiPixelTemplatesRecord,connectionString, "", "2021-11-22 21:00:00"] ), ),  # cell is 25um (local-x) x 100um (local-y) , VBias=350V
    'T33' : ( ','.join( [ 'SiPixelTemplateDBObject_phase2_IT_v7.1.1_25x100_v1_mc' ,SiPixelTemplatesRecord,connectionString, "", "2023-05-16 20:00:00"] ), ),  # cell is 25um (local-x) x 100um (local-y) , VBias=350V, 3D pixels in TBPX L1
}

allTags["TkAlignment"] = {
    'T21' : ( ','.join( [ 'TrackerAlignment_Upgrade2026_T21_design_v0' ,TkAlRecord, connectionString, "", "2023-03-16 15:30:00"] ), ),
    'T25' : ( ','.join( [ 'TrackerAlignment_Upgrade2026_T25_design_v0' ,TkAlRecord, connectionString, "", "2023-03-16 15:30:00"] ), ),
    'T30' : ( ','.join( [ 'TrackerAlignment_Upgrade2026_T30_design_v0' ,TkAlRecord, connectionString, "", "2023-03-16 15:30:00"] ), ),
    'T33' : ( ','.join( [ 'TrackerAlignment_Upgrade2026_T33_design_v0' ,TkAlRecord, connectionString, "", "2023-06-07 21:00:00"] ), ),
}

allTags["TkAPE"] = {
    'T21' : ( ','.join( [ 'TrackerAlignmentErrorsExtended_Upgrade2026_T21_design_v0' ,TkAPERecord, connectionString, "", "2023-03-16 15:30:00"] ), ),
    'T25' : ( ','.join( [ 'TrackerAlignmentErrorsExtended_Upgrade2026_T25_design_v0' ,TkAPERecord, connectionString, "", "2023-03-16 15:30:00"] ), ),
    'T30' : ( ','.join( [ 'TrackerAlignmentErrorsExtended_Upgrade2026_T30_design_v0' ,TkAPERecord, connectionString, "", "2023-03-16 15:30:00"] ), ),
    'T33' : ( ','.join( [ 'TrackerAlignmentErrorsExtended_Upgrade2026_T33_design_v0' ,TkAPERecord, connectionString, "", "2023-06-07 21:00:00"] ), ),
}

allTags["TkSurf"] = {
    'T21' : ( ','.join( [ 'TrackerSurfaceDeformations_Upgrade2026_Zero' ,TkSurfRecord, connectionString, "", "2023-03-16 15:30:00"] ), ),
    'T25' : ( ','.join( [ 'TrackerSurfaceDeformations_Upgrade2026_Zero' ,TkSurfRecord, connectionString, "", "2023-03-16 15:30:00"] ), ),
    'T30' : ( ','.join( [ 'TrackerSurfaceDeformations_Upgrade2026_Zero' ,TkSurfRecord, connectionString, "", "2023-03-16 15:30:00"] ), ),
    'T33' : ( ','.join( [ 'TrackerSurfaceDeformations_Upgrade2026_Zero' ,TkSurfRecord, connectionString, "", "2023-03-16 15:30:00"] ), ),
}

##
## Outer Tracker records (to be filled if necessary)
##

'''
allTags["OTLA"] = {
    'T21' : ( ','.join( [ 'SiPhase2OuterTrackerLorentzAngle_v0_mc' ,TrackerLARecord,connectionString, "", "2020-07-19 17:00:00.000"] ), ),  #uH = 0.07/T
    'T25' : ( ','.join( [ 'SiPhase2OuterTrackerLorentzAngle_v0_mc' ,TrackerLARecord,connectionString, "", "2020-07-19 17:00:00.000"] ), ),  #uH = 0.07/T
    'T30' : ( ','.join( [ 'SiPhase2OuterTrackerLorentzAngle_v0_mc' ,TrackerLARecord,connectionString, "", "2020-07-19 17:00:00.000"] ), ),  #uH = 0.07/T
    'T33' : ( ','.join( [ 'SiPhase2OuterTrackerLorentzAngle_v0_mc' ,TrackerLARecord,connectionString, "", "2020-07-19 17:00:00.000"] ), ),  #uH = 0.07/T
}

allTags["SimOTLA"] = {
    'T21' : ( ','.join( [ 'SiPhase2OuterTrackerLorentzAngleSim_v0_mc' ,TrackerSimLARecord,connectionString, "", "2020-07-19 17:00:00.000"] ), ),  #uH = 0.07/T
    'T25' : ( ','.join( [ 'SiPhase2OuterTrackerLorentzAngleSim_v0_mc' ,TrackerSimLARecord,connectionString, "", "2020-07-19 17:00:00.000"] ), ),  #uH = 0.07/T
    'T30' : ( ','.join( [ 'SiPhase2OuterTrackerLorentzAngleSim_v0_mc' ,TrackerSimLARecord,connectionString, "", "2020-07-19 17:00:00.000"] ), ),  #uH = 0.07/T
    'T33' : ( ','.join( [ 'SiPhase2OuterTrackerLorentzAngleSim_v0_mc' ,TrackerSimLARecord,connectionString, "", "2020-07-19 17:00:00.000"] ), ),  #uH = 0.07/T
}
'''

##
## All of the following conditions are not yet in active use, but will be activated in GT along the way
##

'''
allTags["Template2Dnum"] = {
}

allTags["Template2Dden"] = {
}
'''

# list of active tags to be replaced
activeKeys = ["LA", "LAWidth", "SimLA", "LAfromAlignment", "GenError", "Template", "TkAlignment", "TkAPE", "TkSurf"] #,"SimOTLA","OTLA"]

# list of geometries supported
activeDets = ["T21","T25","T30","T33"]
phase2GTs = {}
for det in activeDets:
    appendedTags = ()
    for key in activeKeys:
        if (det in allTags[key]):
            appendedTags += allTags[key][det]
        else :
            pass
    phase2GTs["phase2_realistic_"+det] = ('phase2_realistic', appendedTags)

# method called in autoCond
def autoCondPhase2(autoCond):
    for key,val in phase2GTs.items():
        if len(val)==1 :
           autoCond[key] = ( autoCond[val[0]] )
        else:
           autoCond[key] = ( autoCond[val[0]],) + val[1]

    return autoCond
