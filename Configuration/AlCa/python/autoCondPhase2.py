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

#combines in a single dict of dict the tags defined below
allTags={}

allTags["LA"] = {
    'T35' : ( ','.join( [ 'SiPixelLorentzAngle_phase2_IT_v7.1.1_25x100_v1_mc' ,SiPixelLARecord,connectionString, "", "2023-05-16 20:00:00.000"] ), ),  #uH = 0.053/T (TBPX L2,L3,L4), uH=0.0/T (TBPX L1 TEPX+TFPX)
    'T36' : ( ','.join( [ 'SiPixelLorentzAngle_phase2_IT_v7.4.1_25x100_v2_mc' ,SiPixelLARecord,connectionString, "", "2024-04-08 16:00:00.000"] ), ),  #uH = 0.053/T (TBPX L2,L3,L4), uH=0.0/T (TBPX L1 TEPX+TFPX)
    'T37' : ( ','.join( [ 'SiPixelLorentzAngle_phase2_IT_v7.4.1_25x100_v2_mc' ,SiPixelLARecord,connectionString, "", "2024-04-08 16:00:00.000"] ), ),  #uH = 0.053/T (TBPX L2,L3,L4), uH=0.0/T (TBPX L1 TEPX+TFPX)
    'T38' : ( ','.join( [ 'SiPixelLorentzAngle_phase2_IT_v7.4.1_25x100_v2_mc' ,SiPixelLARecord,connectionString, "", "2024-04-08 16:00:00.000"] ), ),  #uH = 0.053/T (TBPX L2,L3,L4), uH=0.0/T (TBPX L1 TEPX+TFPX)
}

allTags["LAWidth"] = {
    'T35' : ( ','.join( [ 'SiPixelLorentzAngle_phase2_IT_v7.1.1_25x100_empty_mc' ,SiPixelLARecord,connectionString, "forWidth", "2023-12-02 15:55:00.000"] ), ),  # uH=0.0/T (fall-back to offset)
    'T36' : ( ','.join( [ 'SiPixelLorentzAngle_phase2_IT_v7.4.1_25x100_empty_v2_mc' ,SiPixelLARecord,connectionString, "forWidth", "2024-04-08 16:00:00.000"] ), ),  # uH=0.0/T (fall-back to offset)
    'T37' : ( ','.join( [ 'SiPixelLorentzAngle_phase2_IT_v7.4.1_25x100_empty_v2_mc' ,SiPixelLARecord,connectionString, "forWidth", "2024-04-08 16:00:00.000"] ), ),  # uH=0.0/T (fall-back to offset)
    'T38' : ( ','.join( [ 'SiPixelLorentzAngle_phase2_IT_v7.4.1_25x100_empty_v2_mc' ,SiPixelLARecord,connectionString, "forWidth", "2024-04-08 16:00:00.000"] ), ),  # uH=0.0/T (fall-back to offset)
}

allTags["LAfromAlignment"] = {
    'T35' : ( ','.join( [ 'SiPixelLorentzAngle_phase2_IT_v7.1.1_25x100_empty_mc' ,SiPixelLARecord,connectionString, "fromAlignment", "2023-12-02 15:55:00.000"] ), ),  # uH=0.0/T (fall-back to offset)
    'T36' : ( ','.join( [ 'SiPixelLorentzAngle_phase2_IT_v7.4.1_25x100_empty_v2_mc' ,SiPixelLARecord,connectionString, "fromAlignment", "2024-04-08 16:00:00.000"] ), ),  # uH=0.0/T (fall-back to offset)
    'T37' : ( ','.join( [ 'SiPixelLorentzAngle_phase2_IT_v7.4.1_25x100_empty_v2_mc' ,SiPixelLARecord,connectionString, "fromAlignment", "2024-04-08 16:00:00.000"] ), ),  # uH=0.0/T (fall-back to offset)
    'T38' : ( ','.join( [ 'SiPixelLorentzAngle_phase2_IT_v7.4.1_25x100_empty_v2_mc' ,SiPixelLARecord,connectionString, "fromAlignment", "2024-04-08 16:00:00.000"] ), ),  # uH=0.0/T (fall-back to offset)
}

allTags["SimLA"] = {
    'T35' : ( ','.join( [ 'SiPixelSimLorentzAngle_phase2_IT_v7.1.1_25x100_v1_mc' ,SiPixelSimLARecord,connectionString, "", "2024-03-07 21:00:00.000"] ), ),#uH = 0.053/T (TBPX L2,L3,L4), uH=0.0/T (TBPX L1 TEPX+TFPX)
    'T36' : ( ','.join( [ 'SiPixelSimLorentzAngle_phase2_IT_v7.4.1_25x100_v2_mc' ,SiPixelSimLARecord,connectionString, "", "2024-04-08 16:00:00.000"] ), ), #uH = 0.053/T (TBPX  L2,L3,L4), uH=0.0/T (TBPX L1 TEPX+TFPX)
    'T37' : ( ','.join( [ 'SiPixelSimLorentzAngle_phase2_IT_v7.4.1_25x100_v2_mc' ,SiPixelSimLARecord,connectionString, "", "2024-04-08 16:00:00.000"] ), ), #uH = 0.053/T (TBPX  L2,L3,L4), uH=0.0/T (TBPX L1 TEPX+TFPX)
    'T38' : ( ','.join( [ 'SiPixelSimLorentzAngle_phase2_IT_v7.4.1_25x100_v2_mc' ,SiPixelSimLARecord,connectionString, "", "2024-04-08 16:00:00.000"] ), ), #uH = 0.053/T (TBPX  L2,L3,L4), uH=0.0/T (TBPX L1 TEPX+TFPX)
}

allTags["GenError"] = {
    'T35' : ( ','.join( [ 'SiPixelGenErrorDBObject_phase2_IT_v7.1.1_25x100_v1_mc' ,SiPixelGenErrorRecord,connectionString, "", "2023-05-16 20:00:00"] ), ),  # cell is 25um (local-x) x 100um (local-y) , VBias=350V, 3D pixels in TBPX L1
    'T36' : ( ','.join( [ 'SiPixelGenErrorDBObject_phase2_IT_v7.4.1_25x100_v2_mc' ,SiPixelGenErrorRecord,connectionString, "", "2024-04-08 16:00:00.000"] ), ),  # cell is 25um (local-x) x 100um (local-y) , VBias=350V, 3D pixels in TBPX L1
    'T37' : ( ','.join( [ 'SiPixelGenErrorDBObject_phase2_IT_v7.4.1_25x100_v2_mc' ,SiPixelGenErrorRecord,connectionString, "", "2024-04-08 16:00:00.000"] ), ),  # cell is 25um (local-x) x 100um (local-y) , VBias=350V, 3D pixels in TBPX L1
    'T38' : ( ','.join( [ 'SiPixelGenErrorDBObject_phase2_IT_v7.4.1_25x100_v2_mc' ,SiPixelGenErrorRecord,connectionString, "", "2024-04-08 16:00:00.000"] ), ),  # cell is 25um (local-x) x 100um (local-y) , VBias=350V, 3D pixels in TBPX L1
}

allTags["Template"] = {
    'T35' : ( ','.join( [ 'SiPixelTemplateDBObject_phase2_IT_v7.1.1_25x100_v1_mc' ,SiPixelTemplatesRecord,connectionString, "", "2023-05-16 20:00:00"] ), ),  # cell is 25um (local-x) x 100um (local-y) , VBias=350V, 3D pixels in TBPX L1
    'T36' : ( ','.join( [ 'SiPixelTemplateDBObject_phase2_IT_v7.4.1_25x100_v2_mc' ,SiPixelTemplatesRecord,connectionString, "", "2024-04-08 16:00:00.000"] ), ),  # cell is 25um (local-x) x 100um (local-y) , VBias=350V, 3D pixels in TBPX L1
    'T37' : ( ','.join( [ 'SiPixelTemplateDBObject_phase2_IT_v7.4.1_25x100_v2_mc' ,SiPixelTemplatesRecord,connectionString, "", "2024-04-08 16:00:00.000"] ), ),  # cell is 25um (local-x) x 100um (local-y) , VBias=350V, 3D pixels in TBPX L1
    'T38' : ( ','.join( [ 'SiPixelTemplateDBObject_phase2_IT_v7.4.1_25x100_v2_mc' ,SiPixelTemplatesRecord,connectionString, "", "2024-04-08 16:00:00.000"] ), ),  # cell is 25um (local-x) x 100um (local-y) , VBias=350V, 3D pixels in TBPX L1
}

allTags["TkAlignment"] = {
    'T35' : ( ','.join( [ 'TrackerAlignment_Upgrade2026_T33_design_v1' ,TkAlRecord, connectionString, "", "2024-09-12 15:37:00"] ), ),
    'T36' : ( ','.join( [ 'TrackerAlignment_Upgrade2026_T36_design_v1' ,TkAlRecord, connectionString, "", "2024-09-12 15:37:00"] ), ),
    'T37' : ( ','.join( [ 'TrackerAlignment_Upgrade2026_T37_design_v1' ,TkAlRecord, connectionString, "", "2024-09-12 15:37:00"] ), ),
    'T38' : ( ','.join( [ 'TrackerAlignment_Upgrade2026_T38_design_v1' ,TkAlRecord, connectionString, "", "2024-09-12 15:37:00"] ), ),
}

allTags["TkAPE"] = {
    'T35' : ( ','.join( [ 'TrackerAlignmentErrorsExtended_Upgrade2026_T33_design_v1' ,TkAPERecord, connectionString, "", "2024-09-12 15:37:00"] ), ),
    'T36' : ( ','.join( [ 'TrackerAlignmentErrorsExtended_Upgrade2026_T36_design_v1' ,TkAPERecord, connectionString, "", "2024-09-12 15:37:00"] ), ),
    'T37' : ( ','.join( [ 'TrackerAlignmentErrorsExtended_Upgrade2026_T36_design_v1' ,TkAPERecord, connectionString, "", "2024-09-12 15:37:00"] ), ),
    'T38' : ( ','.join( [ 'TrackerAlignmentErrorsExtended_Upgrade2026_T36_design_v1' ,TkAPERecord, connectionString, "", "2024-09-12 15:37:00"] ), ),
}

allTags["TkSurf"] = {
    'T35' : ( ','.join( [ 'TrackerSurfaceDeformations_Upgrade2026_Zero' ,TkSurfRecord, connectionString, "", "2023-03-16 15:30:00"] ), ),
    'T36' : ( ','.join( [ 'TrackerSurfaceDeformations_Upgrade2026_Zero' ,TkSurfRecord, connectionString, "", "2023-04-14 15:00:00"] ), ),
    'T37' : ( ','.join( [ 'TrackerSurfaceDeformations_Upgrade2026_Zero' ,TkSurfRecord, connectionString, "", "2023-04-14 15:00:00"] ), ),
    'T38' : ( ','.join( [ 'TrackerSurfaceDeformations_Upgrade2026_Zero' ,TkSurfRecord, connectionString, "", "2023-04-14 15:00:00"] ), ),
}

##
## Outer Tracker records (to be filled if necessary)
##

'''
allTags["OTLA"] = {
    'T35' : ( ','.join( [ 'SiPhase2OuterTrackerLorentzAngle_v0_mc' ,TrackerLARecord,connectionString, "", "2020-07-19 17:00:00.000"] ), ),  #uH = 0.07/T
}

allTags["SimOTLA"] = {
    'T35' : ( ','.join( [ 'SiPhase2OuterTrackerLorentzAngleSim_v0_mc' ,TrackerSimLARecord,connectionString, "", "2020-07-19 17:00:00.000"] ), ),  #uH = 0.07/T
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
activeDets = ["T35","T36","T37","T38"]
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
