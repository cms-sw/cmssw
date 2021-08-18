from Configuration.StandardSequences.CondDBESSource_cff import GlobalTag as essource
connectionString = essource.connect.value()

SiPixelLARecord           =   "SiPixelLorentzAngleRcd"                  
SiPixelSimLARecord        =   "SiPixelLorentzAngleSimRcd"   
SiPixelGenErrorRecord     =   "SiPixelGenErrorDBObjectRcd"       
SiPixelTemplatesRecord    =   "SiPixelTemplateDBObjectRcd"       
SiPixel2DTemplatesRecord  =   "SiPixel2DTemplateDBObjectRcd"     
TrackerLARecord           =   "SiPhase2OuterTrackerLorentzAngleRcd"
TrackerSimLARecord        =   "SiPhase2OuterTrackerLorentzAngleSimRcd"

##
## Active geometries: https://github.com/cms-sw/cmssw/blob/master/Configuration/Geometry/README.md
##
## T15: Phase2 tilted tracker (v6.1.6) w/ phase 2 pixel (v6.1.3) (Active geometry: same as T14. Material Budget: major update in IT, gathering info from recent Mechanical designs.)
## T21: Phase2 tilted tracker. Outer Tracker (v8.0.0): TBPS update in Layer 1 (facilitate IT insertion) + In all TEDD, update sensors Z inter-spacing. Inner Tracker: (v6.1.5) from previous T17
## (TFPX: Changed sensors spacing within all double-disks + Increased distance between Disks 6 and 7 + TBPX portcards between Disks 6 and 7.)
## T22: Phase2 tilted tracker. Outer Tracker (v8.0.0): same as T21. Inner Tracker: Based on (v6.1.5) (T21), but with 50x50 pixel aspect ratio everywhere.
## T23: Phase2 tilted tracker. Outer Tracker (v8.0.0): same as T21. Inner Tracker: Based on (v6.1.5) (T21), but with 3D sensors in TBPX L1 + TBPX L2 + TFPX R1.
## T25: Phase2 tilted tracker. Outer Tracker (v8.0.0): same as T24/T21. Inner Tracker (v7.0.2): Based on (v6.1.5) (T24/T21), but with 3D sensors in TBPX L1.
## T26: Phase2 tilted tracker. Outer Tracker (v8.0.0): same as T24/T21. Inner Tracker (v7.0.3): Based on (v6.1.5) (T24/T21), but with 3D sensors in TBPX L1 and 50x50 pixel aspect ratio in TFPX and TEPX.
##

#combines in a single dict of dict the tags defined below
allTags={}

allTags["LA"] = {
    'T15' : ( ','.join( [ 'SiPixelLorentzAngle_phase2_T15_v5_mc' ,SiPixelLARecord,connectionString, "", "2020-05-05 20:00:00.000"] ), ),  #uH = 0.053/T (TBPX), uH=0.0/T (TEPX+TFPX)
    'T21' : ( ','.join( [ 'SiPixelLorentzAngle_phase2_T15_v5_mc' ,SiPixelLARecord,connectionString, "", "2020-05-05 20:00:00.000"] ), ),  #uH = 0.053/T (TBPX), uH=0.0/T (TEPX+TFPX)
    'T22' : ( ','.join( [ 'SiPixelLorentzAngle_phase2_T15_v5_mc' ,SiPixelLARecord,connectionString, "", "2020-05-05 20:00:00.000"] ), ),  #uH = 0.053/T (TBPX), uH=0.0/T (TEPX+TFPX)
    'T23' : ( ','.join( [ 'SiPixelLorentzAngle_phase2_T19_v1_mc' ,SiPixelLARecord,connectionString, "", "2020-02-23 14:00:00.000"] ), ),  #uH = 0.053/T (TBPX L3,L4), uH=0.0/T (TBPX L1,L2, TEPX+TFPX)
    'T25' : ( ','.join( [ 'SiPixelLorentzAngle_phase2_T25_v0_mc' ,SiPixelLARecord,connectionString, "", "2021-03-16 20:00:00.000"] ), ),  #uH = 0.053/T (TBPX L2,L3,L4), uH=0.0/T (TBPX L1 TEPX+TFPX)
    'T26' : ( ','.join( [ 'SiPixelLorentzAngle_phase2_T25_v0_mc' ,SiPixelLARecord,connectionString, "", "2021-03-16 20:00:00.000"] ), ),  #uH = 0.053/T (TBPX L2,L3,L4), uH=0.0/T (TBPX L1 TEPX+TFPX)
    'T27' : ( ','.join( [ 'SiPixelLorentzAngle_phase2_T25_v0_mc' ,SiPixelLARecord,connectionString, "", "2021-03-16 20:00:00.000"] ), ),  #uH = 0.053/T (TBPX L2,L3,L4), uH=0.0/T (TBPX L1 TEPX+TFPX)
}

allTags["LAWidth"] = {
    'T15' : ( ','.join( [ 'SiPixelLorentzAngle_phase2_T15_mc_forWidthEmpty' ,SiPixelLARecord,connectionString, "forWidth", "2019-11-05 20:00:00.000"] ), ),  # uH=0.0/T (fall-back to offset)
    'T21' : ( ','.join( [ 'SiPixelLorentzAngle_phase2_T15_mc_forWidthEmpty' ,SiPixelLARecord,connectionString, "forWidth", "2019-11-05 20:00:00.000"] ), ),  # uH=0.0/T (fall-back to offset)
    'T22' : ( ','.join( [ 'SiPixelLorentzAngle_phase2_T15_mc_forWidthEmpty' ,SiPixelLARecord,connectionString, "forWidth", "2019-11-05 20:00:00.000"] ), ),  # uH=0.0/T (fall-back to offset)
    'T23' : ( ','.join( [ 'SiPixelLorentzAngle_phase2_T19_mc_forWidthEmpty' ,SiPixelLARecord,connectionString, "forWidth", "2020-02-23 14:00:00.000"] ), ),  # uH=0.0/T (fall-back to offset)
    'T25' : ( ','.join( [ 'SiPixelLorentzAngle_phase2_T19_mc_forWidthEmpty' ,SiPixelLARecord,connectionString, "forWidth", "2020-02-23 14:00:00.000"] ), ),  # uH=0.0/T (fall-back to offset)
    'T26' : ( ','.join( [ 'SiPixelLorentzAngle_phase2_T19_mc_forWidthEmpty' ,SiPixelLARecord,connectionString, "forWidth", "2020-02-23 14:00:00.000"] ), ),  # uH=0.0/T (fall-back to offset)
    'T27' : ( ','.join( [ 'SiPixelLorentzAngle_phase2_T19_mc_forWidthEmpty' ,SiPixelLARecord,connectionString, "forWidth", "2020-02-23 14:00:00.000"] ), ),  # uH=0.0/T (fall-back to offset)
}

allTags["LAfromAlignment"] = {
    'T15' : ( ','.join( [ 'SiPixelLorentzAngle_phase2_T15_mc_forWidthEmpty' ,SiPixelLARecord,connectionString, "fromAlignment", "2019-11-05 20:00:00.000"] ), ),  # uH=0.0/T (not in use)
    'T21' : ( ','.join( [ 'SiPixelLorentzAngle_phase2_T15_mc_forWidthEmpty' ,SiPixelLARecord,connectionString, "fromAlignment", "2019-11-05 20:00:00.000"] ), ),  # uH=0.0/T (not in use)
    'T22' : ( ','.join( [ 'SiPixelLorentzAngle_phase2_T15_mc_forWidthEmpty' ,SiPixelLARecord,connectionString, "fromAlignment", "2019-11-05 20:00:00.000"] ), ),  # uH=0.0/T (not in use)
    'T23' : ( ','.join( [ 'SiPixelLorentzAngle_phase2_T19_mc_forWidthEmpty' ,SiPixelLARecord,connectionString, "fromAlignment", "2020-02-23 14:00:00.000"] ), ),  # uH=0.0/T (not in use)
    'T25' : ( ','.join( [ 'SiPixelLorentzAngle_phase2_T19_mc_forWidthEmpty' ,SiPixelLARecord,connectionString, "fromAlignment", "2020-02-23 14:00:00.000"] ), ),  # uH=0.0/T (not in use)
    'T26' : ( ','.join( [ 'SiPixelLorentzAngle_phase2_T19_mc_forWidthEmpty' ,SiPixelLARecord,connectionString, "fromAlignment", "2020-02-23 14:00:00.000"] ), ),  # uH=0.0/T (not in use)
    'T27' : ( ','.join( [ 'SiPixelLorentzAngle_phase2_T19_mc_forWidthEmpty' ,SiPixelLARecord,connectionString, "fromAlignment", "2020-02-23 14:00:00.000"] ), ),  # uH=0.0/T (not in use)
}

allTags["SimLA"] = {
    'T15' : ( ','.join( [ 'SiPixelSimLorentzAngle_phase2_T15_v5_mc' ,SiPixelSimLARecord,connectionString, "", "2020-05-05 20:00:00.000"] ), ), #uH = 0.053/T (TBPX), uH=0.0/T (TEPX+TFPX)
    'T21' : ( ','.join( [ 'SiPixelSimLorentzAngle_phase2_T15_v5_mc' ,SiPixelSimLARecord,connectionString, "", "2020-05-05 20:00:00.000"] ), ), #uH = 0.053/T (TBPX), uH=0.0/T (TEPX+TFPX)
    'T22' : ( ','.join( [ 'SiPixelSimLorentzAngle_phase2_T15_v5_mc' ,SiPixelSimLARecord,connectionString, "", "2020-05-05 20:00:00.000"] ), ), #uH = 0.053/T (TBPX), uH=0.0/T (TEPX+TFPX)
    'T23' : ( ','.join( [ 'SiPixelSimLorentzAngle_phase2_T19_v1_mc' ,SiPixelSimLARecord,connectionString, "", "2020-02-23 14:00:00.000"] ), ), #uH = 0.053/T (TBPX L3,L4), uH=0.0/T (TBPX L1,L2, TEPX+TFPX)
    'T25' : ( ','.join( [ 'SiPixelSimLorentzAngle_phase2_T25_v0_mc' ,SiPixelSimLARecord,connectionString, "", "2021-03-16 20:00:00.000"] ), ), #uH = 0.053/T (TBPX L2,L3,L4), uH=0.0/T (TBPX L1 TEPX+TFPX)
    'T26' : ( ','.join( [ 'SiPixelSimLorentzAngle_phase2_T25_v0_mc' ,SiPixelSimLARecord,connectionString, "", "2021-03-16 20:00:00.000"] ), ), #uH = 0.053/T (TBPX L2,L3,L4), uH=0.0/T (TBPX L1 TEPX+TFPX)
    'T27' : ( ','.join( [ 'SiPixelSimLorentzAngle_phase2_T25_v0_mc' ,SiPixelSimLARecord,connectionString, "", "2021-03-16 20:00:00.000"] ), ), #uH = 0.053/T (TBPX L2,L3,L4), uH=0.0/T (TBPX L1 TEPX+TFPX)
}

allTags["GenError"] = {
    'T15' : ( ','.join( [ 'SiPixelGenErrorDBObject_phase2_IT_v6.1.5_25x100_v3_mc',SiPixelGenErrorRecord,connectionString, "", "2021-01-27 10:00:00"] ), ),  # cell is 25um (local-x) x 100um (local-y) , VBias=350V
    'T21' : ( ','.join( [ 'SiPixelGenErrorDBObject_phase2_IT_v6.1.5_25x100_v3_mc',SiPixelGenErrorRecord,connectionString, "", "2021-01-27 10:00:00"] ), ),  # cell is 25um (local-x) x 100um (local-y) , VBias=350V
    'T22' : ( ','.join( [ 'SiPixelGenErrorDBObject_phase2_IT_v6.1.5_50x50_v5_mc' ,SiPixelGenErrorRecord,connectionString, "", "2021-01-27 10:00:00"] ), ),  # cell is 50um (local-x) x 50um (local-y) , VBias=350V
    'T23' : ( ','.join( [ 'SiPixelGenErrorDBObject_phase2_IT_v7.0.0_25x100_v2_mc' ,SiPixelGenErrorRecord,connectionString, "", "2021-04-17 20:00:00"] ), ),  # cell is 25um (local-x) x 100um (local-y) , VBias=350V, 3D pixels in TBPX L1+L2 and TFPX R1
    'T25' : ( ','.join( [ 'SiPixelGenErrorDBObject_phase2_IT_v7.0.2_25x100_v2_mc' ,SiPixelGenErrorRecord,connectionString, "", "2021-04-17 20:00:00"] ), ),  # cell is 25um (local-x) x 100um (local-y) , VBias=350V, 3D pixels in TBPX L1
    'T26' : ( ','.join( [ 'SiPixelGenErrorDBObject_phase2_IT_v7.0.3_mixed_v2_mc' ,SiPixelGenErrorRecord,connectionString, "", "2021-04-17 20:00:00"] ), ),  # TBPX cells are 25um (local-x) x 100um (local-y), TFPX TEPX 50 um x 50 um , VBias=350V, 3D pixels in TBPX L1
}

allTags["Template"] = {
    'T15' : ( ','.join( [ 'SiPixelTemplateDBObject_phase2_IT_v6.1.5_25x100_v3_mc',SiPixelTemplatesRecord,connectionString, "", "2021-01-27 10:00:00"] ), ),  # cell is 25um (local-x) x 100um (local-y) , VBias=350V
    'T21' : ( ','.join( [ 'SiPixelTemplateDBObject_phase2_IT_v6.1.5_25x100_v3_mc',SiPixelTemplatesRecord,connectionString, "", "2021-01-27 10:00:00"] ), ),  # cell is 25um (local-x) x 100um (local-y) , VBias=350V
    'T22' : ( ','.join( [ 'SiPixelTemplateDBObject_phase2_IT_v6.1.5_50x50_v5_mc' ,SiPixelTemplatesRecord,connectionString, "", "2021-01-27 10:00:00"] ), ),  # cell is 50um (local-x) x 50um (local-y) , VBias=350V
    'T23' : ( ','.join( [ 'SiPixelTemplateDBObject_phase2_IT_v7.0.0_25x100_v2_mc' ,SiPixelTemplatesRecord,connectionString, "", "2021-04-17 20:00:00"] ), ),  # cell is 25um (local-x) x 100um (local-y) , VBias=350V, 3D pixels in TBPX L1+L2 and TFPX R1
    'T25' : ( ','.join( [ 'SiPixelTemplateDBObject_phase2_IT_v7.0.2_25x100_v2_mc' ,SiPixelTemplatesRecord,connectionString, "", "2021-04-17 20:00:00"] ), ),  # cell is 25um (local-x) x 100um (local-y) , VBias=350V, 3D pixels in TBPX L1
    'T26' : ( ','.join( [ 'SiPixelTemplateDBObject_phase2_IT_v7.0.3_mixed_v2_mc' ,SiPixelTemplatesRecord,connectionString, "", "2021-04-17 20:00:00"] ), ),  # TBPX cells are 25um (local-x) x 100um (local-y), TFPX TEPX 50 um x 50 um , VBias=350V, 3D pixels in TBPX L1
}

##
## Outer Tracker records (to be filled if necessary)
##

'''
allTags["OTLA"] = {
    'T15' : ( ','.join( [ 'SiPhase2OuterTrackerLorentzAngle_v0_mc' ,TrackerLARecord,connectionString, "", "2020-07-19 17:00:00.000"] ), ),  #uH = 0.07/T
    'T21' : ( ','.join( [ 'SiPhase2OuterTrackerLorentzAngle_v0_mc' ,TrackerLARecord,connectionString, "", "2020-07-19 17:00:00.000"] ), ),  #uH = 0.07/T
    'T22' : ( ','.join( [ 'SiPhase2OuterTrackerLorentzAngle_v0_mc' ,TrackerLARecord,connectionString, "", "2020-07-19 17:00:00.000"] ), ),  #uH = 0.07/T
    'T23' : ( ','.join( [ 'SiPhase2OuterTrackerLorentzAngle_v0_mc' ,TrackerLARecord,connectionString, "", "2020-07-19 17:00:00.000"] ), ),  #uH = 0.07/T
    'T25' : ( ','.join( [ 'SiPhase2OuterTrackerLorentzAngle_v0_mc' ,TrackerLARecord,connectionString, "", "2020-07-19 17:00:00.000"] ), ),  #uH = 0.07/T
    'T26' : ( ','.join( [ 'SiPhase2OuterTrackerLorentzAngle_v0_mc' ,TrackerLARecord,connectionString, "", "2020-07-19 17:00:00.000"] ), ),  #uH = 0.07/T
    'T27' : ( ','.join( [ 'SiPhase2OuterTrackerLorentzAngle_v0_mc' ,TrackerLARecord,connectionString, "", "2020-07-19 17:00:00.000"] ), ),  #uH = 0.07/T
}

allTags["SimOTLA"] = {
    'T15' : ( ','.join( [ 'SiPhase2OuterTrackerLorentzAngleSim_v0_mc' ,TrackerSimLARecord,connectionString, "", "2020-07-19 17:00:00.000"] ), ),  #uH = 0.07/T
    'T21' : ( ','.join( [ 'SiPhase2OuterTrackerLorentzAngleSim_v0_mc' ,TrackerSimLARecord,connectionString, "", "2020-07-19 17:00:00.000"] ), ),  #uH = 0.07/T
    'T22' : ( ','.join( [ 'SiPhase2OuterTrackerLorentzAngleSim_v0_mc' ,TrackerSimLARecord,connectionString, "", "2020-07-19 17:00:00.000"] ), ),  #uH = 0.07/T
    'T23' : ( ','.join( [ 'SiPhase2OuterTrackerLorentzAngleSim_v0_mc' ,TrackerSimLARecord,connectionString, "", "2020-07-19 17:00:00.000"] ), ),  #uH = 0.07/T
    'T25' : ( ','.join( [ 'SiPhase2OuterTrackerLorentzAngleSim_v0_mc' ,TrackerSimLARecord,connectionString, "", "2020-07-19 17:00:00.000"] ), ),  #uH = 0.07/T
    'T26' : ( ','.join( [ 'SiPhase2OuterTrackerLorentzAngleSim_v0_mc' ,TrackerSimLARecord,connectionString, "", "2020-07-19 17:00:00.000"] ), ),  #uH = 0.07/T
    'T27' : ( ','.join( [ 'SiPhase2OuterTrackerLorentzAngleSim_v0_mc' ,TrackerSimLARecord,connectionString, "", "2020-07-19 17:00:00.000"] ), ),  #uH = 0.07/T
}
'''
##
## All of the following conditions are not yet in active use, but will be activated in GT along the way
##

allTags["Template2Dnum"] = {
    'T15' : ( ','.join( [ 'SiPixel2DTemplateDBObject_phase2_T15_v0_num' ,SiPixel2DTemplatesRecord,connectionString, "numerator", "2019-07-15 12:00:00.000"] ), ),
}

allTags["Template2Dden"] = {
    'T15' : ( ','.join( [ 'SiPixel2DTemplateDBObject_phase2_T15_v0_den' ,SiPixel2DTemplatesRecord,connectionString, "denominator", "2019-07-15 12:00:00.000"] ), ),
}

# list of active tags to be replaced
activeKeys = ["LA","LAWidth","SimLA","LAfromAlignment","GenError","Template"]#,"SimOTLA","OTLA"]

# list of geometries supported
activeDets = ["T15","T21","T22","T23","T25","T26","T27"]
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
