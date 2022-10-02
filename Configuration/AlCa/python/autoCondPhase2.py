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
## T21: Phase2 tilted tracker. Outer Tracker (v8.0.0): TBPS update in Layer 1 (facilitate IT insertion) + In all TEDD, update sensors Z inter-spacing. Inner Tracker: (v6.1.5) from previous T17
## (TFPX: Changed sensors spacing within all double-disks + Increased distance between Disks 6 and 7 + TBPX portcards between Disks 6 and 7.)
## T30: Phase2 tilted tracker, exploratory geometry *only to be used in D91 for now*. Outer Tracker (v8.0.1): based on v8.0.0 with updated TB2S spacing. Inner Tracker (v6.4.0): based on v6.1.5 but TFPX with more realistic module positions

#combines in a single dict of dict the tags defined below
allTags={}

allTags["LA"] = {
    'T21' : ( ','.join( [ 'SiPixelLorentzAngle_phase2_T15_v5_mc' ,SiPixelLARecord,connectionString, "", "2020-05-05 20:00:00.000"] ), ),  #uH = 0.053/T (TBPX), uH=0.0/T (TEPX+TFPX)
    'T30' : ( ','.join( [ 'SiPixelLorentzAngle_phase2_IT_v6.4.0_25x100_v1_mc' ,SiPixelLARecord,connectionString, "", "2021-11-22 21:00:00.000"] ), ),  #uH = 0.053/T (TBPX), uH=0.0/T (TEPX+TFPX)
}

allTags["LAWidth"] = {
    'T21' : ( ','.join( [ 'SiPixelLorentzAngle_phase2_T15_mc_forWidthEmpty' ,SiPixelLARecord,connectionString, "forWidth", "2019-11-05 20:00:00.000"] ), ),  # uH=0.0/T (fall-back to offset)
    'T30' : ( ','.join( [ 'SiPixelLorentzAngle_phase2_IT_v6.4.0_25x100_empty_mc' ,SiPixelLARecord,connectionString, "forWidth", "2021-11-29 20:00:00.000"] ), ),  # uH=0.0/T (fall-back to offset)
}

allTags["LAfromAlignment"] = {
    'T21' : ( ','.join( [ 'SiPixelLorentzAngle_phase2_T15_mc_forWidthEmpty' ,SiPixelLARecord,connectionString, "fromAlignment", "2019-11-05 20:00:00.000"] ), ),  # uH=0.0/T (not in use)
    'T30' : ( ','.join( [ 'SiPixelLorentzAngle_phase2_IT_v6.4.0_25x100_empty_mc' ,SiPixelLARecord,connectionString, "fromAlignment", "2021-11-29 20:00:00.000"] ), ),  # uH=0.0/T (fall-back to offset)
}

allTags["SimLA"] = {
    'T21' : ( ','.join( [ 'SiPixelSimLorentzAngle_phase2_T15_v5_mc' ,SiPixelSimLARecord,connectionString, "", "2020-05-05 20:00:00.000"] ), ), #uH = 0.053/T (TBPX), uH=0.0/T (TEPX+TFPX)
    'T30' : ( ','.join( [ 'SiPixelSimLorentzAngle_phase2_IT_v6.4.0_25x100_v1_mc' ,SiPixelSimLARecord,connectionString, "", "2021-12-03 16:00:00.000"] ), ), #uH = 0.053/T (TBPX), uH=0.0/T (TEPX+TFPX)
}

allTags["GenError"] = {
    'T21' : ( ','.join( [ 'SiPixelGenErrorDBObject_phase2_IT_v6.1.5_25x100_v3_mc',SiPixelGenErrorRecord,connectionString, "", "2021-01-27 10:00:00"] ), ),  # cell is 25um (local-x) x 100um (local-y) , VBias=350V
    'T30' : ( ','.join( [ 'SiPixelGenErrorDBObject_phase2_IT_v6.4.0_25x100_v1_mc',SiPixelGenErrorRecord,connectionString, "", "2021-11-22 21:00:00"] ), ),  # cell is 25um (local-x) x 100um (local-y) , VBias=350V
}

allTags["Template"] = {
    'T21' : ( ','.join( [ 'SiPixelTemplateDBObject_phase2_IT_v6.1.5_25x100_v3_mc',SiPixelTemplatesRecord,connectionString, "", "2021-01-27 10:00:00"] ), ),  # cell is 25um (local-x) x 100um (local-y) , VBias=350V
    'T30' : ( ','.join( [ 'SiPixelTemplateDBObject_phase2_IT_v6.4.0_25x100_v1_mc',SiPixelTemplatesRecord,connectionString, "", "2021-11-22 21:00:00"] ), ),  # cell is 25um (local-x) x 100um (local-y) , VBias=350V
}

##
## Outer Tracker records (to be filled if necessary)
##

'''
allTags["OTLA"] = {
    'T21' : ( ','.join( [ 'SiPhase2OuterTrackerLorentzAngle_v0_mc' ,TrackerLARecord,connectionString, "", "2020-07-19 17:00:00.000"] ), ),  #uH = 0.07/T
    'T30' : ( ','.join( [ 'SiPhase2OuterTrackerLorentzAngle_v0_mc' ,TrackerLARecord,connectionString, "", "2020-07-19 17:00:00.000"] ), ),  #uH = 0.07/T
}

allTags["SimOTLA"] = {
    'T21' : ( ','.join( [ 'SiPhase2OuterTrackerLorentzAngleSim_v0_mc' ,TrackerSimLARecord,connectionString, "", "2020-07-19 17:00:00.000"] ), ),  #uH = 0.07/T
    'T30' : ( ','.join( [ 'SiPhase2OuterTrackerLorentzAngleSim_v0_mc' ,TrackerSimLARecord,connectionString, "", "2020-07-19 17:00:00.000"] ), ),  #uH = 0.07/T
}
'''
##
## All of the following conditions are not yet in active use, but will be activated in GT along the way
##

allTags["Template2Dnum"] = {
}

allTags["Template2Dden"] = {
}

# list of active tags to be replaced
activeKeys = ["LA","LAWidth","SimLA","LAfromAlignment","GenError","Template"]#,"SimOTLA","OTLA"]

# list of geometries supported
activeDets = ["T21","T30"]
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
