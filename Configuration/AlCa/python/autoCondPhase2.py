import six
from Configuration.StandardSequences.CondDBESSource_cff import GlobalTag as essource
connectionString = essource.connect.value()

SiPixelLARecord           =   "SiPixelLorentzAngleRcd"                  
SiPixelSimLARecord        =   "SiPixelLorentzAngleSimRcd"   
SiPixelGenErrorRecord     =   "SiPixelGenErrorDBObjectRcd"       
SiPixelTemplatesRecord    =   "SiPixelTemplateDBObjectRcd"       
SiPixel2DTemplatesRecord  =   "SiPixel2DTemplateDBObjectRcd"     

##
## Active geometries: https://github.com/cms-sw/cmssw/blob/master/Configuration/Geometry/README.md
##
## T15: Phase2 tilted tracker (v6.1.6) w/ phase 2 pixel (v6.1.3) (Active geometry: same as T14. Material Budget: major update in IT, gathering info from recent Mechanical designs.)
## T17: Phase2 tilted tracker (v6.1.6) w/ phase 2 pixel (v6.1.5) TFPX: Changed sensors spacing within all double-disks + Increased distance between Disks 6 and 7 + Put TBPX portcards between Disks 6 and 7.
## T19: Phase2 tilted tracker (v6.1.6) w/ phase 2 pixel (v7.0.0) Inner Tracker description with 3D sensors in TBPX L1 + TBPX L2 + TFPX R1.
## T20: Phase2 tilted tracker (v6.1.6) Outer Tracker: All sensors 200 um -> 290 um + Update in Module MB + PS modules: s-sensor 164 um longer + Major update in OTST MB. Inner Tracker: (v6.1.5) from T17 is called.
##

#combines in a single dict of dict the tags defined below
allTags={}

allTags["LA"] = {
    'T15' : ( ','.join( [ 'SiPixelLorentzAngle_phase2_T15_v5_mc' ,SiPixelLARecord,connectionString, "", "2020-05-05 20:00:00.000"] ), ),  #uH = 0.053/T (TBPX), uH=0.0/T (TEPX+TFPX)
    'T17' : ( ','.join( [ 'SiPixelLorentzAngle_phase2_T15_v5_mc' ,SiPixelLARecord,connectionString, "", "2020-05-05 20:00:00.000"] ), ),  #uH = 0.053/T (TBPX), uH=0.0/T (TEPX+TFPX)
    'T19' : ( ','.join( [ 'SiPixelLorentzAngle_phase2_T19_v1_mc' ,SiPixelLARecord,connectionString, "", "2020-02-23 14:00:00.000"] ), ),  #uH = 0.053/T (TBPX L3,L4), uH=0.0/T (TBPX L1,L2, TEPX+TFPX)
    'T20' : ( ','.join( [ 'SiPixelLorentzAngle_phase2_T15_v5_mc' ,SiPixelLARecord,connectionString, "", "2020-05-05 20:00:00.000"] ), ),  #uH = 0.053/T (TBPX), uH=0.0/T (TEPX+TFPX)
}

allTags["LAWidth"] = {
    'T15' : ( ','.join( [ 'SiPixelLorentzAngle_phase2_T15_mc_forWidthEmpty' ,SiPixelLARecord,connectionString, "forWidth", "2019-11-05 20:00:00.000"] ), ),  # uH=0.0/T (fall-back to offset)
    'T17' : ( ','.join( [ 'SiPixelLorentzAngle_phase2_T15_mc_forWidthEmpty' ,SiPixelLARecord,connectionString, "forWidth", "2019-11-05 20:00:00.000"] ), ),  # uH=0.0/T (fall-back to offset)
    'T19' : ( ','.join( [ 'SiPixelLorentzAngle_phase2_T19_mc_forWidthEmpty' ,SiPixelLARecord,connectionString, "forWidth", "2020-02-23 14:00:00.000"] ), ),  # uH=0.0/T (fall-back to offset)
    'T20' : ( ','.join( [ 'SiPixelLorentzAngle_phase2_T15_mc_forWidthEmpty' ,SiPixelLARecord,connectionString, "forWidth", "2019-11-05 20:00:00.000"] ), ),  # uH=0.0/T (fall-back to offset)
}

allTags["LAfromAlignment"] = {
    'T15' : ( ','.join( [ 'SiPixelLorentzAngle_phase2_T15_mc_forWidthEmpty' ,SiPixelLARecord,connectionString, "fromAlignment", "2019-11-05 20:00:00.000"] ), ),  # uH=0.0/T (not in use)
    'T17' : ( ','.join( [ 'SiPixelLorentzAngle_phase2_T15_mc_forWidthEmpty' ,SiPixelLARecord,connectionString, "fromAlignment", "2019-11-05 20:00:00.000"] ), ),  # uH=0.0/T (not in use)
    'T19' : ( ','.join( [ 'SiPixelLorentzAngle_phase2_T19_mc_forWidthEmpty' ,SiPixelLARecord,connectionString, "fromAlignment", "2020-02-23 14:00:00.000"] ), ),  # uH=0.0/T (not in use)
    'T20' : ( ','.join( [ 'SiPixelLorentzAngle_phase2_T15_mc_forWidthEmpty' ,SiPixelLARecord,connectionString, "fromAlignment", "2019-11-05 20:00:00.000"] ), ),  # uH=0.0/T (not in use)
}

allTags["SimLA"] = {
    'T15' : ( ','.join( [ 'SiPixelSimLorentzAngle_phase2_T15_v5_mc' ,SiPixelSimLARecord,connectionString, "", "2020-05-05 20:00:00.000"] ), ), #uH = 0.053/T (TBPX), uH=0.0/T (TEPX+TFPX)
    'T17' : ( ','.join( [ 'SiPixelSimLorentzAngle_phase2_T15_v5_mc' ,SiPixelSimLARecord,connectionString, "", "2020-05-05 20:00:00.000"] ), ), #uH = 0.053/T (TBPX), uH=0.0/T (TEPX+TFPX)
    'T19' : ( ','.join( [ 'SiPixelSimLorentzAngle_phase2_T19_v1_mc' ,SiPixelSimLARecord,connectionString, "", "2020-02-23 14:00:00.000"] ), ), #uH = 0.053/T (TBPX L3,L4), uH=0.0/T (TBPX L1,L2, TEPX+TFPX)
    'T20' : ( ','.join( [ 'SiPixelSimLorentzAngle_phase2_T15_v5_mc' ,SiPixelSimLARecord,connectionString, "", "2020-05-05 20:00:00.000"] ), ), #uH = 0.053/T (TBPX), uH=0.0/T (TEPX+TFPX)
}

allTags["GenError"] = {
    'T15' : ( ','.join( [ 'SiPixelGenErrorDBObject_phase2_T15_v5_mc' ,SiPixelGenErrorRecord,connectionString, "", "2020-05-02 23:00:00.000"] ), ),  # cell is 25um (local-x) x 100um (local-y) , VBias=350V
    'T17' : ( ','.join( [ 'SiPixelGenErrorDBObject_phase2_T15_v5_mc' ,SiPixelGenErrorRecord,connectionString, "", "2020-05-02 23:00:00.000"] ), ),  # cell is 25um (local-x) x 100um (local-y) , VBias=350V
    'T20' : ( ','.join( [ 'SiPixelGenErrorDBObject_phase2_T15_v5_mc' ,SiPixelGenErrorRecord,connectionString, "", "2020-05-02 23:00:00.000"] ), ),  # cell is 25um (local-x) x 100um (local-y) , VBias=350V
}

allTags["Template"] = {
    'T15' : ( ','.join( [ 'SiPixelTemplateDBObject_phase2_T15_v5_mc',SiPixelTemplatesRecord,connectionString, "", "2020-05-02 23:00:00.000"] ), ),  # cell is 25um (local-x) x 100um (local-y) , VBias=350V
    'T17' : ( ','.join( [ 'SiPixelTemplateDBObject_phase2_T15_v5_mc',SiPixelTemplatesRecord,connectionString, "", "2020-05-02 23:00:00.000"] ), ),  # cell is 25um (local-x) x 100um (local-y) , VBias=350V
    'T20' : ( ','.join( [ 'SiPixelTemplateDBObject_phase2_T15_v5_mc',SiPixelTemplatesRecord,connectionString, "", "2020-05-02 23:00:00.000"] ), ),  # cell is 25um (local-x) x 100um (local-y) , VBias=350V
}

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
activeKeys = ["LA","LAWidth","SimLA","LAfromAlignment","GenError","Template"]

# list of geometries supported
activeDets = ["T15","T17","T19","T20"]
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
    for key,val in six.iteritems(phase2GTs):
        if len(val)==1 :
           autoCond[key] = ( autoCond[val[0]] )
        else:
           autoCond[key] = ( autoCond[val[0]],) + val[1]

    return autoCond
