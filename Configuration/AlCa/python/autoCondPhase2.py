import six
from Configuration.StandardSequences.CondDBESSource_cff import GlobalTag as essource
connectionString = essource.connect.value()

SiPixelLARecord           =   "SiPixelLorentzAngleRcd"                  
SiPixelSimLARecord        =   "SiPixelLorentzAngleSimRcd"   
SiPixelGenErrorRecord     =   "SiPixelGenErrorDBObjectRcd"       
SiPixelTemplatesRecord    =   "SiPixelTemplateDBObjectRcd"       
SiPixel2DTemplatesRecord  =   "SiPixel2DTemplateDBObjectRcd"     

#combines in a single dict of dict the tags defined below
allTags={}


allTags["LA"] = {
    'T6'  : ( ','.join( [ 'SiPixelLorentzAngle_phase2_T6_v2_mc'  ,SiPixelLARecord,connectionString, "", "2019-11-05 20:00:00.000"] ), ),  #uH = 0.106/T (TBPX), uH=0.0/T (TEPX+TFPX)
    'T14' : ( ','.join( [ 'SiPixelLorentzAngle_phase2_T14_v2_mc' ,SiPixelLARecord,connectionString, "", "2019-11-05 20:00:00.000"] ), ),  #uH = 0.106/T (TBPX), uH=0.0/T (TEPX+TFPX)
    'T15' : ( ','.join( [ 'SiPixelLorentzAngle_phase2_T15_v2_mc' ,SiPixelLARecord,connectionString, "", "2019-11-05 20:00:00.000"] ), ),  #uH = 0.053/T (TBPX), uH=0.0/T (TEPX+TFPX)
}

allTags["LAWidth"] = {
    'T6'  : ( ','.join( [ 'SiPixelLorentzAngle_phase2_T6_mc_forWidthEmpty'  ,SiPixelLARecord,connectionString, "forWidth", "2019-11-05 20:00:00.000"] ), ),  # uH=0.0/T (fall-back to offset)
    'T14' : ( ','.join( [ 'SiPixelLorentzAngle_phase2_T14_mc_forWidthEmpty' ,SiPixelLARecord,connectionString, "forWidth", "2019-11-05 20:00:00.000"] ), ),  # uH=0.0/T (fall-back to offset)
    'T15' : ( ','.join( [ 'SiPixelLorentzAngle_phase2_T15_mc_forWidthEmpty' ,SiPixelLARecord,connectionString, "forWidth", "2019-11-05 20:00:00.000"] ), ),  # uH=0.0/T (fall-back to offset)
}

allTags["LAfromAlignment"] = {
    'T6'  : ( ','.join( [ 'SiPixelLorentzAngle_phase2_T6_mc_forWidthEmpty'  ,SiPixelLARecord,connectionString, "fromAlignment", "2019-11-05 20:00:00.000"] ), ),  # uH=0.0/T (not in use)
    'T14' : ( ','.join( [ 'SiPixelLorentzAngle_phase2_T14_mc_forWidthEmpty' ,SiPixelLARecord,connectionString, "fromAlignment", "2019-11-05 20:00:00.000"] ), ),  # uH=0.0/T (not in use)
    'T15' : ( ','.join( [ 'SiPixelLorentzAngle_phase2_T15_mc_forWidthEmpty' ,SiPixelLARecord,connectionString, "fromAlignment", "2019-11-05 20:00:00.000"] ), ),  # uH=0.0/T (not in use)
}

allTags["SimLA"] = {
    'T6'  : ( ','.join( [ 'SiPixelSimLorentzAngle_phase2_T6_v2_mc'  ,SiPixelSimLARecord,connectionString, "", "2019-11-05 20:00:00.000"] ), ), #uH = 0.0431/T (TBPX), uH=0.0/T (TEPX+TFPX)
    'T14' : ( ','.join( [ 'SiPixelSimLorentzAngle_phase2_T14_v2_mc' ,SiPixelSimLARecord,connectionString, "", "2019-11-05 20:00:00.000"] ), ), #uH = 0.0431/T (TBPX), uH=0.0/T (TEPX+TFPX)
    'T15' : ( ','.join( [ 'SiPixelSimLorentzAngle_phase2_T15_v2_mc' ,SiPixelSimLARecord,connectionString, "", "2019-11-05 20:00:00.000"] ), ), #uH = 0.0431/T (TBPX), uH=0.0/T (TEPX+TFPX)
}

allTags["GenError"] = {
    'T6'  : ( ','.join( [ 'SiPixelGenErrorDBObject_phase2_T6_v2_mc'  ,SiPixelGenErrorRecord,connectionString, "", "2019-11-05 20:00:00.000"] ), ),  # cell is 25um (local-x) x 100um (local-y) , VBias=350V
    'T14' : ( ','.join( [ 'SiPixelGenErrorDBObject_phase2_T14_v2_mc' ,SiPixelGenErrorRecord,connectionString, "", "2019-11-05 20:00:00.000"] ), ),  # cell is 25um (local-x) x 100um (local-y) , VBias=350V
    'T15' : ( ','.join( [ 'SiPixelGenErrorDBObject_phase2_T15_v2_mc' ,SiPixelGenErrorRecord,connectionString, "", "2019-11-05 20:00:00.000"] ), ),  # cell is 25um (local-x) x 100um (local-y) , VBias=350V
}

allTags["Template"] = {
    'T6'  : ( ','.join( [ 'SiPixelTemplateDBObject_phase2_T6_v2_mc'  ,SiPixelTemplatesRecord,connectionString, "", "2019-11-05 20:00:00.000"] ), ),  # cell is 25um (local-x) x 100um (local-y) , VBias=350V
    'T14' : ( ','.join( [ 'SiPixelTemplateDBObject_phase2_T14_v2_mc' ,SiPixelTemplatesRecord,connectionString, "", "2019-11-05 20:00:00.000"] ), ),  # cell is 25um (local-x) x 100um (local-y) , VBias=350V
    'T15' : ( ','.join( [ 'SiPixelTemplateDBObject_phase2_T15_v2_mc' ,SiPixelTemplatesRecord,connectionString, "" , "2019-11-05 20:00:00.000"] ), ), # cell is 25um (local-x) x 100um (local-y) , VBias=350V
}

##
## All of the following conditions are not yet in active use, but will be activated in GT along the way
##

allTags["Template2Dnum"] = {
    'T6'  : ( ','.join( [ 'SiPixel2DTemplateDBObject_phase2_T6_v0_num'  ,SiPixel2DTemplatesRecord,connectionString, "numerator", "2019-07-15 12:00:00.000"] ), ),
    'T14' : ( ','.join( [ 'SiPixel2DTemplateDBObject_phase2_T14_v0_num' ,SiPixel2DTemplatesRecord,connectionString, "numerator", "2019-07-15 12:00:00.000"] ), ),
    'T15' : ( ','.join( [ 'SiPixel2DTemplateDBObject_phase2_T15_v0_num' ,SiPixel2DTemplatesRecord,connectionString, "numerator", "2019-07-15 12:00:00.000"] ), ),
}

allTags["Template2Dden"] = {
    'T6'  : ( ','.join( [ 'SiPixel2DTemplateDBObject_phase2_T6_v0_den'  ,SiPixel2DTemplatesRecord,connectionString, "denominator", "2019-07-15 12:00:00.000"] ), ),
    'T14' : ( ','.join( [ 'SiPixel2DTemplateDBObject_phase2_T14_v0_den' ,SiPixel2DTemplatesRecord,connectionString, "denominator", "2019-07-15 12:00:00.000"] ), ),
    'T15' : ( ','.join( [ 'SiPixel2DTemplateDBObject_phase2_T15_v0_den' ,SiPixel2DTemplatesRecord,connectionString, "denominator", "2019-07-15 12:00:00.000"] ), ),
}

# list of active tags to be replaced
activeKeys = ["LA","LAWidth","SimLA","LAfromAlignment","GenError","Template"]

# list of geometries supported
activeDets = ["T6","T14","T15"]
phase2GTs = {}
for det in activeDets:
    appendedTags = ()
    for key in activeKeys:
       appendedTags += allTags[key][det]
    phase2GTs["phase2_realistic_"+det] = ('phase2_realistic', appendedTags)

# method called in autoCond
def autoCondPhase2(autoCond):
    for key,val in six.iteritems(phase2GTs):
        if len(val)==1 :
           autoCond[key] = ( autoCond[val[0]] )
        else:
           autoCond[key] = ( autoCond[val[0]],) + val[1]

    return autoCond
