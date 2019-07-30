# default value for all customizations
connectionString = "frontier://FrontierProd/CMS_CONDITIONS"

SiPixelLARecord           =   "SiPixelLorentzAngleRcd"                  
SiPixelSimLARecord        =   "SiPixelLorentzAngleSimRcd"   
SiPixelGenErrorRecord     =   "SiPixelGenErrorDBObjectRcd"       
SiPixelTemplatesRecord    =   "SiPixelTemplateDBObjectRcd"       
SiPixel2DTemplatesRecord  =   "SiPixel2DTemplateDBObjectRcd"     

LA = {
    'T6'  : ( ','.join( [ 'SiPixelLorentzAngle_phase2_T6_v0_mc'  ,SiPixelLARecord,connectionString, "", "2019-07-15 12:00:00.000"] ), ),
    'T14' : ( ','.join( [ 'SiPixelLorentzAngle_phase2_T14_v0_mc' ,SiPixelLARecord,connectionString, "", "2019-07-15 12:00:00.000"] ), ),
    'T15' : ( ','.join( [ 'SiPixelLorentzAngle_phase2_T15_v0_mc' ,SiPixelLARecord,connectionString, "", "2019-07-15 12:00:00.000"] ), ),
}

LAWidth = {
    'T6'  : ( ','.join( [ 'SiPixelLorentzAngle_phase2_forWidth_T6_v0_mc'  ,SiPixelLARecord,connectionString, "forWidth", "2019-07-15 12:00:00.000"] ), ),
    'T14' : ( ','.join( [ 'SiPixelLorentzAngle_phase2_forWidth_T14_v0_mc' ,SiPixelLARecord,connectionString, "forWidth", "2019-07-15 12:00:00.000"] ), ),
    'T15' : ( ','.join( [ 'SiPixelLorentzAngle_phase2_forWidth_T15_v0_mc' ,SiPixelLARecord,connectionString, "forWidth", "2019-07-15 12:00:00.000"] ), ),
}

SimLA = {
    'T6'  : ( ','.join( [ 'SiPixelSimLorentzAngle_phase2_T6_v0_mc'  ,SiPixelSimLARecord,connectionString, "", "2019-07-15 12:00:00.000"] ), ),
    'T14' : ( ','.join( [ 'SiPixelSimLorentzAngle_phase2_T14_v0_mc' ,SiPixelSimLARecord,connectionString, "", "2019-07-15 12:00:00.000"] ), ),
    'T15' : ( ','.join( [ 'SiPixelSimLorentzAngle_phase2_T15_v0_mc' ,SiPixelSimLARecord,connectionString, "", "2019-07-15 12:00:00.000"] ), ),
}

##
## All of the following conditions are not yet in active use, but will be activated in GT along the way
##

GenError = {
    'T6'  : ( ','.join( [ 'SiPixelGenErrorDBOject_phase2_T6_v0_mc'  ,SiPixelGenErrorRecord,connectionString, "", "2019-07-15 12:00:00.000"] ), ),
    'T14' : ( ','.join( [ 'SiPixelGenErrorDBOject_phase2_T14_v0_mc' ,SiPixelGenErrorRecord,connectionString, "", "2019-07-15 12:00:00.000"] ), ),
    'T15' : ( ','.join( [ 'SiPixelGenErrorDBOject_phase2_T15_v0_mc' ,SiPixelGenErrorRecord,connectionString, "", "2019-07-15 12:00:00.000"] ), ),
}

Template = {
    'T6'  : ( ','.join( [ 'SiPixelTemplateDBObject_phase2_T6_v0_mc'  ,SiPixelTemplatesRecord,connectionString, "", "2019-07-15 12:00:00.000"] ), ),
    'T14' : ( ','.join( [ 'SiPixelTemplateDBObject_phase2_T14_v0_mc' ,SiPixelTemplatesRecord,connectionString, "", "2019-07-15 12:00:00.000"] ), ),
    'T15' : ( ','.join( [ 'SiPixelTemplateDBObject_phase2_T15_v0_mc' ,SiPixelTemplatesRecord,connectionString, "" , "2019-07-15 12:00:00.000"] ), ),
}

Template2Dnum = {
    'T6'  : ( ','.join( [ 'SiPixel2DTemplateDBObject_phase2_T6_v0_num'  ,SiPixel2DTemplatesRecord,connectionString, "numerator", "2019-07-15 12:00:00.000"] ), ),
    'T14' : ( ','.join( [ 'SiPixel2DTemplateDBObject_phase2_T14_v0_num' ,SiPixel2DTemplatesRecord,connectionString, "numerator", "2019-07-15 12:00:00.000"] ), ),
    'T15' : ( ','.join( [ 'SiPixel2DTemplateDBObject_phase2_T15_v0_num' ,SiPixel2DTemplatesRecord,connectionString, "numerator", "2019-07-15 12:00:00.000"] ), ),
}

Template2Dden = {
    'T6'  : ( ','.join( [ 'SiPixel2DTemplateDBObject_phase2_T6_v0_den'  ,SiPixel2DTemplatesRecord,connectionString, "denominator", "2019-07-15 12:00:00.000"] ), ),
    'T14' : ( ','.join( [ 'SiPixel2DTemplateDBObject_phase2_T14_v0_den' ,SiPixel2DTemplatesRecord,connectionString, "denominator", "2019-07-15 12:00:00.000"] ), ),
    'T15' : ( ','.join( [ 'SiPixel2DTemplateDBObject_phase2_T15_v0_den' ,SiPixel2DTemplatesRecord,connectionString, "denominator", "2019-07-15 12:00:00.000"] ), ),
}

#combines in a single dict of dict the tags defined above
allTags={
    "LA" : LA ,
    "LAWidth": LAWidth ,
    "SimLA" : SimLA ,
    "GenError": GenError ,
    "Template": Template ,
    "Template2Dnum": Template2Dnum ,
    "Template2Dden": Template2Dden
}

# list of active tags to be replaced
activeKeys = ["LA","LAWidth","SimLA"]

# list og geometries supported
activeDets = ["T6","T14","T15"]
phase2GTs = {}
for det in activeDets:
    appendedTags = ()
    for key in activeKeys:
       appendedTags += allTags[key][det]
    phase2GTs["phase2_realistic_"+det] = ('phase2_realistic', appendedTags)

def autoCondPhase2(autoCond):
    for key,val in phase2GTs.iteritems():
        if len(val)==1 :
           autoCond[key] = ( autoCond[val[0]] )
        else:
           autoCond[key] = ( autoCond[val[0]],) + val[1]

    return autoCond
