# default value for all customizations
connectionString = "frontier://FrontierPrep/CMS_CONDITIONS"

SiPixelLARecord           =   "SiPixelLorentzAngleRcd"                  
SiPixelSimLARecord        =   "SiPixelLorentzAngleSimRcd"   
SiPixelGenErrorRecord     =   "SiPixelGenErrorDBObjectRcd"       
SiPixelTemplatesRecord    =   "SiPixelTemplateDBObjectRcd"       
SiPixel2DTemplatesRecord  =   "SiPixel2DTemplateDBObjectRcd"     

LA = {
    'T5'  : ( ','.join( [ 'SiPixelLorentzAngle_phase2_T5_v0_mc'  ,SiPixelLARecord,connectionString,"", "2019-07-15 12:00:00.000"] ), ),
    'T6'  : ( ','.join( [ 'SiPixelLorentzAngle_phase2_T6_v0_mc'  ,SiPixelLARecord,connectionString,"", "2019-07-15 12:00:00.000"] ), ),
    'T11' : ( ','.join( [ 'SiPixelLorentzAngle_phase2_T11_v0_mc' ,SiPixelLARecord,connectionString,"", "2019-07-15 12:00:00.000"] ), ),
    'T14' : ( ','.join( [ 'SiPixelLorentzAngle_phase2_T14_v0_mc' ,SiPixelLARecord,connectionString,"", "2019-07-15 12:00:00.000"] ), ),
    'T15' : ( ','.join( [ 'SiPixelLorentzAngle_phase2_T15_v0_mc' ,SiPixelLARecord,connectionString,"", "2019-07-15 12:00:00.000"] ), ),
}

LAWidth = {
    'T5'  : ( ','.join( [ 'SiPixelLorentzAngle_phase2_forWidth_T5_v0_mc'  ,SiPixelLARecord,connectionString,"forWidth", "2019-07-15 12:00:00.000"] ), ),
    'T6'  : ( ','.join( [ 'SiPixelLorentzAngle_phase2_forWidth_T6_v0_mc'  ,SiPixelLARecord,connectionString,"forWidth", "2019-07-15 12:00:00.000"] ), ),
    'T11' : ( ','.join( [ 'SiPixelLorentzAngle_phase2_forWidth_T11_v0_mc' ,SiPixelLARecord,connectionString,"forWidth", "2019-07-15 12:00:00.000"] ), ),
    'T14' : ( ','.join( [ 'SiPixelLorentzAngle_phase2_forWidth_T14_v0_mc' ,SiPixelLARecord,connectionString,"forWidth", "2019-07-15 12:00:00.000"] ), ),
    'T15' : ( ','.join( [ 'SiPixelLorentzAngle_phase2_forWidth_T15_v0_mc' ,SiPixelLARecord,connectionString,"forWidth", "2019-07-15 12:00:00.000"] ), ),
}

SimLA = {
    'T5'  : ( ','.join( [ 'SiPixelSimLorentzAngle_phase2_T5_v0_mc'  ,SiPixelSimLARecord,connectionString,"", "2019-07-15 12:00:00.000"] ), ),
    'T6'  : ( ','.join( [ 'SiPixelSimLorentzAngle_phase2_T6_v0_mc'  ,SiPixelSimLARecord,connectionString,"", "2019-07-15 12:00:00.000"] ), ),
    'T11' : ( ','.join( [ 'SiPixelSimLorentzAngle_phase2_T11_v0_mc' ,SiPixelSimLARecord,connectionString,"", "2019-07-15 12:00:00.000"] ), ),
    'T14' : ( ','.join( [ 'SiPixelSimLorentzAngle_phase2_T14_v0_mc' ,SiPixelSimLARecord,connectionString,"", "2019-07-15 12:00:00.000"] ), ),
    'T15' : ( ','.join( [ 'SiPixelSimLorentzAngle_phase2_T15_v0_mc' ,SiPixelSimLARecord,connectionString,"", "2019-07-15 12:00:00.000"] ), ),
}

GenError = {
    'T5'  : ( ','.join( [ 'SiPixelGenErrorDBOject_phase2_T5_v0_mc'  ,SiPixelGenErrorRecord,connectionString,"", "2019-07-15 12:00:00.000"] ), ),
    'T6'  : ( ','.join( [ 'SiPixelGenErrorDBOject_phase2_T6_v0_mc'  ,SiPixelGenErrorRecord,connectionString,"", "2019-07-15 12:00:00.000"] ), ),
    'T11' : ( ','.join( [ 'SiPixelGenErrorDBOject_phase2_T11_v0_mc' ,SiPixelGenErrorRecord,connectionString,"", "2019-07-15 12:00:00.000"] ), ),
    'T14' : ( ','.join( [ 'SiPixelGenErrorDBOject_phase2_T14_v0_mc' ,SiPixelGenErrorRecord,connectionString,"", "2019-07-15 12:00:00.000"] ), ),
    'T15' : ( ','.join( [ 'SiPixelGenErrorDBOject_phase2_T15_v0_mc' ,SiPixelGenErrorRecord,connectionString,"", "2019-07-15 12:00:00.000"] ), ),
}

Template = {
    'T5'  : ( ','.join( [ 'SiPixelTemplateDBObject_phase2_T5_v0_mc'  ,SiPixelTemplatesRecord,connectionString,"", "2019-07-15 12:00:00.000"] ), ),
    'T6'  : ( ','.join( [ 'SiPixelTemplateDBObject_phase2_T6_v0_mc'  ,SiPixelTemplatesRecord,connectionString,"", "2019-07-15 12:00:00.000"] ), ),
    'T11' : ( ','.join( [ 'SiPixelTemplateDBObject_phase2_T11_v0_mc' ,SiPixelTemplatesRecord,connectionString,"", "2019-07-15 12:00:00.000"] ), ),
    'T14' : ( ','.join( [ 'SiPixelTemplateDBObject_phase2_T14_v0_mc' ,SiPixelTemplatesRecord,connectionString,"", "2019-07-15 12:00:00.000"] ), ),
    'T15' : ( ','.join( [ 'SiPixelTemplateDBObject_phase2_T15_v0_mc' ,SiPixelTemplatesRecord,connectionString,"", "2019-07-15 12:00:00.000"] ), ),
}

Template2Dnum = {
    'T5'  : ( ','.join( [ 'SiPixel2DTemplateDBObject_phase2_T5_v0_num'  ,SiPixel2DTemplatesRecord,connectionString,"numerator", "2019-07-15 12:00:00.000"] ), ),
    'T6'  : ( ','.join( [ 'SiPixel2DTemplateDBObject_phase2_T6_v0_num'  ,SiPixel2DTemplatesRecord,connectionString,"numerator", "2019-07-15 12:00:00.000"] ), ),
    'T11' : ( ','.join( [ 'SiPixel2DTemplateDBObject_phase2_T11_v0_num' ,SiPixel2DTemplatesRecord,connectionString,"numerator", "2019-07-15 12:00:00.000"] ), ),
    'T14' : ( ','.join( [ 'SiPixel2DTemplateDBObject_phase2_T14_v0_num' ,SiPixel2DTemplatesRecord,connectionString,"numerator", "2019-07-15 12:00:00.000"] ), ),
    'T15' : ( ','.join( [ 'SiPixel2DTemplateDBObject_phase2_T15_v0_num' ,SiPixel2DTemplatesRecord,connectionString,"numerator", "2019-07-15 12:00:00.000"] ), ),
}

Template2Dden = {
    'T5'  : ( ','.join( [ 'SiPixel2DTemplateDBObject_phase2_T5_v0_den'  ,SiPixel2DTemplatesRecord,connectionString,"denominator", "2019-07-15 12:00:00.000"] ), ),
    'T6'  : ( ','.join( [ 'SiPixel2DTemplateDBObject_phase2_T6_v0_den'  ,SiPixel2DTemplatesRecord,connectionString,"denominator", "2019-07-15 12:00:00.000"] ), ),
    'T11' : ( ','.join( [ 'SiPixel2DTemplateDBObject_phase2_T11_v0_den' ,SiPixel2DTemplatesRecord,connectionString,"denominator", "2019-07-15 12:00:00.000"] ), ),
    'T14' : ( ','.join( [ 'SiPixel2DTemplateDBObject_phase2_T14_v0_den' ,SiPixel2DTemplatesRecord,connectionString,"denominator", "2019-07-15 12:00:00.000"] ), ),
    'T15' : ( ','.join( [ 'SiPixel2DTemplateDBObject_phase2_T15_v0_den' ,SiPixel2DTemplatesRecord,connectionString,"denominator", "2019-07-15 12:00:00.000"] ), ),
}

phase2GTs = {    
#   'symbolic GT'           : ('base GT',[('payload1',payload2')])
    'phase2_realistic_T5'   : ('phase2_realistic',LA['T5'] +LAWidth['T5'] +SimLA['T5']),
    'phase2_realistic_T6'   : ('phase2_realistic',LA['T6'] +LAWidth['T6'] +SimLA['T6']),
    'phase2_realistic_T11'  : ('phase2_realistic',LA['T11']+LAWidth['T11']+SimLA['T11']),
    'phase2_realistic_T14'  : ('phase2_realistic',LA['T14']+LAWidth['T14']+SimLA['T14']),
    'phase2_realistic_T15'  : ('phase2_realistic',LA['T15']+LAWidth['T15']+SimLA['T15']),
}

def autoCondPhase2(autoCond):
    for key,val in phase2GTs.iteritems():
        if len(val)==1 :
           autoCond[key] = ( autoCond[val[0]] )
        else:
           autoCond[key] = ( autoCond[val[0]],) + val[1]

    return autoCond
