import six
from Configuration.StandardSequences.CondDBESSource_cff import GlobalTag as essource
connectionString = essource.connect.value()

SiPixelLARecord           =   "SiPixelLorentzAngleRcd"                  
SiPixelSimLARecord        =   "SiPixelLorentzAngleSimRcd"   
SiPixelGenErrorRecord     =   "SiPixelGenErrorDBObjectRcd"       
SiPixelTemplatesRecord    =   "SiPixelTemplateDBObjectRcd"       
SiPixel2DTemplatesRecord  =   "SiPixel2DTemplateDBObjectRcd"     
TrackerDTCCablingRecord   =   "TrackerDetToDTCELinkCablingMapRcd"

#combines in a single dict of dict the tags defined below
allTags={}

## As Outer Tracker in T6 is different from all the subsequent active ones (>=T14), this needs to be specified outside of the Global Tag.
allTags["DTCCabling"] = {
    "T6" :  ( ','.join( [ 'TrackerDetToDTCELinkCablingMap__OT614_200_IT404_layer2_10G__T6__OTOnly' ,TrackerDTCCablingRecord, connectionString, "", "2020-03-27 11:30:00.000"] ), ), # DTC cabling map provided for T6 geometry (taken from http://ghugo.web.cern.ch/ghugo/layouts/cabling/OT614_200_IT404_layer2_10G/cablingOuter.html)
}

#v5 versions of LA used to match versioning of pixel templates, but values are identical to v2
allTags["LA"] = {
    'T6'  : ( ','.join( [ 'SiPixelLorentzAngle_phase2_T6_v5_mc'  ,SiPixelLARecord,connectionString, "", "2020-05-05 20:00:00.000"] ), ),  #uH = 0.106/T (TBPX), uH=0.0/T (TEPX+TFPX)
    'T14' : ( ','.join( [ 'SiPixelLorentzAngle_phase2_T14_v5_mc' ,SiPixelLARecord,connectionString, "", "2020-05-05 20:00:00.000"] ), ),  #uH = 0.106/T (TBPX), uH=0.0/T (TEPX+TFPX)
    'T15' : ( ','.join( [ 'SiPixelLorentzAngle_phase2_T15_v5_mc' ,SiPixelLARecord,connectionString, "", "2020-05-05 20:00:00.000"] ), ),  #uH = 0.053/T (TBPX), uH=0.0/T (TEPX+TFPX)
    'T19' : ( ','.join( [ 'SiPixelLorentzAngle_phase2_T19_v1_mc' ,SiPixelLARecord,connectionString, "", "2020-02-23 14:00:00.000"] ), ),  #uH = 0.053/T (TBPX L3,L4), uH=0.0/T (TBPX L1,L2, TEPX+TFPX)
}

allTags["LAWidth"] = {
    'T6'  : ( ','.join( [ 'SiPixelLorentzAngle_phase2_T6_mc_forWidthEmpty'  ,SiPixelLARecord,connectionString, "forWidth", "2019-11-05 20:00:00.000"] ), ),  # uH=0.0/T (fall-back to offset)
    'T14' : ( ','.join( [ 'SiPixelLorentzAngle_phase2_T14_mc_forWidthEmpty' ,SiPixelLARecord,connectionString, "forWidth", "2019-11-05 20:00:00.000"] ), ),  # uH=0.0/T (fall-back to offset)
    'T15' : ( ','.join( [ 'SiPixelLorentzAngle_phase2_T15_mc_forWidthEmpty' ,SiPixelLARecord,connectionString, "forWidth", "2019-11-05 20:00:00.000"] ), ),  # uH=0.0/T (fall-back to offset)
    'T19' : ( ','.join( [ 'SiPixelLorentzAngle_phase2_T19_mc_forWidthEmpty' ,SiPixelLARecord,connectionString, "forWidth", "2020-02-23 14:00:00.000"] ), ),  # uH=0.0/T (fall-back to offset)
}

allTags["LAfromAlignment"] = {
    'T6'  : ( ','.join( [ 'SiPixelLorentzAngle_phase2_T6_mc_forWidthEmpty'  ,SiPixelLARecord,connectionString, "fromAlignment", "2019-11-05 20:00:00.000"] ), ),  # uH=0.0/T (not in use)
    'T14' : ( ','.join( [ 'SiPixelLorentzAngle_phase2_T14_mc_forWidthEmpty' ,SiPixelLARecord,connectionString, "fromAlignment", "2019-11-05 20:00:00.000"] ), ),  # uH=0.0/T (not in use)
    'T15' : ( ','.join( [ 'SiPixelLorentzAngle_phase2_T15_mc_forWidthEmpty' ,SiPixelLARecord,connectionString, "fromAlignment", "2019-11-05 20:00:00.000"] ), ),  # uH=0.0/T (not in use)
    'T19' : ( ','.join( [ 'SiPixelLorentzAngle_phase2_T19_mc_forWidthEmpty' ,SiPixelLARecord,connectionString, "fromAlignment", "2020-02-23 14:00:00.000"] ), ),  # uH=0.0/T (not in use)
}

#v5 versions of SimLA used to match versioning of pixel templates, but values are indentical to  v2
allTags["SimLA"] = {
    'T6'  : ( ','.join( [ 'SiPixelSimLorentzAngle_phase2_T6_v5_mc'  ,SiPixelSimLARecord,connectionString, "", "2020-05-05 20:00:00.000"] ), ), #uH = 0.106/T (TBPX), uH=0.0/T (TEPX+TFPX)
    'T14' : ( ','.join( [ 'SiPixelSimLorentzAngle_phase2_T14_v5_mc' ,SiPixelSimLARecord,connectionString, "", "2020-05-05 20:00:00.000"] ), ), #uH = 0.106/T (TBPX), uH=0.0/T (TEPX+TFPX)
    'T15' : ( ','.join( [ 'SiPixelSimLorentzAngle_phase2_T15_v5_mc' ,SiPixelSimLARecord,connectionString, "", "2020-05-05 20:00:00.000"] ), ), #uH = 0.053/T (TBPX), uH=0.0/T (TEPX+TFPX)
    'T19' : ( ','.join( [ 'SiPixelSimLorentzAngle_phase2_T19_v1_mc' ,SiPixelSimLARecord,connectionString, "", "2020-02-23 14:00:00.000"] ), ), #uH = 0.053/T (TBPX L3,L4), uH=0.0/T (TBPX L1,L2, TEPX+TFPX)
}

allTags["GenError"] = {
    'T6'  : ( ','.join( [ 'SiPixelGenErrorDBObject_phase2_T6_v5_mc'  ,SiPixelGenErrorRecord,connectionString, "", "2020-05-02 23:00:00.000"] ), ),  # cell is 25um (local-x) x 100um (local-y) , VBias=350V
    'T14' : ( ','.join( [ 'SiPixelGenErrorDBObject_phase2_T14_v5_mc' ,SiPixelGenErrorRecord,connectionString, "", "2020-05-02 23:00:00.000"] ), ),  # cell is 25um (local-x) x 100um (local-y) , VBias=350V
    'T15' : ( ','.join( [ 'SiPixelGenErrorDBObject_phase2_T15_v5_mc' ,SiPixelGenErrorRecord,connectionString, "", "2020-05-02 23:00:00.000"] ), ),  # cell is 25um (local-x) x 100um (local-y) , VBias=350V
}

allTags["Template"] = {
    'T6'  : ( ','.join( [ 'SiPixelTemplateDBObject_phase2_T6_v5_mc' ,SiPixelTemplatesRecord,connectionString, "", "2020-05-02 23:00:00.000"] ), ),  # cell is 25um (local-x) x 100um (local-y) , VBias=350V
    'T14' : ( ','.join( [ 'SiPixelTemplateDBObject_phase2_T14_v5_mc',SiPixelTemplatesRecord,connectionString, "", "2020-05-02 23:00:00.000"] ), ),  # cell is 25um (local-x) x 100um (local-y) , VBias=350V
    'T15' : ( ','.join( [ 'SiPixelTemplateDBObject_phase2_T15_v5_mc',SiPixelTemplatesRecord,connectionString, "", "2020-05-02 23:00:00.000"] ), ),  # cell is 25um (local-x) x 100um (local-y) , VBias=350V
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
activeKeys = ["DTCCabling","LA","LAWidth","SimLA","LAfromAlignment","GenError","Template"]

# list of geometries supported
activeDets = ["T6","T14","T15","T19"]
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
