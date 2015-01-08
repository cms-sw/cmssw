import FWCore.ParameterSet.Config as cms
import os

#SET = "71212"
#SET = "90322"
#SET = "120812"
#SET = "130503"

#SUBSET = ""
#SUBSET = "2pi_scaled"
#SUBSET = "Run1"
#SUBSET = "Run2"

#B_NOM = "0T"
#B_NOM = "2T"
#B_NOM = "3T"
#B_NOM = "3_5T"
#B_NOM = "3.8T"
#B_NOM = "4T"


process = cms.Process("DumpToDB")

process.load("CondCore.DBCommon.CondDBSetup_cfi")

process.source = cms.Source("EmptySource",
    numberEventsInRun = cms.untracked.uint32(1),
    firstRun = cms.untracked.uint32(300000)
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)


process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = "START72_V1::All"
#process.GlobalTag.pfnPrefix = cms.untracked.string('frontier://FrontierProd/')


if SUBSET == "" : 
    TAG = "MFConfig_"+SET+"_"+B_NOM
else :
    TAG = "MFConfig_"+SET+"_"+SUBSET+"_"+B_NOM

FILE = TAG+".db"

try:
    os.remove(FILE)
except OSError:
    pass

# VDrift, TTrig, TZero, Noise or channels Map into DB
process.PoolDBOutputService = cms.Service("PoolDBOutputService",
                                          process.CondDBSetup,
                                          connect = cms.string("sqlite_file:"+FILE),
                                          toPut = cms.VPSet(cms.PSet(record = cms.string("MagFieldConfigRcd"),
                                                                     tag = cms.string("MagFieldConfig"))))

def createMetadata(aTag,aComment):
    txtfile = open(aTag+'.txt', 'w')
    txtfile.write('{\n')
    txtfile.write('   "destinationDatabase": "oracle://cms_orcoff_prep/CMS_COND_GEOMETRY",\n')
    txtfile.write('   "destinationTags": {\n')
    txtfile.write('      "'+TAG+'": {\n')
    txtfile.write('         "dependencies": {},\n')
    txtfile.write('         "synchronizeTo": "offline"\n')
    txtfile.write('        }\n')
    txtfile.write('    },\n')
    txtfile.write('    "inputTag": "MagFieldConfig",\n')
    txtfile.write('    "since": 1,\n')
    txtfile.write('    "userText": "'+aComment+'"\n')
    txtfile.write('}\n')
    return



if SET=="71212" :
    if SUBSET!="": raise NameError("configuration invalid: "+SET)
    versions = {'0T':   'grid_1103l_071212_4t',
                '2T':   'grid_1103l_071212_2t',
                '3T':   'grid_1103l_071212_3t',
                '3_5T': 'grid_1103l_071212_3_5t',
                #'3.8T': 'grid_1103l_071212_3_8t', #Deprecated
                '4T':   'grid_1103l_071212_4t'} 
    param    = {'0T':   0.,
                '2T':   2.,
                '3T':   3.,
                '3_5T': 3.5,
                #'3_8T': 3.8,
                '4T':   4.}
    process.dumpToDB = cms.EDAnalyzer("MagFieldConfigDBWriter",
          scalingVolumes = cms.vint32(),
          scalingFactors = cms.vdouble(),
          version = cms.string(versions[B_NOM]),
          geometryVersion = cms.int32(90322),
          paramLabel = cms.string('OAE_1103l_071212'),
          paramData = cms.vdouble(param[B_NOM]),
          gridFiles = cms.VPSet(
            cms.PSet( # Default tables, replicate sector 1
               volumes   = cms.string('1-312'),
               sectors   = cms.string('0') ,
               master    = cms.int32(1),
               path      = cms.string('grid.[v].bin'),
           )
         )
    )
    if B_NOM == '0T' :
        process.dumpToDB.paramLabel = 'Uniform'
    

if SET=="90322" :
    if B_NOM!="3_8T" or SUBSET!="2pi_scaled":  raise NameError("configuration invalid: "+SET)

    from MagneticField.Engine.ScalingFactors_090322_2pi_090520_cfi import *
    
    process.dumpToDB = cms.EDAnalyzer("MagFieldConfigDBWriter",
      fieldScaling,
      version = cms.string('grid_1103l_090322_3_8t'),
      geometryVersion = cms.int32(90322),
      paramLabel = cms.string('OAE_1103l_071212'),
      paramData = cms.vdouble(3.8),
    
      gridFiles = cms.VPSet(
          cms.PSet( # Default tables, replicate sector 1
              volumes   = cms.string('1-312'),
              sectors   = cms.string('0') ,
              master    = cms.int32(1),
              path      = cms.string('grid.[v].bin'),
          ),
          cms.PSet( # Specific volumes in Barrel, sector 3
              volumes   = cms.string('176-186,231-241,286-296'),
              sectors   = cms.string('3') ,
              master    = cms.int32(3),
              path      = cms.string('S3/grid.[v].bin'),
          ),
          cms.PSet( # Specific volumes in Barrel, sector 4
              volumes   = cms.string('176-186,231-241,286-296'),
              sectors   = cms.string('4') ,
              master    = cms.int32(4),
              path      = cms.string('S4/grid.[v].bin'),
          ),
          cms.PSet(  # Specific volumes in Barrel and endcaps, sector 9
              volumes   = cms.string('14,15,20,21,24-27,32,33,40,41,48,49,56,57,62,63,70,71,286-296'),
              sectors   = cms.string('9') ,
              master    = cms.int32(9),
              path      = cms.string('S9/grid.[v].bin'),
          ),
          cms.PSet(  # Specific volumes in Barrel and endcaps, sector 10
              volumes   = cms.string('14,15,20,21,24-27,32,33,40,41,48,49,56,57,62,63,70,71,286-296'),
              sectors   = cms.string('10') ,
              master    = cms.int32(10),
              path      = cms.string('S10/grid.[v].bin'),
          ),                                              
          cms.PSet( # Specific volumes in Barrel and endcaps, sector 11
              volumes   = cms.string('14,15,20,21,24-27,32,33,40,41,48,49,56,57,62,63,70,71,286-296'),
              sectors   = cms.string('11') ,
              master    = cms.int32(11),
              path      = cms.string('S11/grid.[v].bin'),
          ),
       )
    )
      

elif SET=="120812" :
  if B_NOM!="3_8T" :  raise NameError("B_NOM invalid for SET "+SET)
  if    SUBSET=="Run1" : VERSION = 'grid_120812_3_8t_v7_small'
  elif  SUBSET=="Run2" : VERSION = 'grid_120812_3_8t_v7_large'
  else : raise NameError("invalid SUBSET: "+SUBSET+ " for "+TAG )
  
  process.dumpToDB = cms.EDAnalyzer("MagFieldConfigDBWriter",
      scalingVolumes = cms.vint32(),
      scalingFactors = cms.vdouble(),
      version = cms.string(VERSION),
      geometryVersion = cms.int32(120812),
      paramLabel = cms.string('OAE_1103l_071212'),
      paramData = cms.vdouble(3.8),
    
      gridFiles = cms.VPSet(
          # Specific tables for each sector
          cms.PSet( 
              volumes   = cms.string('1001-1010,1012-1027,1030-1033,1036-1041,1044-1049,1052-1057,1060-1063,1066-1071,1074-1077,1080-1083,1130-1133,1138-1360'),
              sectors   = cms.string('0') ,
              master    = cms.int32(0),
              path      = cms.string('s[s]_1/grid.[v].bin'),
          ),
          cms.PSet(
              volumes   = cms.string('2001-2010,2012-2027,2030-2033,2036-2041,2044-2049,2052-2057,2060-2063,2066-2071,2074-2077,2080-2083,2130-2133,2138-2360'),
              sectors   = cms.string('0'),
              master    = cms.int32(0),
              path      = cms.string('s[s]_2/grid.[v].bin'),
          ),
          # Replicate sector 1 for volumes outside any detector
          cms.PSet( 
              volumes   = cms.string('1011,1028-1029,1034-1035,1042-1043,1050-1051,1058-1059,1064-1065,1072-1073,1078-1079,1084-1129,1136-1137'),
              sectors   = cms.string('0'),
              master    = cms.int32(1),
              path      = cms.string('s01_1/grid.[v].bin'),
          ),
          cms.PSet(
              volumes   = cms.string('2011,2028-2029,2034-2035,2042-2043,2050-2051,2058-2059,2064-2065,2072-2073,2078-2079,2084-2129,2136-2137'),
              sectors   = cms.string('0'),
              master    = cms.int32(1),
              path      = cms.string('s01_2/grid.[v].bin'),
          ),   
         # Replicate sector 4 for the volume outside CASTOR, to avoid aliasing due to the plates in the cylinder gap
         # between the collar and the rotating shielding.
         cms.PSet(
             volumes   = cms.string('1134-1135'),
             sectors   = cms.string('0'),
             master    = cms.int32(4),
             path      = cms.string('s04_1/grid.[v].bin'),
         ),
         cms.PSet(
             volumes   = cms.string('2134-2135'),
             sectors   = cms.string('0'),
             master    = cms.int32(4),
             path      = cms.string('s04_2/grid.[v].bin'),
         ),
      )
    )
    

    
elif SET=="130503" :
  versions = {'3_5T': 'grid_130503_3_5t_v9',
              '3_8T': 'grid_130503_3_8t_v9'}
  param    = {'3_5T': 3.5,
              '3_8T': 3.8}
  
  if    SUBSET=="Run1" : VERSION = versions[B_NOM]+'_small'
  elif  SUBSET=="Run2" : VERSION = versions[B_NOM]+'_large'
  else : raise NameError("invalid SUBSET: "+SUBSET+ " for "+TAG )

  process.dumpToDB = cms.EDAnalyzer("MagFieldConfigDBWriter",
      scalingVolumes = cms.vint32(),
      scalingFactors = cms.vdouble(),
      version = cms.string(VERSION),
      geometryVersion = cms.int32(130503),
      paramLabel = cms.string('OAE_1103l_071212'),
      paramData = cms.vdouble(param[B_NOM]),
    
      gridFiles = cms.VPSet(
          # Volumes for which specific tables are used for each sector
          cms.PSet(
              volumes   = cms.string('1001-1010,1012-1027,1030-1033,1036-1041,1044-1049,1052-1057,1060-1063,1066-1071,1074-1077,1080-1083,1130-1133,1138-1402,' + 
                                     '2001-2010,2012-2027,2030-2033,2036-2041,2044-2049,2052-2057,2060-2063,2066-2071,2074-2077,2080-2083,2130-2133,2138-2402'),
              sectors   = cms.string('0') ,
              master    = cms.int32(0),
              path      = cms.string('s[s]/grid.[v].bin'),
          ),
         # Replicate sector 1 for volumes outside any detector
           cms.PSet(
              volumes   = cms.string('1011,1028-1029,1034-1035,1042-1043,1050-1051,1058-1059,1064-1065,1072-1073,1078-1079,1084-1129,1136-1137,' +
                                     '2011,2028-2029,2034-2035,2042-2043,2050-2051,2058-2059,2064-2065,2072-2073,2078-2079,2084-2129,2136-2137'),
              sectors   = cms.string('0'),
              master    = cms.int32(1),
              path      = cms.string('s01/grid.[v].bin'),
          ),

         # Replicate sector 4 for the volume outside CASTOR, to avoid aliasing due to the plates in the cylinder gap
         # between the collar and the rotating shielding.
         cms.PSet(
             volumes   = cms.string('1134-1135,2134-2135'),
             sectors   = cms.string('0'),
             master    = cms.int32(4),
             path      = cms.string('s04/grid.[v].bin'),
         ),
    )
  )


process.p = cms.Path(process.dumpToDB)
    
createMetadata(TAG,"Mag field configuration for map "+TAG)

