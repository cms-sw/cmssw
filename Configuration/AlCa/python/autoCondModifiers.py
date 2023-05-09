##
## Append for 0T conditions
##
from Configuration.StandardSequences.CondDBESSource_cff import GlobalTag as essource
connectionString = essource.connect.value()

# method called in autoCond
def autoCond0T(autoCond):
    
    ConditionsFor0T =  ','.join( ['RunInfo_0T_v1_mc', "RunInfoRcd", connectionString, "", "2020-07-01 12:00:00.000"] )
    GlobalTags0T = {}
    for key,val in autoCond.items():
        if "phase" in key:    # restrict to phase1 upgrade GTs
            GlobalTags0T[key+"_0T"] = (autoCond[key], ConditionsFor0T)

    autoCond.update(GlobalTags0T)
    return autoCond

def autoCondHLTHI(autoCond):

    GlobalTagsHLTHI = {}

    # emulate hybrid ZeroSuppression on the VirginRaw data of 2015
    FullPedestalsForHLTHI =  ','.join( ['SiStripFullPedestals_GR10_v1_hlt', "SiStripPedestalsRcd", connectionString, "", "2021-03-11 12:00:00.000"] )
    MenuForHLTHI =  ','.join( ['L1Menu_CollisionsHeavyIons2015_v5_uGT_xml', "L1TUtmTriggerMenuRcd", connectionString, "", "2021-03-11 12:00:00.000"] )

    for key,val in autoCond.items():
        if key == 'run2_hlt_relval':    # modification of HLT relval GT
            GlobalTagsHLTHI['run2_hlt_hi'] = (autoCond[key], FullPedestalsForHLTHI, MenuForHLTHI)

    autoCond.update(GlobalTagsHLTHI)
    return autoCond

def autoCondDDD(autoCond):

    GlobalTagsDDD = {}
    # substitute the DD4hep geometry tags with DDD ones
    CSCRECODIGI_Geometry_ddd    =  ','.join( ['CSCRECODIGI_Geometry_112YV2'            , "CSCRecoDigiParametersRcd"      , connectionString, ""        , "2021-09-28 12:00:00.000"] )
    CSCRECO_Geometry_ddd        =  ','.join( ['CSCRECO_Geometry_112YV2'                , "CSCRecoGeometryRcd"            , connectionString, ""        , "2021-09-28 12:00:00.000"] )
    DTRECO_Geometry_ddd         =  ','.join( ['DTRECO_Geometry_112YV2'                 , "DTRecoGeometryRcd"             , connectionString, ""        , "2021-09-28 12:00:00.000"] )
    GEMRECO_Geometry_ddd        =  ','.join( ['GEMRECO_Geometry_123YV2'                , "GEMRecoGeometryRcd"            , connectionString, ""        , "2022-02-02 12:00:00.000"] )
    XMLFILE_Geometry_ddd        =  ','.join( ['XMLFILE_Geometry_123YV1_Extended2021_mc', "GeometryFileRcd"               , connectionString, "Extended", "2022-01-21 12:00:00.000"] )
    HCALParameters_Geometry_ddd =  ','.join( ['HCALParameters_Geometry_112YV2'         , "HcalParametersRcd"             , connectionString, ""        , "2021-09-28 12:00:00.000"] )
    TKRECO_Geometry_ddd         =  ','.join( ['TKRECO_Geometry_120YV2'                 , "IdealGeometryRecord"           , connectionString, ""        , "2021-09-28 12:00:00.000"] )
    CTRECO_Geometry_ddd         =  ','.join( ['CTRECO_Geometry_112YV2'                 , "PCaloTowerRcd"                 , connectionString, ""        , "2021-09-28 12:00:00.000"] )
    EBRECO_Geometry_ddd         =  ','.join( ['EBRECO_Geometry_112YV2'                 , "PEcalBarrelRcd"                , connectionString, ""        , "2021-09-28 12:00:00.000"] )
    EERECO_Geometry_ddd         =  ','.join( ['EERECO_Geometry_112YV2'                 , "PEcalEndcapRcd"                , connectionString, ""        , "2021-09-28 12:00:00.000"] )
    EPRECO_Geometry_ddd         =  ','.join( ['EPRECO_Geometry_112YV2'                 , "PEcalPreshowerRcd"             , connectionString, ""        , "2021-09-28 12:00:00.000"] )
    HCALRECO_Geometry_ddd       =  ','.join( ['HCALRECO_Geometry_112YV2'               , "PHcalRcd"                      , connectionString, ""        , "2021-09-28 12:00:00.000"] )
    TKParameters_Geometry_ddd   =  ','.join( ['TKParameters_Geometry_112YV2'           , "PTrackerParametersRcd"         , connectionString, ""        , "2021-09-28 12:00:00.000"] )
    ZDCRECO_Geometry_ddd        =  ','.join( ['ZDCRECO_Geometry_112YV2'                , "PZdcRcd"                       , connectionString, ""        , "2021-09-28 12:00:00.000"] )
    RPCRECO_Geometry_ddd        =  ','.join( ['RPCRECO_Geometry_123YV1'                , "RPCRecoGeometryRcd"            , connectionString, ""        , "2022-01-21 12:00:00.000"] )
    PPSRECO_Geometry_ddd        =  ','.join( ['PPSRECO_Geometry_121YV2_2021_mc'        , "VeryForwardIdealGeometryRecord", connectionString, ""        , "2021-12-02 12:00:00.000"] )

    for key,val in autoCond.items():
        if 'phase1_202' in key:    # modification of the DDD relval GT
            GlobalTagsDDD[key+'_ddd'] = (autoCond[key],
                                         CSCRECODIGI_Geometry_ddd,
                                         CSCRECO_Geometry_ddd,
                                         DTRECO_Geometry_ddd,
                                         GEMRECO_Geometry_ddd,
                                         XMLFILE_Geometry_ddd,
                                         HCALParameters_Geometry_ddd,
                                         TKRECO_Geometry_ddd,
                                         CTRECO_Geometry_ddd,
                                         EBRECO_Geometry_ddd,
                                         EERECO_Geometry_ddd,
                                         EPRECO_Geometry_ddd,
                                         HCALRECO_Geometry_ddd,
                                         TKParameters_Geometry_ddd,
                                         ZDCRECO_Geometry_ddd,
                                         RPCRECO_Geometry_ddd,
                                         PPSRECO_Geometry_ddd)
    autoCond.update(GlobalTagsDDD)
    return autoCond

def autoCond2017ppRef5TeV(autoCond):

    GlobalTag2017ppRef5TeV  = {}
    # substitute tags needed for 2017 ppRef 5 TeV GT
    BeamSpotObjects_2017ppRef5TeV           =  ','.join( ['BeamSpotObjects_pp_2017G_MC_2021sample_for_UL' , "BeamSpotObjectsRcd",           connectionString, "", "2021-10-28 12:00:00.000"] )
    EcalLaserAPDPNRatios_2017ppRef5TeV      =  ','.join( ['EcalLaserAPDPNRatios_mc_Run2017G_306580'       , "EcalLaserAPDPNRatiosRcd",      connectionString, "", "2021-10-28 12:00:00.000"] )
    EcalPedestals_2017ppRef5TeV             =  ','.join( ['EcalPedestals_Run2017G_306580'                 , "EcalPedestalsRcd",             connectionString, "", "2021-10-28 12:00:00.000"] )
    EcalTPGLinearizationConst_2017ppRef5TeV =  ','.join( ['EcalTPGLinearizationConst_Run2017G_306580'     , "EcalTPGLinearizationConstRcd", connectionString, "", "2021-10-28 12:00:00.000"] )
    EcalTPGPedestals_2017ppRef5TeV          =  ','.join( ['EcalTPGPedestals_Run2017G_306580'              , "EcalTPGPedestalsRcd",          connectionString, "", "2021-10-28 12:00:00.000"] )
    L1Menu_2017ppRef5TeV                    =  ','.join( ['L1Menu_pp502Collisions2017_v4_m6_xml'          , "L1TUtmTriggerMenuRcd",         connectionString, "", "2021-10-28 12:00:00.000"] )

    for key,val in autoCond.items():
        if 'phase1_2017_realistic' in key:
            GlobalTag2017ppRef5TeV[key+'_ppref'] = (autoCond[key],
                                         BeamSpotObjects_2017ppRef5TeV,
                                         EcalLaserAPDPNRatios_2017ppRef5TeV,
                                         EcalPedestals_2017ppRef5TeV,
                                         EcalTPGLinearizationConst_2017ppRef5TeV,
                                         EcalTPGPedestals_2017ppRef5TeV,
                                         L1Menu_2017ppRef5TeV)
    autoCond.update(GlobalTag2017ppRef5TeV)
    return autoCond

def autoCondRelValForRun2(autoCond):

    GlobalTagRelValForRun2 = {}
    L1GtTriggerMenuForRelValForRun2 =    ','.join( ['L1Menu_Collisions2015_25nsStage1_v5' , "L1GtTriggerMenuRcd",             connectionString, "", "2023-01-28 12:00:00.000"] )
    L1TUtmTriggerMenuForRelValForRun2 =  ','.join( ['L1Menu_Collisions2018_v2_1_0-d1_xml' , "L1TUtmTriggerMenuRcd",           connectionString, "", "2023-01-28 12:00:00.000"] )

    for key,val in autoCond.items():
        if 'run2_data' in key or 'run2_hlt' in key:
            GlobalTagRelValForRun2[key+'_relval'] = (autoCond[key],
                                         L1GtTriggerMenuForRelValForRun2,
                                         L1TUtmTriggerMenuForRelValForRun2)
    autoCond.update(GlobalTagRelValForRun2)
    return autoCond

def autoCondRelValForRun3(autoCond):

    GlobalTagRelValForRun3 = {}
    L1GtTriggerMenuForRelValForRun3 =    ','.join( ['L1Menu_Collisions2015_25nsStage1_v5' , "L1GtTriggerMenuRcd",             connectionString, "", "2023-01-28 12:00:00.000"] )
    L1TUtmTriggerMenuForRelValForRun3 =  ','.join( ['L1Menu_Collisions2023_v1_1_0-v2_xml' , "L1TUtmTriggerMenuRcd",           connectionString, "", "2023-05-02 12:00:00.000"] )

    for key,val in autoCond.items():
        if 'run3_data' in key or 'run3_hlt' in key :
            GlobalTagRelValForRun3[key+'_relval'] = (autoCond[key],
                                         L1GtTriggerMenuForRelValForRun3,
                                         L1TUtmTriggerMenuForRelValForRun3)
    autoCond.update(GlobalTagRelValForRun3)
    return autoCond

