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

def autoCondDD4HEP(autoCond):

    GlobalTagsDDHEP = {}
    # substitute the DDD geometry tags with DD4HEP ones
    CSCRECODIGI_Geometry_dd4hep    =  ','.join( ['CSCRECODIGI_Geometry_120DD4hepV1'            , "CSCRecoDigiParametersRcd", connectionString, "", "2021-09-28 12:00:00.000"] )
    CSCRECO_Geometry_dd4hep        =  ','.join( ['CSCRECO_Geometry_120DD4hepV1'                , "CSCRecoGeometryRcd"      , connectionString, "", "2021-09-28 12:00:00.000"] )
    DTRECO_Geometry_dd4hep         =  ','.join( ['DTRECO_Geometry_120DD4hepV1'                 , "DTRecoGeometryRcd"       , connectionString, "", "2021-09-28 12:00:00.000"] )
    GEMRECO_Geometry_dd4hep        =  ','.join( ['GEMRECO_Geometry_120DD4hepV1'                , "GEMRecoGeometryRcd"      , connectionString, "", "2021-09-28 12:00:00.000"] )
    XMLFILE_Geometry_dd4hep        =  ','.join( ['XMLFILE_Geometry_121DD4hepV1_Extended2021_mc', "GeometryFileRcd"         , connectionString, "Extended", "2021-09-28 12:00:00.000"] )
    HCALParameters_Geometry_dd4hep =  ','.join( ['HCALParameters_Geometry_120DD4hepV1'         , "HcalParametersRcd"       , connectionString, "", "2021-09-28 12:00:00.000"] )
    TKRECO_Geometry_dd4hep         =  ','.join( ['TKRECO_Geometry_121DD4hepV1'                 , "IdealGeometryRecord"     , connectionString, "", "2021-09-28 12:00:00.000"] )
    CTRECO_Geometry_dd4hep         =  ','.join( ['CTRECO_Geometry_120DD4hepV1'                 , "PCaloTowerRcd"           , connectionString, "", "2021-09-28 12:00:00.000"] )
    EBRECO_Geometry_dd4hep         =  ','.join( ['EBRECO_Geometry_120DD4hepV1'                 , "PEcalBarrelRcd"          , connectionString, "", "2021-09-28 12:00:00.000"] )
    EERECO_Geometry_dd4hep         =  ','.join( ['EERECO_Geometry_120DD4hepV1'                 , "PEcalEndcapRcd"          , connectionString, "", "2021-09-28 12:00:00.000"] )
    EPRECO_Geometry_dd4hep         =  ','.join( ['EPRECO_Geometry_120DD4hepV1'                 , "PEcalPreshowerRcd"       , connectionString, "", "2021-09-28 12:00:00.000"] )
    HCALRECO_Geometry_dd4hep       =  ','.join( ['HCALRECO_Geometry_120DD4hepV1'               , "PHcalRcd"                , connectionString, "", "2021-09-28 12:00:00.000"] )
    TKParameters_Geometry_dd4hep   =  ','.join( ['TKParameters_Geometry_120DD4hepV1'           , "PTrackerParametersRcd"   , connectionString, "", "2021-09-28 12:00:00.000"] )
    ZDCRECO_Geometry_dd4hep        =  ','.join( ['ZDCRECO_Geometry_120DD4hepV1'                , "PZdcRcd"                 , connectionString, "", "2021-09-28 12:00:00.000"] )
    RPCRECO_Geometry_dd4hep        =  ','.join( ['RPCRECO_Geometry_120DD4hepV1'                , "RPCRecoGeometryRcd"      , connectionString, "", "2021-09-28 12:00:00.000"] )

    for key,val in autoCond.items():
        if key == 'phase1_2021_realistic':    # modification of the DD4HEP relval GT
            GlobalTagsDDHEP['phase1_2021_dd4hep'] = (autoCond[key],
                                                     CSCRECODIGI_Geometry_dd4hep,
                                                     CSCRECO_Geometry_dd4hep,
                                                     DTRECO_Geometry_dd4hep,
                                                     GEMRECO_Geometry_dd4hep,
                                                     XMLFILE_Geometry_dd4hep,
                                                     HCALParameters_Geometry_dd4hep,
                                                     TKRECO_Geometry_dd4hep,
                                                     CTRECO_Geometry_dd4hep,
                                                     EBRECO_Geometry_dd4hep,
                                                     EERECO_Geometry_dd4hep,
                                                     EPRECO_Geometry_dd4hep,
                                                     HCALRECO_Geometry_dd4hep,
                                                     TKParameters_Geometry_dd4hep,
                                                     ZDCRECO_Geometry_dd4hep,
                                                     RPCRECO_Geometry_dd4hep)
    autoCond.update(GlobalTagsDDHEP)
    return autoCond
