# https://cms-conddb.cern.ch/browser/#search
# default value for all L1T menus
connectionString = "frontier://FrontierProd/CMS_CONDITIONS"

# L1T legacy (Fake) / stage-1 (Fake1)
l1MenuRecord = "L1GtTriggerMenuRcd"
l1MenuLabel = ""

# L1T stage-2
l1tMenuRecord = "L1TUtmTriggerMenuRcd"
l1tMenuLabel = ""

#The snapshot time has been set as starting point as the one of PR 12095.
#Next time you change the customisations, change also the snapshot time in the affected tuple,
#and leave unchanged the snapshot times for the other tuples.

l1Menus = {
    'Fake'         : ( ','.join( [ 'L1GtTriggerMenu_L1Menu_Collisions2012_v3_mc'             , l1MenuRecord,connectionString, l1MenuLabel, "2015-10-26 12:00:00.000"] ), ),
    'Fake1'        : ( ','.join( [ 'L1Menu_Collisions2015_25nsStage1_v5'                     , l1MenuRecord,connectionString, l1MenuLabel, "2015-10-26 12:00:00.000"] ), ),
    'Fake2'        : ( ','.join( [ 'L1Menu_Collisions2016_v9_m2_xml'                         ,l1tMenuRecord,connectionString,l1tMenuLabel, "2016-10-06 19:36:53.000"] ), ),
    'FULL'         : ( ','.join( [ 'L1Menu_Collisions2024_v1_2_0_xml'                        ,l1tMenuRecord,connectionString,l1tMenuLabel, "2024-04-24 12:00:00.000"] ), ),
    'GRun'         : ( ','.join( [ 'L1Menu_Collisions2024_v1_2_0_xml'                        ,l1tMenuRecord,connectionString,l1tMenuLabel, "2024-04-24 12:00:00.000"] ), ),
    '2024v13'      : ( ','.join( [ 'L1Menu_Collisions2024_v1_2_0_xml'                        ,l1tMenuRecord,connectionString,l1tMenuLabel, "2024-04-24 12:00:00.000"] ), ),
    'HIon'         : ( ','.join( [ 'L1Menu_CollisionsHeavyIons2023_v1_1_5_xml'               ,l1tMenuRecord,connectionString,l1tMenuLabel, "2023-10-24 16:59:55.000"] ), ),
    'PIon'         : ( ','.join( [ 'L1Menu_HeavyIons2016_v3_m2_xml'                          ,l1tMenuRecord,connectionString,l1tMenuLabel, "2016-11-22 11:11:00.000"] ), ),
    'PRef'         : ( ','.join( [ 'L1Menu_CollisionsPPRef2023_v1_1_2_xml'                   ,l1tMenuRecord,connectionString,l1tMenuLabel, "2023-09-21 19:00:00.000"] ), ),
    'Special'      : ( ','.join( [ 'L1Menu_Collisions2024_v1_2_0_xml'                        ,l1tMenuRecord,connectionString,l1tMenuLabel, "2024-04-24 12:00:00.000"] ), ),
}

hltGTs = {

#   'symbolic GT'            : ('base GT',[('payload1',payload2')])

    'run1_mc_Fake'           : ('run1_mc'                 ,l1Menus['Fake']),
    'run2_mc_Fake'           : ('run2_mc'                 ,l1Menus['Fake']),
    'run2_mc_Fake1'          : ('run2_mc_l1stage1'        ,l1Menus['Fake1']),
    'run2_mc_Fake2'          : ('run2_mc'                 ,l1Menus['Fake2']),
    'run3_mc_FULL'           : ('phase1_2024_realistic'   ,l1Menus['FULL']),
    'run3_mc_GRun'           : ('phase1_2024_realistic'   ,l1Menus['GRun']),
    'run3_mc_2024v13'        : ('phase1_2024_realistic'   ,l1Menus['2024v13']),
    'run3_mc_HIon'           : ('phase1_2023_realistic_hi',l1Menus['HIon']),
    'run3_mc_PIon'           : ('phase1_2024_realistic'   ,l1Menus['PIon']),
    'run3_mc_PRef'           : ('phase1_2024_realistic'   ,l1Menus['PRef']),
    'run3_mc_Special'        : ('phase1_2024_realistic'   ,l1Menus['Special']),

    'run1_hlt_Fake'          : ('run2_hlt_relval'         ,l1Menus['Fake']),
    'run2_hlt_Fake'          : ('run2_hlt_relval'         ,l1Menus['Fake']),
    'run2_hlt_Fake1'         : ('run2_hlt_relval'         ,l1Menus['Fake1']),
    'run2_hlt_Fake2'         : ('run2_hlt_relval'         ,l1Menus['Fake2']),
    'run3_hlt_FULL'          : ('run3_hlt'                ,l1Menus['FULL']),
    'run3_hlt_GRun'          : ('run3_hlt'                ,l1Menus['GRun']),
    'run3_hlt_2024v13'       : ('run3_hlt'                ,l1Menus['2024v13']),
    'run3_hlt_HIon'          : ('run3_hlt'                ,l1Menus['HIon']),
    'run3_hlt_PIon'          : ('run3_hlt'                ,l1Menus['PIon']),
    'run3_hlt_PRef'          : ('run3_hlt'                ,l1Menus['PRef']),
    'run3_hlt_Special'       : ('run3_hlt'                ,l1Menus['Special']),

    'run1_data_Fake'         : ('run2_data'               ,l1Menus['Fake']),
    'run2_data_Fake'         : ('run2_data'               ,l1Menus['Fake']),
    'run2_data_Fake1'        : ('run2_data'               ,l1Menus['Fake1']),
    'run2_data_Fake2'        : ('run2_data'               ,l1Menus['Fake2']),
    'run3_data_FULL'         : ('run3_data_prompt'        ,l1Menus['FULL']),
    'run3_data_GRun'         : ('run3_data_prompt'        ,l1Menus['GRun']),
    'run3_data_2024v13'      : ('run3_data_prompt'        ,l1Menus['2024v13']),
    'run3_data_HIon'         : ('run3_data_prompt'        ,l1Menus['HIon']),
    'run3_data_PIon'         : ('run3_data_prompt'        ,l1Menus['PIon']),
    'run3_data_PRef'         : ('run3_data_prompt'        ,l1Menus['PRef']),
    'run3_data_Special'      : ('run3_data_prompt'        ,l1Menus['Special']),

}

def autoCondHLT(autoCond):
    for key,val in hltGTs.items():
        if len(val)==1 :
           autoCond[key] = ( autoCond[val[0]] )
        else:
           autoCond[key] = ( autoCond[val[0]],) + val[1]

    return autoCond
