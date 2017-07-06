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

l1Menus= {
    'Fake'         : ( ','.join( [ 'L1GtTriggerMenu_L1Menu_Collisions2012_v3_mc'             , l1MenuRecord,connectionString, l1MenuLabel, "2015-10-26 12:00:00.000"] ), ),
    'Fake1'        : ( ','.join( [ 'L1Menu_Collisions2015_25nsStage1_v5'                     , l1MenuRecord,connectionString, l1MenuLabel, "2015-10-26 12:00:00.000"] ), ),
    'Fake2'        : ( ','.join( [ 'L1Menu_Collisions2017_dev_r5_m4_patch_xml'                         ,l1tMenuRecord,connectionString,l1tMenuLabel, "2017-05-19 10:21:02.000"] ), ),
    'FULL'         : ( ','.join( [ 'L1Menu_Collisions2017_dev_r5_m4_patch_xml'                         ,l1tMenuRecord,connectionString,l1tMenuLabel, "2017-05-19 10:21:02.000"] ), ),
    'GRun'         : ( ','.join( [ 'L1Menu_Collisions2017_dev_r5_m4_patch_xml'                         ,l1tMenuRecord,connectionString,l1tMenuLabel, "2017-05-19 10:21:02.000"] ), ),
    'GRun2016'     : ( ','.join( [ 'L1Menu_Collisions2016_v9_m2_xml'                         ,l1tMenuRecord,connectionString,l1tMenuLabel, "2016-10-06 19:36:53.000"] ), ),
    'HIon'         : ( ','.join( [ 'L1Menu_CollisionsHeavyIons2015_v5_uGT_xml'               ,l1tMenuRecord,connectionString,l1tMenuLabel, "2016-03-04 15:00:00.000"] ), ),
    'PIon'         : ( ','.join( [ 'L1Menu_HeavyIons2016_v3_m2_xml'                          ,l1tMenuRecord,connectionString,l1tMenuLabel, "2016-11-22 11:11:00.000"] ), ),
    'PRef'         : ( ','.join( [ 'L1Menu_Collisions2015_5TeV_pp_reference_v5_uGT_v2_mc_xml',l1tMenuRecord,connectionString,l1tMenuLabel, "2016-03-04 15:00:00.000"] ), ),
}

hltGTs = {

#   'symbolic GT'            : ('base GT',[('payload1',payload2')])

    'run1_mc_Fake'           : ('run1_mc'              ,l1Menus['Fake']),
    'run2_mc_Fake'           : ('run2_mc'              ,l1Menus['Fake']),
    'run2_mc_Fake1'          : ('run2_mc'              ,l1Menus['Fake1']),
    'run2_mc_Fake2'          : ('run2_mc'              ,l1Menus['Fake2']),
    'run2_mc_GRun2016'       : ('run2_mc'              ,l1Menus['GRun2016']),
    'run2_mc_FULL'           : ('phase1_2017_realistic',l1Menus['FULL']),
    'run2_mc_GRun'           : ('phase1_2017_realistic',l1Menus['GRun']),
    'run2_mc_HIon'           : ('run2_mc_hi'           ,l1Menus['HIon']),
    'run2_mc_PIon'           : ('phase1_2017_realistic',l1Menus['PIon']),
    'run2_mc_PRef'           : ('phase1_2017_realistic',l1Menus['PRef']),

    'run1_hlt_Fake'          : ('run1_hlt'             ,l1Menus['Fake']),
    'run2_hlt_Fake'          : ('run2_hlt_relval'      ,l1Menus['Fake']),
    'run2_hlt_Fake1'         : ('run2_hlt_relval'      ,l1Menus['Fake1']),
    'run2_hlt_Fake2'         : ('run2_hlt_relval'      ,l1Menus['Fake2']),
    'run2_hlt_GRun2016'      : ('run2_hlt'             ,l1Menus['GRun2016']),
    'run2_hlt_FULL'          : ('phase1_2017_realistic',l1Menus['FULL']),
    'run2_hlt_GRun'          : ('phase1_2017_realistic',l1Menus['GRun']),
    'run2_hlt_HIon'          : ('run2_hlt_hi'          ,l1Menus['HIon']),
    'run2_hlt_PIon'          : ('phase1_2017_realistic',l1Menus['PIon']),
    'run2_hlt_PRef'          : ('phase1_2017_realistic',l1Menus['PRef']),

    'run1_data_Fake'         : ('run1_data'            ,l1Menus['Fake']),
    'run2_data_Fake'         : ('run2_data_relval'     ,l1Menus['Fake']),
    'run2_data_Fake1'        : ('run2_data_relval'     ,l1Menus['Fake1']),
    'run2_data_Fake2'        : ('run2_data_relval'     ,l1Menus['Fake2']),
    'run2_data_GRun2016'     : ('run2_data'            ,l1Menus['GRun2016']),
    'run2_data_FULL'         : ('phase1_2017_realistic',l1Menus['FULL']),
    'run2_data_GRun'         : ('phase1_2017_realistic',l1Menus['GRun']),
    'run2_data_HIon'         : ('run2_data'            ,l1Menus['HIon']),
    'run2_data_PIon'         : ('phase1_2017_realistic',l1Menus['PIon']),
    'run2_data_PRef'         : ('phase1_2017_realistic',l1Menus['PRef']),
    
    'phase1_2017_design_GRun'     : ('phase1_2017_design'   ,l1Menus['GRun']),
    'phase1_2017_realistic_GRun'  : ('phase1_2017_realistic',l1Menus['GRun']),
}

def autoCondHLT(autoCond):
    for key,val in hltGTs.iteritems():
        if len(val)==1 :
           autoCond[key] = ( autoCond[val[0]] )
        else:
           autoCond[key] = ( autoCond[val[0]],) + val[1]

    return autoCond
