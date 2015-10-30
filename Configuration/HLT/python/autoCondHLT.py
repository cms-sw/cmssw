# https://cms-conddb.cern.ch/browser/#search

#default value for all L1T menus
connectionString = "frontier://FrontierProd/CMS_CONDITIONS"
l1MenuRecord = "L1GtTriggerMenuRcd"
l1MenuLabel = ""

#The snapshot time has been set as starting point as the one of PR 12095.
#Next time you change the customisations, change also the snapshot time in the affected tuple,
#and leave unchanged the snapshot times for the other tuples.

l1Menus= {
    'Fake'         : ( ','.join( [ 'L1GtTriggerMenu_L1Menu_Collisions2012_v3_mc', l1MenuRecord, connectionString, l1MenuLabel, "2015-10-26 12:00:00.000"] ), ),
    'FULL'         : ( ','.join( [ 'L1Menu_Collisions2015_25nsStage1_v5', l1MenuRecord, connectionString, l1MenuLabel, "2015-10-26 12:00:00.000"] ), ),
    'GRun'         : ( ','.join( [ 'L1Menu_Collisions2015_25nsStage1_v5', l1MenuRecord, connectionString, l1MenuLabel, "2015-10-26 12:00:00.000"] ), ),
    '25ns14e33_v1' : ( ','.join( [ 'L1Menu_Collisions2015_25ns_v2_L1T_Scales_20141121_Imp0_0x1030', l1MenuRecord, connectionString, l1MenuLabel, "2015-10-26 12:00:00.000"] ), ),
    '25ns14e33_v3' : ( ','.join( [ 'L1Menu_Collisions2015_25nsStage1_v4', l1MenuRecord, connectionString, l1MenuLabel, "2015-10-26 12:00:00.000"] ), ),
    '25ns14e33_v4' : ( ','.join( [ 'L1Menu_Collisions2015_25nsStage1_v5', l1MenuRecord, connectionString, l1MenuLabel, "2015-10-26 12:00:00.000"] ), ),
    'HIon'         : ( ','.join( [ 'L1Menu_CollisionsHeavyIons2011_v0_nobsc_notau_centrality_q2_singletrack.v1', l1MenuRecord, connectionString, l1MenuLabel, "2015-10-26 12:00:00.000"] ), ),
    'PIon'         : ( ','.join( [ 'L1Menu_Collisions2015_25nsStage1_v5', l1MenuRecord, connectionString, l1MenuLabel, "2015-10-26 12:00:00.000"] ), ),
    '50nsGRun'     : ( ','.join( [ 'L1Menu_Collisions2015_50nsGct_v4', l1MenuRecord, connectionString, l1MenuLabel, "2015-10-26 12:00:00.000"] ), ),
    '50ns_5e33_v1' : ( ','.join( [ 'L1Menu_Collisions2015_50nsGct_v1_L1T_Scales_20141121_Imp0_0x1030', l1MenuRecord, connectionString, l1MenuLabel, "2015-10-26 12:00:00.000"] ), ),
    '50ns_5e33_v3' : ( ','.join( [ 'L1Menu_Collisions2015_50nsGct_v4', l1MenuRecord, connectionString, l1MenuLabel, "2015-10-26 12:00:00.000"] ), ),
    'LowPU'        : ( ','.join( [ 'L1Menu_Collisions2015_lowPU_v4', l1MenuRecord, connectionString, l1MenuLabel, "2015-10-26 12:00:00.000"] ), ),
    '25nsLowPU'    : ( ','.join( [ 'L1Menu_Collisions2015_lowPU_25nsStage1_v6', l1MenuRecord, connectionString, l1MenuLabel, "2015-10-26 12:00:00.000"] ), ),
}

hltGTs = {

#   'symbolic GT'            : ('base GT',[('payload1',payload2')])

    'run1_mc_Fake'           : ('run1_mc'      ,l1Menus['Fake']),
    'run2_mc_FULL'           : ('run2_mc'      ,l1Menus['FULL']),
    'run2_mc_GRun'           : ('run2_mc'      ,l1Menus['GRun']),
    'run2_mc_25ns14e33_v1'   : ('run2_mc'      ,l1Menus['25ns14e33_v1']),
    'run2_mc_25ns14e33_v3'   : ('run2_mc'      ,l1Menus['25ns14e33_v3']),
    'run2_mc_25ns14e33_v4'   : ('run2_mc'      ,l1Menus['25ns14e33_v4']),
    'run2_mc_HIon'           : ('run2_mc_hi'   ,l1Menus['HIon']),
    'run2_mc_PIon'           : ('run2_mc'      ,l1Menus['PIon']),
    'run2_mc_50nsGRun'       : ('run2_mc_50ns' ,l1Menus['50nsGRun']),
    'run2_mc_50ns_5e33_v1'   : ('run2_mc_50ns' ,l1Menus['50ns_5e33_v1']),
    'run2_mc_50ns_5e33_v3'   : ('run2_mc_50ns' ,l1Menus['50ns_5e33_v3']),
    'run2_mc_LowPU'          : ('run2_mc_50ns' ,l1Menus['LowPU']),       
    'run2_mc_25nsLowPU'      : ('run2_mc'      ,l1Menus['25nsLowPU']),       

    'run1_hlt_Fake'          : ('run1_hlt'     ,l1Menus['Fake']),
    'run2_hlt_FULL'          : ('run2_hlt'     ,l1Menus['FULL']),
    'run2_hlt_GRun'          : ('run2_hlt'     ,l1Menus['GRun']),
    'run2_hlt_25ns14e33_v1'  : ('run2_hlt'     ,l1Menus['25ns14e33_v1']),
    'run2_hlt_25ns14e33_v3'  : ('run2_hlt'     ,l1Menus['25ns14e33_v3']),
    'run2_hlt_25ns14e33_v4'  : ('run2_hlt'     ,l1Menus['25ns14e33_v4']),
    'run2_hlt_HIon'          : ('run2_hlt'     ,l1Menus['HIon']),
    'run2_hlt_PIon'          : ('run2_hlt'     ,l1Menus['PIon']),
    'run2_hlt_50nsGRun'      : ('run2_hlt'     ,l1Menus['50nsGRun']),
    'run2_hlt_50ns_5e33_v1'  : ('run2_hlt'     ,l1Menus['50ns_5e33_v1']),
    'run2_hlt_50ns_5e33_v3'  : ('run2_hlt'     ,l1Menus['50ns_5e33_v3']),
    'run2_hlt_LowPU'         : ('run2_hlt'     ,l1Menus['LowPU']),
    'run2_hlt_25nsLowPU'     : ('run2_hlt'     ,l1Menus['25nsLowPU']),       

    'run1_data_Fake'         : ('run1_data'    ,l1Menus['Fake']),
    'run2_data_FULL'         : ('run2_data'    ,l1Menus['FULL']),
    'run2_data_GRun'         : ('run2_data'    ,l1Menus['GRun']),
    'run2_data_25ns14e33_v1' : ('run2_data'    ,l1Menus['25ns14e33_v1']),
    'run2_data_25ns14e33_v3' : ('run2_data'    ,l1Menus['25ns14e33_v3']),
    'run2_data_25ns14e33_v4' : ('run2_data'    ,l1Menus['25ns14e33_v4']),
    'run2_data_HIon'         : ('run2_data'    ,l1Menus['HIon']),
    'run2_data_PIon'         : ('run2_data'    ,l1Menus['PIon']),
    'run2_data_50nsGRun'     : ('run2_data'    ,l1Menus['50nsGRun']),
    'run2_data_50ns_5e33_v1' : ('run2_data'    ,l1Menus['50ns_5e33_v1']),
    'run2_data_50ns_5e33_v3' : ('run2_data'    ,l1Menus['50ns_5e33_v3']),
    'run2_data_LowPU'        : ('run2_data'    ,l1Menus['LowPU']),
    'run2_data_25nsLowPU'    : ('run2_data'    ,l1Menus['25nsLowPU']),       

}

def autoCondHLT(autoCond):
    for key,val in hltGTs.iteritems():
        if len(val)==1 :
           autoCond[key] = ( autoCond[val[0]] )
        else:
           autoCond[key] = ( autoCond[val[0]],) + val[1]
    return autoCond
