# https://cms-conddb.cern.ch/browser/#search

l1Menus= {
    'Fake'         : 'L1GtTriggerMenu_L1Menu_Collisions2012_v3_mc',
    'FULL'         : 'L1Menu_Collisions2015_25nsStage1_v5',
    'GRun'         : 'L1Menu_Collisions2015_25nsStage1_v5',
    '25ns14e33_v1' : 'L1Menu_Collisions2015_25ns_v2_L1T_Scales_20141121_Imp0_0x1030',
    '25ns14e33_v3' : 'L1Menu_Collisions2015_25nsStage1_v4',
    '25ns14e33_v4' : 'L1Menu_Collisions2015_25nsStage1_v5',
    'HIon'         : 'L1Menu_CollisionsHeavyIons2011_v0_nobsc_notau_centrality_q2_singletrack.v1',
    'PIon'         : 'L1Menu_Collisions2015_25nsStage1_v5',
    '50nsGRun'     : 'L1Menu_Collisions2015_50nsGct_v4',
    '50ns_5e33_v1' : 'L1Menu_Collisions2015_50nsGct_v1_L1T_Scales_20141121_Imp0_0x1030',
    '50ns_5e33_v3' : 'L1Menu_Collisions2015_50nsGct_v4',
    'LowPU'        : 'L1Menu_Collisions2015_lowPU_v4',
    '25nsLowPU'    : 'L1Menu_Collisions2015_lowPU_25nsStage1_v6',
}

for key,val in l1Menus.iteritems():
    l1Menus[key] = (val+',L1GtTriggerMenuRcd,frontier://FrontierProd/CMS_CONDITIONS',)

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
