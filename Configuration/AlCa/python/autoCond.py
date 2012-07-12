autoCond = { 
    'mc'        : 'MC_60_V3::All',
    'startup'   : 'START60_V3::All',
    'com10'     : 'GR_R_60_V3::All', # This should always be the GR_R GT
     # 'hltonline' should be the same as same as 'com10' until a compatible GR_H tag is available, then it should point to the GR_H tag and override the connection string and pfnPrefix for use offline
    'hltonline' : 'GR_R_60_V3::All,frontier://FrontierProd/CMS_COND_31X_GLOBALTAG,frontier://FrontierProd/',
    'hltonline11' : ('GR_R_60_V3::All,frontier://FrontierProd/CMS_COND_31X_GLOBALTAG,frontier://FrontierProd/',
                     'L1GtTriggerMenu_L1Menu_Collisions2012_v1a_mc,L1GtTriggerMenuRcd,frontier://FrontierProd/CMS_COND_31X_L1T'),
    'starthi'   : 'STARTHI60_V3::All'
}
