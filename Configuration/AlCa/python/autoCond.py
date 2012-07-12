autoCond = { 
    'mc'        : 'MC_53_V9::All',
    'startup'   : 'START53_V9::All',
    'com10'     : 'GR_R_53_V9::All',
     # 'hltonline' should be the same as same as 'com10' until a compatible GR_H tag is available, then it should point to the GR_H tag and override the connection string and pfnPrefix for use offline
    'hltonline' : 'GR_R_53_V9::All,frontier://FrontierProd/CMS_COND_31X_GLOBALTAG,frontier://FrontierProd/',
    'hltonline11' : ('GR_R_53_V9::All,frontier://FrontierProd/CMS_COND_31X_GLOBALTAG,frontier://FrontierProd/','L1GtTriggerMenu_L1Menu_Collisions2012_v1a_mc,L1GtTriggerMenuRcd,frontier://FrontierProd/CMS_COND_31X_L1T'),
    'starthi'   : 'STARTHI53_V7::All'
}
