autoCond = { 
    'mc'        : 'MC_42_V17::All', # This is not good for 2010 MC production
    'startup'   : 'START42_V17A::All', # For 2010 MC production
    'com10'     : 'FT_R_42_V10A::All', # Use for 2010 data reprocessing
     # 'hltonline' should be the same as same as 'com10' until a compatible GR_H tag is available, then it should point to the GR_H tag and override the connection string and pfnPrefix for use offline
    'hltonline' : 'FT_R_42_V10A::All,frontier://FrontierProd/CMS_COND_31X_GLOBALTAG,frontier://FrontierProd/',
    'hltonline11' : ('FT_R_42_V10A::All,frontier://FrontierProd/CMS_COND_31X_GLOBALTAG,frontier://FrontierProd/','L1GtTriggerMenu_L1Menu_Collisions2012_v0_mc,L1GtTriggerMenuRcd,frontier://FrontierProd/CMS_COND_31X_L1T'),
    'starthi'   : 'STARTHI53_V3::All'
}
