autoCond = { 
    'mc'        : 'MC_52_V5::All',
    'startup'   : 'START52_V5::All',
    'com10'     : 'GR_R_52_V7::All',
     # 'hltonline' should be the same as same as 'com10' until a compatible GR_H tag is available, then it should point to the GR_H tag and override the connection string and pfnPrefix for use offline
    'hltonline' : 'GR_H_V28::All,frontier://FrontierProd/CMS_COND_31X_GLOBALTAG,frontier://FrontierProd/',
    'starthi'   : 'STARTHI52_V6::All'
}
