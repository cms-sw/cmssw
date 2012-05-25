# Remarks:
# - 'com10' must always be the GR_R GT
# - 'hltonline' should be the same as same as 'com10' until a compatible GR_H tag is available, then it should point to the GR_H tag and override the connection string and pfnPrefix for use offline
# - 'hltonline11' is the same as above, but it runs over 2011 data. To run the HLT the workflow needs to also rereco the L1 hence the L1 menu must be forced explicitly. The one for MC is used and must
# be kept updated.

autoCond = { 
    'mc'        : 'MC_52_V10::All',
    'startup'   : 'START52_V10::All',
    'com10'     : 'GR_R_52_V7::All', # This should always be the GR_R GT
     # 'hltonline' should be the same as same as 'com10' until a compatible GR_H tag is available, then it should point to the GR_H tag and override the connection string and pfnPrefix for use offline
    'hltonline' : 'GR_H_V29::All,frontier://FrontierProd/CMS_COND_31X_GLOBALTAG,frontier://FrontierProd/',
    'hltonline11' : ('GR_H_V29::All,frontier://FrontierProd/CMS_COND_31X_GLOBALTAG,frontier://FrontierProd/',
                     'L1GtTriggerMenu_L1Menu_Collisions2012_v1a_mc,L1GtTriggerMenuRcd,frontier://FrontierProd/CMS_COND_31X_L1T'),
    'starthi'   : 'STARTHI52_V9::All'
}
