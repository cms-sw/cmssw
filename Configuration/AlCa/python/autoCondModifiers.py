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
