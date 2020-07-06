##
## Append for 0T conditions
##
import six
from Configuration.StandardSequences.CondDBESSource_cff import GlobalTag as essource
connectionString = essource.connect.value()

# method called in autoCond
def autoCond0T(autoCond):
    
    ConditionsFor0T =  ','.join( ['RunInfo_0T_v1_mc', "RunInfoRcd", connectionString, "", "2020-07-01 12:00:00.000"] )
    GlobalTags0T = {}
    for key,val in six.iteritems(autoCond):
        if "phase" in key:    # restric to phase1 upgrade GTs
            GlobalTags0T[key+"_0T"] = (autoCond[key], ConditionsFor0T)

    autoCond.update(GlobalTags0T)

    return autoCond


