
hltGTs = {

#   'symbolic GT'            : ('base GT',[('payload1',payload2')])

    'run1_mc_Fake'           : ('run1_mc',),
    'run1_hlt_Fake'          : ('run1_hlt',),
    'run1_data_Fake'         : ('run1_data',),

    'run2_mc_FULL'           : ('run2_mc',),
    'run2_mc_GRun'           : ('run2_mc',),
    'run2_mc_25ns14e33_v1'   : ('run2_mc',),
    'run2_mc_25ns14e33_v2'   : ('run2_mc',),
    'run2_mc_HIon'           : ('run2_mc_hi',),
    'run2_mc_PIon'           : ('run2_mc',),
    'run2_mc_50nsGRun'       : ('run2_mc_50ns',),
    'run2_mc_50ns_5e33_v1'   : ('run2_mc_50ns',),
    'run2_mc_50ns_5e33_v2'   : ('run2_mc_50ns',),
    'run2_mc_LowPU'          : ('run2_mc_50ns',),

    'run2_hlt_FULL'          : ('run2_hlt',),
    'run2_hlt_GRun'          : ('run2_hlt',),
    'run2_hlt_25ns14e33_v1'  : ('run2_hlt',),
    'run2_hlt_25ns14e33_v2'  : ('run2_hlt',),
    'run2_hlt_HIon'          : ('run2_hlt',),
    'run2_hlt_PIon'          : ('run2_hlt',),
    'run2_hlt_50nsGRun'      : ('run2_hlt',),
    'run2_hlt_50ns_5e33_v1'  : ('run2_hlt',),
    'run2_hlt_50ns_5e33_v2'  : ('run2_hlt',),
    'run2_hlt_LowPU'         : ('run2_hlt',),

    'run2_data_FULL'         : ('run2_data',),
    'run2_data_GRun'         : ('run2_data',),
    'run2_data_25ns14e33_v1' : ('run2_data',),
    'run2_data_25ns14e33_v2' : ('run2_data',),
    'run2_data_HIon'         : ('run2_data',),
    'run2_data_PIon'         : ('run2_data',),
    'run2_data_50nsGRun'     : ('run2_data',),
    'run2_data_50ns_5e33_v1' : ('run2_data',),
    'run2_data_50ns_5e33_v2' : ('run2_data',),
    'run2_data_LowPU'        : ('run2_data',),

}

def autoCondHLT(autoCond):
    for key,val in hltGTs.iteritems():
        if len(val)==1 :
           autoCond[key] = ( autoCond[val[0]] )
        else:
           autoCond[key] = ( autoCond[val[0]],) + val[1]
    return autoCond
