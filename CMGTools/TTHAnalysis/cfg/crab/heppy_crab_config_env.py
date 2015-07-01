# here we set all crab options that are not fixed
# values we'll be taken from environment variables set in launchall.py
# fixed options will be taken from heppy_crab_config.py

import imp
file = open( "heppy_crab_config.py", 'r' )
cfg = imp.load_source( 'cfg', "heppy_crab_config.py", file)
config = cfg.config
import os
import re
dataset=os.environ["DATASET"]
NJOBS=int(os.environ["NJOBS"])
production_label = os.environ["PROD_LABEL"]
cmg_version = os.environ["CMG_VERSION"]
debug  = os.environ["DEBUG"] == 'True'
useAAA = os.environ["USEAAA"] == 'True'

if debug:
    NJOBS = 4
    NEVENTS = 200

print "Will send dataset", dataset, "with", NJOBS, "jobs"

config.General.requestName = dataset + "_" + cmg_version # task name
config.General.workArea = 'crab_' + production_label # crab dir name

# this will divide task in *exactly* NJOBS jobs (for this we need JobType.pluginName = 'PrivateMC' and Data.splitting = 'EventBased')
config.Data.unitsPerJob = 10
config.Data.totalUnits = config.Data.unitsPerJob * NJOBS

# arguments to pass to scriptExe. They have to be like "arg=value". 
config.JobType.scriptArgs = ["dataset="+dataset, "total="+str(NJOBS), "useAAA="+str(useAAA)]

# output will be .../$outLFN/$PRIMARY_DS/$PUBLISH_NAME/$TIMESTAMP/$COUNTER/$FILENAME
# https://twiki.cern.ch/twiki/bin/view/CMSPublic/Crab3DataHandling
config.Data.outLFNDirBase += '/babies/' + cmg_version
config.Data.primaryDataset =  production_label
config.Data.publishDataName = dataset
#final output: /store/user/$USER/babies/cmg_version/production_label/dataset/150313_114158/0000/foo.bar

# if NEVENTS variable is set then only nevents will be run
try: 
    NEVENTS
except NameError:
    pass
else:
    config.JobType.scriptArgs += ["nevents="+str(NEVENTS)]
