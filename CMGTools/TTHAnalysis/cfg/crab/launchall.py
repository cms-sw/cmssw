import imp, os

# datasets to run as defined from run_susyMT2.cfg
# number of jobs to run per dataset decided based on splitFactor and fineSplitFactor from cfg file
# in principle one only needs to modify the following two lines:
production_label = "prod74Xdata_noJSON"
cmg_version = 'MT2_CMGTools-from-CMSSW_7_4_3'

debug  = False
useAAA = False

handle = open("heppy_config.py", 'r')
cfo = imp.load_source("heppy_config", "heppy_config.py", handle)
conf = cfo.config
handle.close()

os.system("scramv1 runtime -sh")
os.system("source /cvmfs/cms.cern.ch/crab3/crab.sh")

os.environ["PROD_LABEL"]  = production_label
os.environ["CMG_VERSION"] = cmg_version
os.environ["DEBUG"]       = str(debug)
os.environ["USEAAA"]      = str(useAAA)

from PhysicsTools.HeppyCore.framework.heppy import split
for comp in conf.components:
    # get splitting from config file according to splitFactor and fineSplitFactor (priority given to the latter)
    NJOBS = len(split([comp]))
    os.environ["NJOBS"] = str(NJOBS)
    os.environ["DATASET"] = str(comp.name)
    os.system("crab submit -c heppy_crab_config_env.py")

os.system("rm -f python.tar.gz")
os.system("rm -f cmgdataset.tar.gz")
os.system("rm -f cafpython.tar.gz")
