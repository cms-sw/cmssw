import imp, os

# datasets to run as defined from run_susyMT2.cfg
# number of jobs to run per dataset decided based on splitFactor and fineSplitFactor from cfg file
# in principle one only needs to modify the following two lines:
production_label = "prod747data_test"
cmg_version = 'MT2_CMGTools-from-CMSSW_7_4_7'

debug  = True
useAAA = True

JSON = "$CMSSW_BASE/src/CMGTools/TTHAnalysis/data/json/json_DCSONLY_Run2015B.txt"
#JSON = "$CMSSW_BASE/src/CMGTools/TTHAnalysis/data/json/Cert_246908-251252_13TeV_PromptReco_Collisions15_JSON.txt"

#recreate cached datasets for data that keep evolving (remove)
os.system("rm ~/.cmgdataset/CMS*PromptReco*")

# update most recent DCS-only json
os.system("cp -f /afs/cern.ch/cms/CAF/CMSCOMM/COMM_DQM/certification/Collisions15/13TeV/DCSOnly/json_DCSONLY_Run2015B.txt ../../data/json/")


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
    if comp.isData and 'JSON' in vars():
        os.environ["JSON"] = JSON
    os.system("crab submit -c heppy_crab_config_env.py")

os.system("rm -f python.tar.gz")
os.system("rm -f cmgdataset.tar.gz")
os.system("rm -f cafpython.tar.gz")
