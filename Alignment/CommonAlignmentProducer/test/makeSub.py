#!/bin/python3
import os 

tmp_sh= """#!/bin/sh
cd {}
eval `scramv1 run -sh`
cmsRun GlobalTrackerMuonAlignment_cfg.py start={} end={} outputGPR={} TBMADB={} inputGPR={} inputGT={} fileList={} DOF={} Global={} Barrel={}
"""

tmp_sub="""executable              = $(filename)
arguments               = $(ClusterId)$(ProcId)
output                  = output/$(filename)$(ProcId).out
error                   = error/$(filename)$(ProcId).err
log                     = log/$(filename)$(ProcId).log
universe                = vanilla
+AccountingGroup = "group_u_CMS.CAF.ALCA"
+MaxRuntime             = 288000
queue filename matching files GPR*.sh
"""

tmp_writeDB="""#!/bin/sh
cd {}
eval `scramv1 run -sh`
cmsRun GlobalTrackerMuonAlignment_writeDB_cfg.py outputGPR={} inputGPR={} inputGT={} fileList={} DOF={}
cp {} ..
"""


#GT = "124X_dataRun3_forReRecoCondition_v1"
GT = "124X_dataRun3_Prompt_TrkAl_w40_2022_v1"
flist = "Run2022D.list"
#TBMADB = "Run3TBMAv3.db"
TBMADB = "Run2022BC_newTRK_v1.db"
inputGPR = "GPR_ReReco_It3.db"
outputGPR = "GPR_ReReco_It4"
DOF = "6"
Global = False
Barrel = True
tDir = outputGPR


os.system("mkdir -p "+tDir)
os.chdir(tDir)
os.system('cp ../{} .'.format(flist))
os.system('cp ../GlobalTrackerMuonAlignment_cfg.py .')
os.system('cp ../GlobalTrackerMuonAlignment_writeDB_cfg.py .')
if len(inputGPR) > 3: os.system('cp ../{} .'.format(inputGPR))
if len(TBMADB) > 3: os.system('cp ../{} .'.format(TBMADB))

path = os.getcwd()

tf = len(open(flist).readlines())
nf = 5
nj = divmod(tf,nf)[0] + 1

os.system("mkdir -p output")
os.system("mkdir -p log")
os.system("mkdir -p error")

for x in range(nj):
  tmp = open("GPR{0:03}.sh".format(x), 'w')
  fname = outputGPR+"{0:03}.db".format(x)
  if x == nj-1: tmp.write(tmp_sh.format(path, str(nf*x), str(-1), fname, TBMADB, inputGPR, GT, flist, DOF, Global, Barrel))
  else: tmp.write(tmp_sh.format(path, str(nf*x), str(nf*x+(nf-1)), fname, TBMADB, inputGPR, GT, flist, DOF, Global, Barrel))

outSub = open("GPR.sub", 'w')
outSub.write(tmp_sub)

outWriteDB = open("writeDB.sh", 'w')
outWriteDB.write(tmp_writeDB.format(path, outputGPR+'.db', inputGPR, GT, flist, DOF, outputGPR+'.db'))
os.system("chmod 755 writeDB.sh")
