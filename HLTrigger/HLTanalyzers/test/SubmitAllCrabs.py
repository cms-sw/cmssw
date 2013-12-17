import math, os, sys, string
from array import *

def OpenFile(file_in,iodir):
    """  file_in -- Input file name
         iodir   -- 'r' readonly  'r+' read+write """
    try:
        ifile=open(file_in, iodir)
        # print "Opened file: ",file_in," iodir ",iodir
    except:
        print "Could not open file: ",file_in
        sys.exit(1)
    return ifile

def CloseFile(ifile):
    ifile.close()

dircrab = '/uscms_data/d3/ingabu/TMD/CMSSW_5_3_11/src/HLTrigger/HLTanalyzers/test/CrabCfgs/'

cfgsEM_13 = ['crab_EM_20to30_13.cfg', 'crab_EM_30to80_13.cfg', 'crab_EM_80to170_13.cfg'] 
cfgsEM_825 = ['crab_EM_20to30_825.cfg', 'crab_EM_30to80_825.cfg', 'crab_EM_80to170_825.cfg']
cfgsEM_850 = ['crab_EM_20to30_850.cfg', 'crab_EM_30to80_850.cfg', 'crab_EM_80to170_850.cfg']

cfgsMu_13 = ['crab_Mu_15to20_antiEM_13.cfg', 'crab_Mu_20to30_antiEM_13.cfg', 'crab_Mu_30to50_antiEM_13.cfg', 'crab_Mu_50to80_antiEM_13.cfg', 'crab_Mu_80to120_antiEM_13.cfg']
cfgsMu_825 = ['crab_Mu_15to20_antiEM_825.cfg', 'crab_Mu_20to30_antiEM_825.cfg', 'crab_Mu_30to50_antiEM_825.cfg', 'crab_Mu_50to80_antiEM_825.cfg', 'crab_Mu_80to120_antiEM_825.cfg']
cfgsMu_850 = ['crab_Mu_15to20_antiEM_850.cfg', 'crab_Mu_20to30_antiEM_850.cfg', 'crab_Mu_30to50_antiEM_850.cfg', 'crab_Mu_50to80_antiEM_850.cfg', 'crab_Mu_80to120_antiEM_850.cfg']

cfgsQCD_antiEM_13 = ['crab_QCD_15to30_antiEM_13.cfg', 'crab_QCD_30to50_antiEM_13.cfg', 'crab_QCD_50to80_antiEM_13.cfg', 'crab_QCD_80to120_antiEM_13.cfg', 'crab_QCD_120to170_antiEM_13.cfg']
cfgsQCD_antiEM_825 = ['crab_QCD_15to30_antiEM_825.cfg', 'crab_QCD_30to50_antiEM_825.cfg', 'crab_QCD_50to80_antiEM_825.cfg', 'crab_QCD_80to120_antiEM_825.cfg', 'crab_QCD_120to170_antiEM_825.cfg']
cfgsQCD_antiEM_850 = ['crab_QCD_15to30_antiEM_850.cfg', 'crab_QCD_30to50_antiEM_850.cfg', 'crab_QCD_50to80_antiEM_850.cfg', 'crab_QCD_80to120_antiEM_850.cfg', 'crab_QCD_120to170_antiEM_850.cfg']

cfgsQCD_nofilt_13 = ['crab_QCD_15to30_nofilt_13.cfg', 'crab_QCD_30to50_nofilt_13.cfg', 'crab_QCD_50to80_nofilt_13.cfg', 'crab_QCD_80to120_nofilt_13.cfg', 'crab_QCD_120to170_nofilt_13.cfg', 'crab_QCD_170to300_nofilt_13.cfg', 'crab_QCD_300to470_nofilt_13.cfg', 'crab_QCD_470to600_nofilt_13.cfg', 'crab_QCD_600to800_nofilt_13.cfg', 'crab_QCD_800to1000_nofilt_13.cfg', 'crab_QCD_1000to1400_nofilt_13.cfg', 'crab_QCD_1400to1800_nofilt_13.cfg']
cfgsQCD_nofilt_825 = ['crab_QCD_15to30_nofilt_825.cfg', 'crab_QCD_30to50_nofilt_825.cfg', 'crab_QCD_50to80_nofilt_825.cfg', 'crab_QCD_80to120_nofilt_825.cfg', 'crab_QCD_120to170_nofilt_825.cfg', 'crab_QCD_170to300_nofilt_825.cfg', 'crab_QCD_300to470_nofilt_825.cfg', 'crab_QCD_470to600_nofilt_825.cfg', 'crab_QCD_600to800_nofilt_825.cfg', 'crab_QCD_800to1000_nofilt_825.cfg', 'crab_QCD_1000to1400_nofilt_825.cfg', 'crab_QCD_1400to1800_nofilt_825.cfg']
cfgsQCD_nofilt_850 = ['crab_QCD_15to30_nofilt_850.cfg', 'crab_QCD_30to50_nofilt_850.cfg', 'crab_QCD_50to80_nofilt_850.cfg', 'crab_QCD_80to120_nofilt_850.cfg', 'crab_QCD_120to170_nofilt_850.cfg', 'crab_QCD_170to300_nofilt_850.cfg', 'crab_QCD_300to470_nofilt_850.cfg', 'crab_QCD_470to600_nofilt_850.cfg', 'crab_QCD_600to800_nofilt_850.cfg', 'crab_QCD_800to1000_nofilt_850.cfg', 'crab_QCD_1000to1400_nofilt_850.cfg', 'crab_QCD_1400to1800_nofilt_850.cfg']


#for i in range(len(cfgsEM_13)):
#for i in range(len(cfgsEM_825)):
#for i in range(len(cfgsEM_850)): 
for i in range(len(cfgsMu_13)):
#for i in range(len(cfgsMu_825)):
#for i in range(len(cfgsMu_850)): 
#for i in range(len(cfgsQCD_antiEM_13)):
#for i in range(len(cfgsQCD_antiEM_825)):
#for i in range(len(cfgsQCD_antiEM_850)): 
#for i in range(len(cfgsQCD_nofilt_13)):
#for i in range(len(cfgsQCD_nofilt_825)):
#for i in range(len(cfgsQCD_nofilt_850)):    
    #cfg = cfgsEM_13[i]   
    #cfg = cfgsEM_825[i]   
    #cfg = cfgsEM_850[i]  
    cfg = cfgsMu_13[i]   
    #cfg = cfgsMu_825[i]   
    #cfg = cfgsMu_850[i]
    #cfg = cfgsQCD_antiEM_13[i]   
    #cfg = cfgsQCD_antiEM_825[i]   
    #cfg = cfgsQCD_antiEM_850[i]
    #cfg = cfgsQCD_nofilt_13[i]   
    #cfg = cfgsQCD_nofilt_825[i]   
    #cfg = cfgsQCD_nofilt_850[i]
    ifile = os.path.join(dircrab,cfg)
    print "ifile = ", ifile
    infile = OpenFile(ifile, 'r')
    iline = 0
    workdir = ""
    x = infile.readline()
    while x != "":
        iline+=1
        xx=string.rstrip(x)
        if xx.find("ui_working_dir=")>-1:
            workdir = xx.split("=")[-1] 
        x = infile.readline()

    print "workdir = ", workdir
    crabcr = 'crab -create -cfg' + ' ' + str(ifile)
    crabsub = 'crab -submit -c' + str(workdir)
    os.system(crabcr)
    os.system(crabsub)

    CloseFile(infile)
