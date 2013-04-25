location = "RPCGEM/GEMPhysics/"

inputFiles = [
    'WHTolnu2mu_M125GeV_14TeV_cff',
    'WHTolnu2tau_M125GeV_14TeV_cff',
    'ZHTo2l2mu_M125GeV_14TeV_cff',
    'HToZZ4l_M125GeV_14TeV_cff',
    'muStarTomuZ3mu_M2500GeV_14TeV_cff',
    'muStarTomuZ3mu_M4000GeV_14TeV_cff',
    'WprimeTomunu_M3000GeV_14TeV_cff',
    'WprimeTomunu_M6000GeV_14TeV_cff',
    'ZprimeTo2mu_M3000GeV_14TeV_cff',
    'ZprimeTo2mu_M6000GeV_14TeV_cff']



events = [100000,100000,100000,200000,50000,50000,50000,50000,50000,50000]

f = open('temp.txt', 'w')

for i in range(len(inputFiles)): 
    inputFile = inputFiles[i]
    f.write('cmsDriver.py %s%s '%(location,inputFile))
    f.write(' -s GEN,SIM,DIGI,L1,DIGI2RAW')
    f.write(' --conditions POSTLS161_V12::All')
    f.write(' --geometry Geometry/GEMGeometry/cmsExtendedGeometryPostLS1plusGEMr08v01XML_cfi')
    f.write(' --datatier GEN-SIM-RAW')
    f.write(' --eventcontent RAWSIM')
    f.write(' --evt_type RPCGEM/GEMPhysics/%s'%(inputFile))
    f.write(' -n %d'%(events[i]))
    f.write(' --no_exec')
    outfile = inputFile.replace( '_cff', '_GEN_SIM_DIGI_L1_DIGI2RAW_PU.root')
    f.write(' --fileout %s'%(outfile) )
    f.write(' --pileup AVE_50_BX_25ns')
    f.write(' --customise SLHCUpgradeSimulations/Configuration/postLS1Customs.customise_Digi\n\n')
    
f.close()
