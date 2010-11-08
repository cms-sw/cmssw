import optparse

usage="--list"
parser = optparse.OptionParser(usage)
parser.add_option("--GT")
parser.add_option("--era")
parser.add_option("--release")
parser.add_option("--options",default="")
parser.add_option("--optionsSkim",default="")
parser.add_option("--output",default="RECO,DQM")
parser.add_option("--fullPDs",default=False,action="store_true")
parser.add_option("--PDs",default="")

(options,args)=parser.parse_args()

def Era_alls():
    alcaAndSkimMap={}
    #full list of ever existing PD to reprocess
    PDlist=['MinimumBias',
            'ZeroBias',
            'Commissioning',
            'Cosmics',
            'Mu','MuOnia',
            'EGMonitor','EG','Electron','Photon',
            'JetMETtau','JetMET','Jet','METFwd','BTau',
            'HcalNZS'
            ]
    if options.PDs!="":
        PDlist=options.PDs.split(',')

    from Configuration.Skimming.autoSkim import autoSkim
    from Configuration.PyReleaseValidation.autoAlca import autoAlca
    for PD in PDlist:
        a=''
        s=''
        if PD in autoAlca:
            a=autoAlca[PD]
        if PD in autoSkim:
            s=autoSkim[PD]
        alcaAndSkimMap[PD]=(a,s)

    
    #add a generic one
    alcaAndSkimMap['']=('','LogError')
    
    return alcaAndSkimMap

def Era_12PDs():
    alcaAndSkimMap={}
    if options.fullPDs:
        alcaAndSkimMap=Era_10PDs()
        alcaAndSkimMap.pop('JetMET')
    alcaAndSkimMap['Jet']=('','LogError+DiJet')
    alcaAndSkimMap['METFwd']=('','LogError')
    return alcaAndSkimMap
    
def Era_10PDs():
    alcaAndSkimMap={}
    if options.fullPDs:
        alcaAndSkimMap=Era_8PDs()
        alcaAndSkimMap.pop('JetMETTau')
    alcaAndSkimMap['JetMET']=('','LogError+DiJet')
    alcaAndSkimMap['BTau']=('','LogError','Tau')
    alcaAndSkimMap['MuOnia']=('TkAlJpsiMuMu+TkAlUpsilonMuMu','')
    return alcaAndSkimMap

def Era_8PDs():
    alcaAndSkimMap={}
    alcaAndSkimMap['']=('','LogError')
    alcaAndSkimMap['ZeroBias']=('SiStripCalZeroBias','GoodVtx+LogError')
    alcaAndSkimMap['Commissioning']=('','DT+L1MuBit+RPC+CSCHLT+CSCAlone+MuonTrack+LogError')
    alcaAndSkimMap['MinimumBias']=('SiStripCalZeroBias+SiStripCalMinBias+TkAlMinBias+HcalCalIsoTrk','CSC+MuonTrack+HSCP+BeamBkg+ValSkim+TPG+LogError')
    alcaAndSkimMap['EG']=('EcalCalElectron','LogError')
    alcaAndSkimMap['Mu']=('MuAlCalIsolatedMu+MuAlOverlaps+TkAlMuonIsolated+DtCalib+TkAlZMuMu','LogError')
    alcaAndSkimMap['JetMETTau']=('','LogError+DiJet+Tau')
    #alcaAndSkimMap['AlCaP0']=('EcalCalPi0Calib+EcalCalEtaCalib','')
    #alcaAndSkimMap['AlCaPhiSymEcal']=('EcalCalPhiSym+DQM','')
    alcaAndSkimMap['HcalNZS']=('HcalCalMinBias','')
    #alcaAndSkimMap['Cosmics']=('TkAlBeamHalo+MuAlBeamHaloOverlaps+MuAlBeamHalo+TkAlCosmics0T+MuAlStandAloneCosmics+MuAlGlobalCosmics+MuAlCalIsolatedMu+HcalCalHOCosmics','')
    alcaAndSkimMap['Cosmics']=('','CosmicSP+CSC+LogError')
    alcaAndSkimMap['EGMonitor']=('','EcalRH+TPG+LogError')
    return alcaAndSkimMap

def Era_2PDs():
    alcaAndSkimMap={}
    alcaAndSkimMap['']=('','LogError')
    alcaAndSkimMap['ZeroBias']=('SiStripCalZeroBias','')
    alcaAndSkimMap['MinimumBias']=('SiStripCalMinBias+SiStripCalZeroBias+TkAlMinBias+TkAlMuonIsolated+MuAlCalIsolatedMu+MuAlOverlaps+HcalCalIsoTrk','')
    alcaAndSkimMap['HcalNZS']=('HcalCalMinBias','')
    alcaAndSkimMap['Cosmics']=('','CosmicSP+CSC+LogError')
    return alcaAndSkimMap

def Era_1PDs():
    alcaAndSkimMap={}
    alcaAndSkimMap['MinimumBias']=('SiStripCalMinBias+SiStripCalZeroBias+TkAlMinBias+TkAlMuonIsolated+MuAlCalIsolatedMu+MuAlOverlaps+HcalCalIsoTrk','LogError')
    return alcaAndSkimMap

evt=options.output.split(',')
tiers=[]
for e in evt:
        tiers.append(e)

tiers=','.join(tiers)

com='cmsDriver.py reco -s RAW2DIGI,L1Reco,RECO,DQM%s  --data --magField AutoFromDBCurrent --scenario pp --datatier '+tiers+' --eventcontent '+options.output+' --customise Configuration/GlobalRuns/reco_TLR_'+options.release+'.py --cust_function customisePPData --no_exec --python_filename=rereco_%sCollision_'+options.release+'.py --conditions %s '+options.options
skim='cmsDriver.py skim -s SKIM:%s --data --magField AutoFromDBCurrent --scenario pp --datatier '+tiers+' --eventcontent '+options.output+' --no_exec --python_filename=skim_%s'+options.release+'.py --conditions %s '+options.optionsSkim
comCos='cmsDriver.py reco -s RAW2DIGI,L1Reco,RECO,DQM%s  --data --magField AutoFromDBCurrent --scenario cosmics --datatier '+tiers+' --eventcontent '+options.output+' --customise Configuration/GlobalRuns/reco_TLR_'+options.release+'.py --cust_function customiseCosmicData --no_exec --python_filename=rereco_%sCosmic_'+options.release+'.py --conditions %s '+options.options

import os

alcaAndSkimMap=globals()["Era_%ss"%(options.era)]()
print "list of PD to create reprocessing and skim configuration for:", alcaAndSkimMap.keys()
for PD in alcaAndSkimMap.keys():
    alcaArg=alcaAndSkimMap[PD][0]
    skimArg=alcaAndSkimMap[PD][1]
    print "PD:",PD,"alca=",alcaArg,"skim=",skimArg
    
    c=com
    if PD == 'HcalNZS':
        c=c.replace(',RECO,',',RECO:reconstruction_HcalNZS,')
        print c
        
    cc=comCos
    s=skim
    spec=options.era+'_'
    doCFG=True
    if 'AOD' in options.output:
        spec+='AOD_'
    if (PD==''):
        alca=''
        c+=' --process reRECO'
        cc+=' --process reRECO'
    else:
        spec+=PD+"_"
        if alcaArg=='':
            doCFG=False
            alca=''
        else:
            alca=',ALCA:%s'%(alcaArg,)

    if (options.options !=""):
        spec+="spec_"

    if doCFG:
        os.system(c%(alca,spec,options.GT))
        if PD=='':
            os.system(cc%(',ALCA:TkAlBeamHalo+TkAlCosmics0T+MuAlGlobalCosmics+MuAlCalIsolatedMu+HcalCalHOCosmics+DtCalib',spec,options.GT))

            
    if (skimArg!=""):
        os.system(s%(skimArg,spec,options.GT))

