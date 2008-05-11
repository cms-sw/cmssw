#!/usr/bin/env python

import sys
import os
import optparse

usage=\
"""%prog <job_type> [options].
<job type>: RELVAL, MinBias, JetETXX, JetET20, GammaJets, MuonPTXX, ZW, HCALNZS, HCALIST, TrackerHaloMuon, TrackerCosBON, TrackerCosBOFF, TrackerLaser, HaloMuon  '
"""

parser = optparse.OptionParser(usage)

parser.add_option("--globaltag",
                   help="Name of global conditions to use",
                   default="STARTUP_V2",
                   dest="gt")

(options,args) = parser.parse_args() # by default the arg is sys.argv[1:]


alcaDict2={'MinBias':'SiPixelLorentzAngle+SiStripCalMinBias+RpcCalHLT+DQM',
           'JetET20':'SiPixelLorentzAngle+SiStripCalMinBias+MuAlZMuMu+RpcCalHLT+DQM',
           'JetETXX':'SiPixelLorentzAngle+MuAlZMuMu+RpcCalHLT+DQM',
           'GammaJets':'RpcCalHLT+DQM',
           'MuonPTXX':'SiPixelLorentzAngle+MuAlZMuMu+RpcCalHLT+DQM',
           'ZW':'SiPixelLorentzAngle+MuAlZMuMu+RpcCalHLT+DQM',
           'HCALNZS':'HcalCalMinBias+DQM',
           'HCALIST':'HcalCalIsoTrkNoHLT+DQM',
           'RELVAL':'SiPixelLorentzAngle+SiStripCalMinBias+MuAlZMuMu+RpcCalHLT+DQM',
           'TrackerHaloMuon':'TkAlBeamHalo',
           'TrackerCosBON':'TkAlCosmics',
           'TrackerLaser':'TkAlLAS',
           'TrackerCosBOFF':'TkAlCosmics',
           'HaloMuon':'MuAlBeamHalo+MuAlBeamHaloOverlaps'
           }

alcaDict3={'MinBias':'TkAlMuonIsolated+TkAlJpsiMuMu+TkAlMinBias+EcalCalPhiSym+EcalCalPi0Calib+HcalCalDijets+HcalCalGammaJet+HcalCalMinBias+HcalCalHO+MuAlOverlaps+DQM',
           'JetET20':'TkAlZMuMu+TkAlMuonIsolated+TkAlJpsiMuMu+TkAlUpsilonMuMu+TkAlMinBias+EcalCalElectron+EcalCalPhiSym+EcalCalPi0Calib+HcalCalDijets+HcalCalGammaJet+HcalCalHO+MuAlOverlaps+DQM',
           'JetETXX':'TkAlZMuMu+TkAlMuonIsolated+TkAlJpsiMuMu+TkAlUpsilonMuMu+EcalCalElectron+EcalCalPhiSym+EcalCalPi0Calib+HcalCalDijets+HcalCalGammaJet+HcalCalHO+MuAlOverlaps+DQM',
           'GammaJets':'EcalCalElectron+EcalCalPhiSym+EcalCalPi0Calib+HcalCalDijets+HcalCalGammaJet+DQM',
           'MuonPTXX':'TkAlZMuMu+TkAlMuonIsolated+TkAlJpsiMuMu+TkAlUpsilonMuMu+EcalCalPhiSym+EcalCalPi0Calib+HcalCalDijets+HcalCalGammaJet+HcalCalHO+MuAlOverlaps+DQM',
           'ZW':'TkAlZMuMu+TkAlMuonIsolated+EcalCalElectron+HcalCalHO+MuAlOverlaps+DQM',
           'HCALNZS':'', # 'HcalCalMinBias+DQM',
           'HCALIST':'', #'HcalCalIsoTrkNoHLT+DQM',
           'RELVAL':'TkAlZMuMu+TkAlMuonIsolated+TkAlJpsiMuMu+TkAlUpsilonMuMu+TkAlMinBias+EcalCalElectron+EcalCalPhiSym+EcalCalPi0Calib+HcalCalDijets+HcalCalGammaJet+HcalCalMinBias+HcalCalIsoTrkNoHLT+HcalCalHO+MuAlOverlaps+DQM',
           'TrackerHaloMuon':'', #'TkAlBeamHalo',
           'TrackerCosBON':'', # 'TkAlCosmics',
           'TrackerCosBOFF':'', # 'TkAlCosmics',
           'TrackerLaser':'', #'TkAlLAS',
           'HaloMuon':''#'MuAlBeamHalo+MuAlBeamHaloOverlaps'
           }

recoCustomiseDict = {
                     'MinBias':'Configuration/Spring08Production/iCSA08_MinBias_RECO_cff.py',
                     'JetET20':'',
                     'JetETXX':'',
                     'GammaJets':'',
                     'MuonPTXX':'',
                     'ZW':'',
                     'HCALNZS':'Configuration/Spring08Production/iCSA08_HCALNZS_RECO_cff.py',
                     'HCALIST':'',
                     'RELVAL':'',
                     'TrackerHaloMuon':'Configuration/Spring08Production/iCSA08_TkBeamHalo_RECO_cff.py',
                     'TrackerCosBON':'',
                     'TrackerCosBOFF':'Configuration/Spring08Production/iCSA08_TkCosmicBOFF_RECO_cff.py',
                     'TrackerLaser':'',
                     'HaloMuon':'Configuration/Spring08Production/iCSA08_MuonBeamHalo_RECO_cff.py'
                     }

cffCustomiseDict = {
                     'MinBias':'',
                     'JetET20':'',
                     'JetETXX':'',
                     'GammaJets':'',
                     'MuonPTXX':'',
                     'ZW':'',
                     'HCALNZS':'',
                     'HCALIST':'',
                     'RELVAL':'',
                     'TrackerHaloMuon':'',
                     'TrackerCosBON':'RECO:Configuration/GlobalRuns/data/ReconstructionGR.cff',
                     'TrackerCosBOFF':'RECO:Configuration/GlobalRuns/data/ReconstructionGR.cff',
                     'TrackerLaser':'',
                     'HaloMuon':''
                     }

recoseqCustomiseDict = {
                        'MinBias':'',
                        'JetET20':'',
                        'JetETXX':'',
                        'GammaJets':'',
                        'MuonPTXX':'',
                        'ZW':'',
                        'HCALNZS':'',
                        'HCALIST':'',
                        'RELVAL':'',
                        'TrackerHaloMuon':'',
                        'TrackerCosBON':':reconstructionGR',
                        'TrackerCosBOFF':':reconstructionGR',
                        'TrackerLaser':'',
                        'HaloMuon':''
                        }


typeOfEv=''
if ( len(args)>0):
    typeOfEv=args[0]
if not ( typeOfEv in alcaDict3 ):
    print usage
    sys.exit()

alca2=alcaDict2[typeOfEv]
alca3=alcaDict3[typeOfEv]

recoCustomise = recoCustomiseDict[typeOfEv]
cffCustomise = cffCustomiseDict[typeOfEv]

baseCommand='cmsDriver.py'
conditions='FrontierConditions_GlobalTag,'+options.gt+'::All'
eventcontent='RECOSIM'
steps2='RAW2DIGI,RECO'+recoseqCustomiseDict[typeOfEv]+',POSTRECO'
if ( not (alca2=='')):
    steps2=steps2+',ALCA:'+alca2

steps3='ALCA:'+alca3

extracom=''
if ( len(args)>1):
    for i in args[1:]:
        extracom=extracom+' '+i

command2=baseCommand+' step2_'+typeOfEv+' -s ' + steps2 + ' -n 1000 --filein file:raw.root --eventcontent ' + eventcontent + ' --conditions '+conditions+extracom+' --dump_cfg'
command3=baseCommand+' step3_'+typeOfEv+' -s ' + steps3 + ' -n 1000 --filein file:reco.root ' + ' --conditions '+conditions+extracom+' --dump_cfg'

if ( recoCustomise != '' ):
    command2 = command2+ " --customise "+recoCustomise

if ( cffCustomise != '' ):
    command2 = command2+ " --altcffs "+cffCustomise

    
if ( typeOfEv == 'RELVAL'):
    command2=command2+' --oneoutput'
    command3=command3+' --oneoutput --eventcontent FEVTSIM'
else:
    command3=command3+' --eventcontent none'

os.system(command2)
if ( not ( alca3=='')):
    print command3
    os.system(command3)
