#!/usr/bin/env python

import sys
import os


alcaDict2={'MinBias':'SiPixelLorentzAngle+SiStripCalMinBias+MuAlZMuMu+RpcCalHLT+DQM',
           'JetETXX':'SiPixelLorentzAngle+MuAlZMuMu+RpcCalHLT+DQM',
           'GammaJets':'MuAlZMuMu+RpcCalHLT+DQM',
           'MuonPTXX':'MuAlZMuMu+RpcCalHLT+DQM',
           'ZW':'SiPixelLorentzAngle+SiPixelLorentzAngle+MuAlZMuMu+RpcCalHLT+DQM',
           'HCALNZS':'',
           'HCALIST':'',
           'RELVAL':'SiPixelLorentzAngle+SiStripCalMinBias+MuAlZMuMu+RpcCalHLT+DQM'
           }

alcaDict3={'MinBias':'TkAlMuonIsolated+TkAlJpsiMuMu+TkAlMinBias+EcalCalPhiSym+EcalCalPi0Calib+HcalCalDijets+HcalCalGammaJet+HcalCalMinBias+HcalCalHO+MuAlOverlaps+DQM',
           'JetETXX':'TkAlZMuMu+TkAlMuonIsolated+TkAlJpsiMuMu+TkAlUpsilonMuMu+EcalCalElectron+EcalCalPi0Calib+HcalCalDijets+HcalCalGammaJet+HcalCalHO+MuAlOverlaps+DQM',
           'GammaJets':'EcalCalPi0Calib+HcalCalDijets+HcalCalGammaJet+DQM',
           'MuonPTXX':'TkAlZMuMu+TkAlMuonIsolated+TkAlJpsiMuMu+TkAlUpsilonMuMu+EcalCalPi0Calib+HcalCalDijets+HcalCalGammaJet+HcalCalHO+MuAlOverlaps+DQM',
           'ZW':'TkAlZMuMu+TkAlMuonIsolated+EcalCalElectron+HcalCalHO+MuAlOverlaps+DQM',
           'HCALNZS':'HcalCalMinBias+DQM',
           'HCALIST':'HcalCalIsoTrkNoHLT+DQM',
           'RELVAL':'TkAlZMuMu+TkAlMuonIsolated+TkAlJpsiMuMu+TkAlUpsilonMuMu+TkAlMinBias+EcalCalElectron+EcalCalPhiSym+EcalCalPi0Calib+HcalCalDijets+HcalCalGammaJet+HcalCalMinBias+HcalCalIsoTrkNoHLT+HcalCalHO+MuAlOverlaps+DQM'
           }

typeOfEv=''
if ( len(sys.argv)>1):
    typeOfEv=sys.argv[1]
if not ( typeOfEv in alcaDict3 ):
    print 'Usage; cmsDriver_step2_3.py <job type>'
    print '  <job type>: RELVAL, MinBias, JetETXX, GammaJets, MuonPTXX, ZW, HCALNZS, HCLIST'
    sys.exit()

alca2=alcaDict2[typeOfEv]
alca3=alcaDict3[typeOfEv]

baseCommand='cmsDriver.py'
conditions='FrontierConditions_GlobalTag,STARTUP_V1::All'
eventcontent='RECOSIM'
steps2='RAW2DIGI,RECO,POSTRECO'
if ( not (alca2=='')):
    steps2=steps2+',ALCA:'+alca2

steps3='ALCA:'+alca3

extracom=''
if ( len(sys.argv)>2):
    for i in sys.argv[2:]:
        extracom=extracom+' '+i

command2=baseCommand+' step2_'+typeOfEv+' -s ' + steps2 + ' -n 1000 --filein file:raw.root --eventcontent ' + eventcontent + ' --conditions '+conditions+extracom+' --dump_cfg'
command3=baseCommand+' step3_'+typeOfEv+' -s ' + steps3 + ' -n 1000 --filein file:reco.root ' + ' --conditions '+conditions+extracom+' --dump_cfg'

if ( typeOfEv == 'RELVAL'):
    command2=command2+' --oneoutput'
    command3=command3+' --oneoutput'
else:
    command3=command3+' --fileout none'

#os.system(command2)
if ( not ( alca3=='')):
    print command3
    os.system(command3)
