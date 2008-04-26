#!/usr/bin/env python

import sys
import os
import optparse

usage=\
"""%prog <job_type> [options].
<job type>: RELVAL, MinBias, JetETXX, GammaJets, MuonPTXX, ZW, HCALNZS, HCALIST, TrackerHaloMuon, TrackerCosBON, TrackerCosBOFF, TrackerLaser, HaloMuon  '
"""

parser = optparse.OptionParser(usage)

parser.add_option("--globaltag",
                   help="Name of global conditions to use",
                   default="STARTUP_V2",
                   dest="gt")

(options,args) = parser.parse_args() # by default the arg is sys.argv[1:]


alcaDict2={'MinBias':'SiPixelLorentzAngle+SiStripCalMinBias+MuAlZMuMu+RpcCalHLT+DQM',
           'JetETXX':'SiPixelLorentzAngle+MuAlZMuMu+RpcCalHLT+DQM',
           'GammaJets':'MuAlZMuMu+RpcCalHLT+DQM',
           'MuonPTXX':'MuAlZMuMu+RpcCalHLT+DQM',
           'ZW':'SiPixelLorentzAngle+SiPixelLorentzAngle+MuAlZMuMu+RpcCalHLT+DQM',
           'HCALNZS':'',
           'HCALIST':'',
           'RELVAL':'SiPixelLorentzAngle+SiStripCalMinBias+MuAlZMuMu+RpcCalHLT+DQM',
           'TrackerHaloMuon':'',
           'TrackerCosBON':'',
           'TrackerLaser':'',
           'TrackerCosBOFF':'',
           'HaloMuon':''
           }

alcaDict3={'MinBias':'TkAlMuonIsolated+TkAlJpsiMuMu+TkAlMinBias+EcalCalPhiSym+EcalCalPi0Calib+HcalCalDijets+HcalCalGammaJet+HcalCalMinBias+HcalCalHO+MuAlOverlaps+DQM',
           'JetETXX':'TkAlZMuMu+TkAlMuonIsolated+TkAlJpsiMuMu+TkAlUpsilonMuMu+EcalCalElectron+EcalCalPi0Calib+HcalCalDijets+HcalCalGammaJet+HcalCalHO+MuAlOverlaps+DQM',
           'GammaJets':'EcalCalPi0Calib+HcalCalDijets+HcalCalGammaJet+DQM',
           'MuonPTXX':'TkAlZMuMu+TkAlMuonIsolated+TkAlJpsiMuMu+TkAlUpsilonMuMu+EcalCalPi0Calib+HcalCalDijets+HcalCalGammaJet+HcalCalHO+MuAlOverlaps+DQM',
           'ZW':'TkAlZMuMu+TkAlMuonIsolated+EcalCalElectron+HcalCalHO+MuAlOverlaps+DQM',
           'HCALNZS':'HcalCalMinBias+DQM',
           'HCALIST':'HcalCalIsoTrkNoHLT+DQM',
           'RELVAL':'TkAlZMuMu+TkAlMuonIsolated+TkAlJpsiMuMu+TkAlUpsilonMuMu+TkAlMinBias+EcalCalElectron+EcalCalPhiSym+EcalCalPi0Calib+HcalCalDijets+HcalCalGammaJet+HcalCalMinBias+HcalCalIsoTrkNoHLT+HcalCalHO+MuAlOverlaps+DQM',
           'TrackerHaloMuon':'TkAlBeamHalo',
           'TrackerCosBON':'TkAlCosmics',
           'TrackerCosBOFF':'TkAlCosmics',
           'TrackerLaser':'TkAlLAS',
           'HaloMuon':'MuAlBeamHalo+MulBeamHaloOverlaps'
           }

typeOfEv=''
if ( len(args)>0):
    typeOfEv=args[0]
if not ( typeOfEv in alcaDict3 ):
    print usage
    sys.exit()

alca2=alcaDict2[typeOfEv]
alca3=alcaDict3[typeOfEv]

baseCommand='cmsDriver.py'
conditions='FrontierConditions_GlobalTag,'+options.gt+'::All'
eventcontent='RECOSIM'
steps2='RAW2DIGI,RECO,POSTRECO'
if ( not (alca2=='')):
    steps2=steps2+',ALCA:'+alca2

steps3='ALCA:'+alca3

extracom=''
if ( len(args)>1):
    for i in args[1:]:
        extracom=extracom+' '+i

command2=baseCommand+' step2_'+typeOfEv+' -s ' + steps2 + ' -n 1000 --filein file:raw.root --eventcontent ' + eventcontent + ' --conditions '+conditions+extracom+' --dump_cfg'
command3=baseCommand+' step3_'+typeOfEv+' -s ' + steps3 + ' -n 1000 --filein file:reco.root ' + ' --conditions '+conditions+extracom+' --dump_cfg'

if ( typeOfEv == 'RELVAL'):
    command2=command2+' --oneoutput'
    command3=command3+' --oneoutput --eventcontent FEVTSIM'
else:
    command3=command3+' --eventcontent none'

os.system(command2)
if ( not ( alca3=='')):
    print command3
    os.system(command3)
