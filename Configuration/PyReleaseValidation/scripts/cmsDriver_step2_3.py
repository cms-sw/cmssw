#!/usr/bin/env python

import sys
import os
import optparse

usage=\
"""%prog <job_type> [options].
<job type>: RELVAL, MinBias, JetETXX, JetET20, GammaJets, MuonPTXX, ZW, HCALNZS, HCALIST, TrackerHaloMuon, TrackerCosBON, TrackerCosBOFF, TrackerLaser, HaloMuon  MuonCosBON, MuonCosBOFF '
"""

parser = optparse.OptionParser(usage)

parser.add_option("--globaltag",
                   help="Name of global conditions to use",
                   default="STARTUP_V4",
                   dest="gt")

parser.add_option("--rereco",
                   help="if rereco only RECO+DQM",
                   default=False,
                   dest="rereco",
                   action="store_true")


(options,args) = parser.parse_args() # by default the arg is sys.argv[1:]


alcaDict2={'MinBias':'RpcCalHLT+DQM',
           'JetET20':'MuAlCalIsolatedMu+RpcCalHLT+DQM',
           'JetETXX':'MuAlCalIsolatedMu+RpcCalHLT+DQM',
           'GammaJets':'RpcCalHLT+DQM',
           'MuonPTXX':'MuAlCalIsolatedMu+RpcCalHLT+DQM',
           'ZW':'MuAlCalIsolatedMu+RpcCalHLT+DQM',
           'HCALNZS':'HcalCalMinBias+DQM',
           'HCALIST':'HcalCalIsoTrkNoHLT+DQM',
           'RELVAL':'MuAlCalIsolatedMu+RpcCalHLT+DQM',
           'TrackerHaloMuon':'TkAlBeamHalo',
           'TrackerCosBON':'TkAlCosmics',
           'TrackerCosBOFF':'TkAlCosmics',
           'MuonCosBON':'MuAlZeroFieldGlobalCosmics',
           'MuonCosBOFF':'MuAlZeroFieldGlobalCosmics',
           'TrackerLaser':'TkAlLAS',
           'HaloMuon':'MuAlBeamHalo+MuAlBeamHaloOverlaps'
           }

alcaDict3={'MinBias':'TkAlMuonIsolated+TkAlJpsiMuMu+TkAlMinBias+SiPixelLorentzAngle+SiStripCalMinBias+EcalCalPhiSym+EcalCalPi0Calib+HcalCalDijets+HcalCalGammaJet+HcalCalMinBias+HcalCalHO+MuAlOverlaps+DQM',
           'JetET20':'TkAlZMuMu+TkAlMuonIsolated+TkAlJpsiMuMu+TkAlUpsilonMuMu+TkAlMinBias+SiPixelLorentzAngle+SiStripCalMinBias+EcalCalElectron+EcalCalPhiSym+EcalCalPi0Calib+HcalCalDijets+HcalCalGammaJet+HcalCalHO+MuAlOverlaps+DQM',
           'JetETXX':'TkAlZMuMu+TkAlMuonIsolated+TkAlJpsiMuMu+TkAlUpsilonMuMu+SiPixelLorentzAngle+EcalCalElectron+EcalCalPhiSym+EcalCalPi0Calib+HcalCalDijets+HcalCalGammaJet+HcalCalHO+MuAlOverlaps+DQM',
           'GammaJets':'EcalCalElectron+EcalCalPhiSym+EcalCalPi0Calib+HcalCalDijets+HcalCalGammaJet+DQM',
           'MuonPTXX':'TkAlZMuMu+TkAlMuonIsolated+TkAlJpsiMuMu+TkAlUpsilonMuMu+SiPixelLorentzAngle+EcalCalPhiSym+EcalCalPi0Calib+HcalCalDijets+HcalCalGammaJet+HcalCalHO+MuAlOverlaps+DQM',
           'ZW':'TkAlZMuMu+TkAlMuonIsolated+SiPixelLorentzAngle+EcalCalElectron+HcalCalHO+MuAlOverlaps+DQM',
           'HCALNZS':'', # 'HcalCalMinBias+DQM',
           'HCALIST':'', #'HcalCalIsoTrkNoHLT+DQM',
           'RELVAL':'TkAlZMuMu+TkAlMuonIsolated+TkAlJpsiMuMu+TkAlUpsilonMuMu+TkAlMinBias+SiPixelLorentzAngle+SiStripCalMinBias+EcalCalElectron+EcalCalPhiSym+EcalCalPi0Calib+HcalCalDijets+HcalCalGammaJet+HcalCalIsoTrkNoHLT+MuAlOverlaps+DQM',
           'TrackerHaloMuon':'', #'TkAlBeamHalo',
           'TrackerCosBON':'', # 'TkAlCosmics',
           'TrackerCosBOFF':'', # 'TkAlCosmics',
           'MuonCosBON':'', # 'MuAlZeroFieldGlobalCosmics',
           'MuonCosBOFF':'', # 'MuAlZeroFieldGlobalCosmics',
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
                     'TrackerCosBON':'Configuration/Spring08Production/iCSA08_TkCosmicBON_RECO_cff.py',
                     'TrackerCosBOFF':'Configuration/Spring08Production/iCSA08_TkCosmicBOFF_RECO_cff.py',
                     'MuonCosBON':'Configuration/Spring08Production/iCSA08_MuonCosmicBON_RECO_cff.py',
                     'MuonCosBOFF':'Configuration/Spring08Production/iCSA08_MuonCosmicBOFF_RECO_cff.py',
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
                     'TrackerCosBON':'RECO:Alignment/CommonAlignmentProducer/data/Reconstruction_Cosmics.cff',
                     'TrackerCosBOFF':'RECO:Alignment/CommonAlignmentProducer/data/Reconstruction_Cosmics.cff',
                     'MuonCosBON':'RECO:Alignment/CommonAlignmentProducer/data/Reconstruction_Cosmics.cff',
                     'MuonCosBOFF':'RECO:Alignment/CommonAlignmentProducer/data/Reconstruction_Cosmics.cff',
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
                        'TrackerCosBON':':reconstruction_cosmics',
                        'TrackerCosBOFF':':reconstruction_cosmics',
                        'MuonCosBON':':reconstruction_cosmics',
                        'MuonCosBOFF':':reconstruction_cosmics',
                        'TrackerLaser':'',
                        'HaloMuon':''
                        }

# the cosmic samples all need special treatment
# - no post reco
# - another event content
cosmicSamples=['MuonCosBON', 'MuonCosBOFF','TrackerCosBOFF','TrackerCosBON']
cosmicContentFileName = "Alignment/CommonAlignmentProducer/data/EventContent_cosmics.cff"
cosmicContentName = "RECOSIMCosmics"

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
if typeOfEv == 'RELVAL':
    eventcontent='FEVTDEBUGHLT'
    
steps2='RAW2DIGI,RECO'+recoseqCustomiseDict[typeOfEv]

if typeOfEv in cosmicSamples:
    eventcontent = cosmicContentName+":"+cosmicContentFileName
else:
    steps2=steps2+',POSTRECO'


    

if ( not (alca2=='')):
    steps2=steps2+',ALCA:'+alca2

steps3='ALCA:'+alca3

extracom=''
if ( len(args)>1):
    for i in args[1:]:
        extracom=extracom+' '+i

if options.rereco:
    steps2 = "RAW2DIGI,RECO"+recoseqCustomiseDict[typeOfEv]+",POSTRECO,DQM"
    

command2=baseCommand+' step2_'+typeOfEv+' -s ' + steps2 + ' -n 1000 --filein file:raw.root --eventcontent ' + eventcontent + ' --conditions '+conditions+extracom+' --no_exec'
command3=baseCommand+' step3_'+typeOfEv+' -s ' + steps3 + ' -n 1000 --filein file:reco.root ' + ' --conditions '+conditions+extracom+' --no_exec'

if ( recoCustomise != '' ):
    command2 = command2+ " --customise "+recoCustomise

if ( cffCustomise != '' ):
    command2 = command2+ " --altcffs "+cffCustomise

    
if ( typeOfEv == 'RELVAL'):
    command2=command2+' --datatier GEN-SIM-DIGI-RAW-HLTDEBUG-RECO '+' --oneoutput'
    command3=command3+' --datatier GEN-SIM-DIGI-RAW-HLTDEBUG-RECO '+' --oneoutput --eventcontent FEVTSIM'
else:
    command3=command3+' --eventcontent none'

os.system(command2)
if ( not ( alca3=='') and not options.rereco):
    print command3
    os.system(command3)
