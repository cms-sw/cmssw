import FWCore.ParameterSet.Config as cms

from CMGTools.Production.datasetToSource import datasetToSource
from CMGTools.H2TauTau.tools.setupJSON import setupJSON
# from CMGTools.H2TauTau.tools.setupRecoilCorrection import setupRecoilCorrection
from CMGTools.H2TauTau.tools.setupEmbedding import setupEmbedding
from CMGTools.H2TauTau.objects.jetreco_cff import addAK4Jets
from CMGTools.H2TauTau.tools.setupOutput import addTauMuOutput, addTauEleOutput, addDiTauOutput, addMuEleOutput, addDiMuOutput

sep_line = '-'*70

process = cms.Process("H2TAUTAU")

process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(-1))

numberOfFilesToProcess = -1
debugEventContent = False

# choose from 'tau-mu' 'di-tau' 'tau-ele' 'mu-ele' 'all-separate', 'all'
# channel = 'all'
channel = 'di-mu'

# newSVFit enables the svfit mass reconstruction used for the H->tau tau analysis.
# if false, much faster processing but mass is wrong. 
newSVFit = False
tauScaling = 0 # JAN: to be implemented downstream

# increase to 1000 before running on the batch, to reduce size of log files
# on your account
reportInterval = 100

print sep_line
print 'channel', channel
print 'newSVFit', newSVFit
print 'tau scaling =', tauScaling

# Input & JSON             -------------------------------------------------

# dataset_user = 'htautau_group' 
# dataset_name = '/VBF_HToTauTau_M-125_13TeV-powheg-pythia6/Spring14dr-PU20bx25_POSTLS170_V5-v1/AODSIM/SS14/'
# dataset_files = 'miniAOD-prod_PAT_.*root'

dataset_user = 'CMS'
# dataset_name = '/TTbarH_M-125_13TeV_amcatnlo-pythia8-tauola/Phys14DR-PU40bx25_PHYS14_25_V1-v1/MINIAODSIM'
# dataset_name = '/GluGluToHToTauTau_M-125_13TeV-powheg-pythia6/Phys14DR-PU20bx25_tsg_PHYS14_25_V1-v1/MINIAODSIM'
dataset_name = '/DYJetsToLL_M-50_13TeV-madgraph-pythia8/Phys14DR-PU20bx25_PHYS14_25_V1-v1/MINIAODSIM'
#dataset_name = '/GluGluToHToTauTau_M-125_13TeV-powheg-pythia6/Phys14DR-PU40bx25_PHYS14_25_V1-v2/MINIAODSIM'
# dataset_name = '/VBF_HToTauTau_M-125_13TeV-powheg-pythia6/Phys14DR-PU20bx25_tsg_PHYS14_25_V1-v2/MINIAODSIM'
# dataset_name = '/VBF_HToTauTau_M-125_13TeV-powheg-pythia6/Phys14DR-PU40bx25_PHYS14_25_V1-v1/MINIAODSIM'
# dataset_name = '/TTbarH_M-125_13TeV_amcatnlo-pythia8-tauola/Phys14DR-PU20bx25_tsg_PHYS14_25_V1-v2/MINIAODSIM' ## also in Marias request
# dataset_name = '/TTbarH_M-125_13TeV_amcatnlo-pythia8-tauola/Phys14DR-PU40bx25_PHYS14_25_V1-v1/MINIAODSIM'

dataset_files = '.*root'


process.source = datasetToSource(
    dataset_user,
    dataset_name,
    dataset_files,
    )

process.source.inputCommands=cms.untracked.vstring(
    'keep *'
    )

process.options = cms.untracked.PSet(
        allowUnscheduled = cms.untracked.bool(True)
)

if numberOfFilesToProcess>0:
    process.source.fileNames = process.source.fileNames[:numberOfFilesToProcess]

runOnMC = process.source.fileNames[0].find('Run201')==-1 and process.source.fileNames[0].find('embedded')==-1

if runOnMC == False:
    json = setupJSON(process)


# load the channel paths -------------------------------------------
process.load('CMGTools.H2TauTau.h2TauTau_cff')

# JAN: recoil correction disabled for now; reactivate if necessary
# setting up the recoil correction according to the input file
# recoilEnabled = False
# setupRecoilCorrection( process, runOnMC,
#                        enable=recoilEnabled, is53X=isNewerThan('CMSSW_5_2_X'))



isEmbedded = setupEmbedding(process, channel)
addAK4 = True

# Adding jet collection
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
process.GlobalTag.globaltag = 'PHYS14_25_V2::All'
# process.GlobalTag.globaltag = 'auto:run2_mc'


process.load('JetMETCorrections.Configuration.DefaultJEC_cff')
process.load('PhysicsTools.PatAlgos.slimming.unpackedTracksAndVertices_cfi')
process.load("Configuration.StandardSequences.Geometry_cff")
process.load("Configuration.StandardSequences.MagneticField_38T_cff")
process.load('TrackingTools.TransientTrack.TransientTrackBuilder_cfi')
process.load('RecoBTag.Configuration.RecoBTag_cff')

if addAK4:
    addAK4Jets(process)
    process.mvaMetInputPath.insert(0, process.jetSequenceAK4)

if '25' in dataset_name:
    print 'Using 25 ns MVA MET training'
    process.mvaMETTauMu.inputFileNames = cms.PSet(
        U     = cms.FileInPath('RecoMET/METPUSubtraction/data/RecoilCor_13TeV.root'),
        DPhi  = cms.FileInPath('RecoMET/METPUSubtraction/data/PhiCor_13TeV.root'),
        CovU1 = cms.FileInPath('RecoMET/METPUSubtraction/data/CovU1_13TeV.root'),
        CovU2 = cms.FileInPath('RecoMET/METPUSubtraction/data/CovU2_13TeV.root')
    )
    # process.mvaMETTauMu.inputRecords = cms.PSet(
    #     U = cms.string("U1Correction"),
    #     DPhi = cms.string("PhiCorrection"),
    #     CovU1 = cms.string("CovU1"),
    #     CovU2 = cms.string("CovU2")
    # )
    # process.mvaMETTauMu.inputFileNames = cms.PSet(
    #     U     = cms.FileInPath('RecoMET/METPUSubtraction/data/gbrmet_53_Sep2013_type1.root'),
    #     DPhi  = cms.FileInPath('RecoMET/METPUSubtraction/data/gbrmetphi_53_June2013_type1.root'),
    #     CovU1 = cms.FileInPath('RecoMET/METPUSubtraction/data/gbru1cov_53_Dec2012.root'),
    #     CovU2 = cms.FileInPath('RecoMET/METPUSubtraction/data/gbru2cov_53_Dec2012.root')
    # )



# OUTPUT definition ----------------------------------------------------------
process.outpath = cms.EndPath()


# JAN: In 2015, we should finally make sure that we apply the correction to all
# generator-matched taus, regardless of the process

# 2012: don't apply Tau ES corrections for data (but do for embedded) or 
# processes not containing real taus

# signalTauProcess = (process.source.fileNames[0].find('HToTauTau') != -1) or (process.source.fileNames[0].find('DY') != -1) or isEmbedded

if channel=='all' or channel=='all-separate':
    process.schedule = cms.Schedule(
        process.mvaMetInputPath,
        process.tauMuPath,
        process.tauElePath,
        process.muElePath,    
        process.diTauPath,
        process.outpath
        )
elif channel=='tau-mu':
    process.schedule = cms.Schedule(
        process.mvaMetInputPath,
        process.tauMuPath,
        process.outpath
        )
elif channel=='tau-ele':
    process.schedule = cms.Schedule(
        process.mvaMetInputPath,
        process.tauElePath,
        process.outpath
        )
elif channel=='di-tau':
    process.schedule = cms.Schedule(
        process.mvaMetInputPath,
        process.diTauPath,
        process.outpath
        )
elif channel=='mu-ele':
    process.schedule = cms.Schedule(
        process.mvaMetInputPath,
        process.muElePath,
        process.outpath
        )
elif channel=='di-mu':
    process.schedule = cms.Schedule(
        process.mvaMetInputPath,
        process.diMuPath,
        process.outpath
        )
else:
    raise ValueError('unrecognized channel')    

### Enable printouts like this:
# process.cmgTauMuCorSVFitPreSel.verbose = True

if channel=='tau-mu' or 'all' in channel:
    addTauMuOutput(process, debugEventContent, addPreSel=False, oneFile=(channel=='all'))
if channel=='tau-ele' or 'all' in channel:
    addTauEleOutput(process, debugEventContent, addPreSel=False, oneFile=(channel=='all'))
if channel=='mu-ele' or 'all' in channel:
    addMuEleOutput(process, debugEventContent, addPreSel=False, oneFile=(channel=='all'))
if channel=='di-mu' or 'all' in channel:
    addDiMuOutput(process, debugEventContent, addPreSel=False, oneFile=(channel=='all'))
if channel=='di-tau' or 'all' in channel:
    addDiTauOutput(process, debugEventContent, addPreSel=False, oneFile=(channel=='all'))

# Message logger setup.
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = reportInterval
process.options = cms.untracked.PSet(wantSummary = cms.untracked.bool(False))

if newSVFit:
    process.cmgTauMuCorSVFitPreSel.SVFitVersion = 2
    process.cmgTauEleCorSVFitPreSel.SVFitVersion = 2
    process.cmgDiTauCorSVFitPreSel.SVFitVersion = 2
    process.cmgMuEleCorSVFitPreSel.SVFitVersion = 2
else:
    process.cmgTauMuCorSVFitPreSel.SVFitVersion = 1
    process.cmgTauEleCorSVFitPreSel.SVFitVersion = 1
    process.cmgDiTauCorSVFitPreSel.SVFitVersion = 1
    process.cmgMuEleCorSVFitPreSel.SVFitVersion = 1

print sep_line
print 'INPUT:'
print sep_line
print process.source.fileNames
print
if not runOnMC:
    print 'json:', json
print
print sep_line
print 'PROCESSING'
print sep_line
print 'runOnMC:', runOnMC
print 
