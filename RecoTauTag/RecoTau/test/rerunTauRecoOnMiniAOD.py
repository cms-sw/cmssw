import FWCore.ParameterSet.Config as cms
######
# Configuration to run tau ReReco+PAT at MiniAOD samples
# M. Bluj, NCBJ Warsaw
# based on work of J. Steggemann, CERN
# Created: 9 Nov. 2017
######

######
runType = 'signal'
# runType = 'background'
# runType = 'data'
maxEvents = 100
# maxEvents = -1


# If 'reclusterJets' set true a new collection of uncorrected ak4PFJets is
# built to seed taus (as at RECO), otherwise standard slimmedJets are used
reclusterJets = True
# reclusterJets = False

# set true for upgrade studies
phase2 = False
# phase2 = True
if phase2 and runType == 'data':
    print('There is not Phase2 data, yet! Setting phase2 to False')
    phase2 = False

# Output mode
outMode = 0  # store original MiniAOD and new selectedPatTaus
# outMode = 1 #store original MiniAOD, new selectedPatTaus, and all PFtau products as in AOD (except of unsuported ones)

print('Running Tau reco&id with MiniAOD inputs:')
print('\t Run type:', runType)
print('\t Recluster jets:', reclusterJets)
print('\t Use Phase2 settings:', phase2)
print('\t Output mode:', outMode)

#####
from Configuration.Eras.Era_Run2_2018_cff import Run2_2018
era = Run2_2018
if phase2:
    from Configuration.Eras.Era_Phase2_timing_cff import Phase2_timing
    era = Phase2_timing
process = cms.Process("TAURECO", era)
# for CH reco
process.load("Configuration.StandardSequences.MagneticField_cff")
if not phase2:
    process.load("Configuration.Geometry.GeometryRecoDB_cff")
else:
    process.load('Configuration.Geometry.GeometryExtended2023D17Reco_cff')

#####
readFiles = cms.untracked.vstring()
secFiles = cms.untracked.vstring()
process.source = cms.Source(
    "PoolSource", fileNames=readFiles, secondaryFileNames=secFiles)

process.maxEvents.input=maxEvents

print('\t Max events:', process.maxEvents.input.value())

if runType == 'signal':
    readFiles.extend([
        #'file:patMiniAOD_standard.root'
        '/store/relval/CMSSW_10_5_0_pre1/RelValZTT_13/MINIAODSIM/PU25ns_103X_upgrade2018_realistic_v8-v1/20000/EA29017F-9967-3F41-BB8A-22C44A454235.root'
    ])
elif runType == 'background':
    readFiles.extend([
        #'file:patMiniAOD_standard.root'
        '/store/relval/CMSSW_10_5_0_pre1/RelValQCD_FlatPt_15_3000HS_13/MINIAODSIM/PU25ns_103X_mcRun2_asymptotic_v3-v1/20000/A5CBC261-E3AB-C842-896F-E6AFB38DD22F.root'
    ])
elif runType == 'data':
    readFiles.extend([
        #'/store/data/Run2018D/SingleMuon/MINIAOD/12Nov2019_UL2018-v4/710000/B7163712-7B03-D949-91C9-EB5DD2E1D4C3.root' # SingleMuon PD
        '/store/data/Run2018D/Tau/MINIAOD/12Nov2019_UL2018-v1/00000/01415E2B-7CE5-B94C-93BD-0796FC40BD97.root' # Tau PD
    ])
else:
    print('Unknown runType =',runType,'; Use \"signal\" or \"background\" or \"data\"')
    exit(1)

#####
import RecoTauTag.Configuration.tools.adaptToRunAtMiniAOD as tauAtMiniTools

#####
tauAtMiniTools.addTauReReco(process)

#####
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
from Configuration.AlCa.GlobalTag import GlobalTag
if not phase2 and runType != 'data':
    process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:phase1_2018_realistic', '')
elif phase2:
    process.GlobalTag = GlobalTag(
        process.GlobalTag, 'auto:phase2_realistic', '')
else: # data
    process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run2_data', '')

#####
# mode = 0: store original MiniAOD and new selectedPatTaus
# mode = 1: store original MiniAOD, new selectedPatTaus, and all PFtau products as in AOD (except of unsuported ones)
process.output = tauAtMiniTools.setOutputModule(mode=outMode)
if runType == 'signal':
    process.output.fileName = 'miniAOD_TauReco_ggH.root'
    if reclusterJets:
        process.output.fileName = 'miniAOD_TauReco_ak4PFJets_ggH.root'
elif runType == 'background':
    process.output.fileName = 'miniAOD_TauReco_QCD.root'
    if reclusterJets:
        process.output.fileName = 'miniAOD_TauReco_ak4PFJets_QCD.root'
else: # data
    process.output.fileName = 'miniAOD_TauReco_data.root'
    if reclusterJets:
        process.output.fileName = 'miniAOD_TauReco_ak4PFJets_data.root'
process.out = cms.EndPath(process.output)

#####
tauAtMiniTools.adaptTauToMiniAODReReco(process, reclusterJets)

if runType == 'data':
    from PhysicsTools.PatAlgos.tools.coreTools import runOnData
    runOnData(process, names = ['Taus'], outputModules = [])

#####
process.load('FWCore.MessageService.MessageLogger_cfi')
if process.maxEvents.input.value() > 10:
    process.MessageLogger.cerr.FwkReport.reportEvery = process.maxEvents.input.value()//10
if process.maxEvents.input.value() > 10000 or process.maxEvents.input.value() < 0:
    process.MessageLogger.cerr.FwkReport.reportEvery = 1000

#####
process.options = dict( numberOfThreads = 4,
                      # numberOfThreads = 1,
                        numberOfStreams = 0,
                        wantSummary = True
)
print('\t No. of threads:', process.options.numberOfThreads.value(), ', no. of streams:', process.options.numberOfStreams.value())


