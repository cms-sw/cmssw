import FWCore.ParameterSet.Config as cms
######
# Configuration to run tau ReReco+PAT at MiniAOD samples
# M. Bluj, NCBJ Warsaw
# based on work of J. Steggemann, CERN
# Created: 9 Nov. 2017
######

######
runSignal = True
# runSignal=False
maxEvents = 100
# maxEvents=-1


# If 'reclusterJets' set true a new collection of uncorrected ak4PFJets is
# built to seed taus (as at RECO), otherwise standard slimmedJets are used
reclusterJets = True
# reclusterJets = False

# set true for upgrade studies
phase2 = False
# phase2 = True

# Output mode
outMode = 0  # store original MiniAOD and new selectedPatTaus
# outMode = 1 #store original MiniAOD, new selectedPatTaus, and all PFtau products as in AOD (except of unsuported ones)

print 'Running Tau reco&id with MiniAOD inputs:'
print '\t Run on signal:', runSignal
print '\t Recluster jets:', reclusterJets
print '\t Use Phase2 settings:', phase2
print '\t Output mode:', outMode

#####
from Configuration.StandardSequences.Eras import eras
era = eras.Run2_2017
if phase2:
    era = eras.Phase2_timing
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

process.maxEvents = cms.untracked.PSet(
    input=cms.untracked.int32(maxEvents)
)
print '\t Max events:', process.maxEvents.input.value()

if runSignal:
    readFiles.extend([
        #'file:patMiniAOD_standard.root'
        '/store/relval/CMSSW_10_1_0_pre3/RelValZTT_13UP18/MINIAODSIM/PUpmx25ns_101X_upgrade2018_realistic_v3_cc7-v1/10000/2808A251-DE31-E811-BFBC-0242AC130002.root'
        #'root://cms-xrd-global.cern.ch///store/relval/CMSSW_10_0_0_pre2/RelValZTT_13/MINIAODSIM/PUpmx25ns_100X_mc2017_realistic_v1-v1/20000/B01F0774-17E1-E711-9826-0CC47A4D7654.root',
    ])
else:
    readFiles.extend([
        #'file:patMiniAOD_standard.root'
        '/store/relval/CMSSW_10_1_0_pre3/RelValZTT_13UP18/MINIAODSIM/PUpmx25ns_101X_upgrade2018_realistic_v3_cc7-v1/10000/2808A251-DE31-E811-BFBC-0242AC130002.root'
        #'root://cms-xrd-global.cern.ch///store/relval/CMSSW_10_0_0_pre2/RelValQCD_FlatPt_15_3000HS_13/MINIAODSIM/PUpmx25ns_100X_mcRun2_asymptotic_v2_FastSim-v1/20000/78318DC3-40E0-E711-BCFE-0CC47A4D763C.root',
        #'root://cms-xrd-global.cern.ch///store/relval/CMSSW_10_0_0_pre2/RelValQCD_FlatPt_15_3000HS_13/MINIAODSIM/PUpmx25ns_100X_mcRun2_asymptotic_v2_FastSim-v1/20000/E6F528C8-40E0-E711-9F06-0CC47A4C8E56.root',
    ])

#####
import RecoTauTag.Configuration.tools.adaptToRunAtMiniAOD as tauAtMiniTools

#####
tauAtMiniTools.addTauReReco(process)

#####
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
from Configuration.AlCa.GlobalTag import GlobalTag
if not phase2:
    process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run2_mc', '')
    process.GlobalTag.globaltag = '94X_mc2017_realistic_v1'
else:
    process.GlobalTag = GlobalTag(
        process.GlobalTag, 'auto:phase2_realistic', '')

#####
# mode = 0: store original MiniAOD and new selectedPatTaus
# mode = 1: store original MiniAOD, new selectedPatTaus, and all PFtau products as in AOD (except of unsuported ones)
process.output = tauAtMiniTools.setOutputModule(mode=outMode)
if runSignal:
    process.output.fileName = 'miniAOD_TauReco_ggH.root'
    if reclusterJets:
        process.output.fileName = 'miniAOD_TauReco_ak4PFJets_ggH.root'
else:
    process.output.fileName = 'miniAOD_TauReco_QCD.root'
    if reclusterJets:
        process.output.fileName = 'miniAOD_TauReco_ak4PFJets_QCD.root'
process.out = cms.EndPath(process.output)

#####
tauAtMiniTools.adaptTauToMiniAODReReco(process, reclusterJets)

#####
process.load('FWCore.MessageService.MessageLogger_cfi')
if process.maxEvents.input.value() > 10:
    process.MessageLogger.cerr.FwkReport.reportEvery = process.maxEvents.input.value()//10
if process.maxEvents.input.value() > 10000 or process.maxEvents.input.value() < 0:
    process.MessageLogger.cerr.FwkReport.reportEvery = 1000

#####
process.options = cms.untracked.PSet(
)
process.options.numberOfThreads = cms.untracked.uint32(4)
# process.options.numberOfThreads=cms.untracked.uint32(1)
process.options.numberOfStreams = cms.untracked.uint32(0)
print '\t No. of threads:', process.options.numberOfThreads.value(), ', no. of streams:', process.options.numberOfStreams.value()

process.options = cms.untracked.PSet(
    process.options,
    wantSummary=cms.untracked.bool(True)
)
