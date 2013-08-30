import FWCore.ParameterSet.Config as cms

import os

process = cms.PSet()

process.fwliteInput = cms.PSet(
    fileNames = cms.vstring(),
    
    ##maxEvents = cms.int32(100000),
    maxEvents = cms.int32(-1),
    
    outputEvery = cms.uint32(100000)
)

#----------------------------------------------------------------------------------------------------
inputFilePath  = "/data2/veelken/CMSSW_5_3_x/Ntuples/tauIdMVATraining/v1_2/"
inputFilePath += "user/veelken/CMSSW_5_3_x/Ntuples/tauIdMVATraining/v1_2/"

signalSamples = [
    "ZplusJets_madgraph"
]
smHiggsMassPoints = [ 80, 90, 100, 110, 120, 130, 140 ]
for massPoint in smHiggsMassPoints:
    ggSampleName = "ggHiggs%1.0ftoTauTau" % massPoint
    signalSamples.append(ggSampleName)
    vbfSampleName = "vbfHiggs%1.0ftoTauTau" % massPoint
    signalSamples.append(vbfSampleName)
mssmHiggsMassPoints = [ 160, 180, 200, 250, 300, 350, 400, 450, 500, 600, 700, 800, 900, 1000 ]
for massPoint in mssmHiggsMassPoints:
    ggSampleName = "ggA%1.0ftoTauTau" % massPoint
    signalSamples.append(ggSampleName)
    bbSampleName = "bbA%1.0ftoTauTau" % massPoint
    signalSamples.append(bbSampleName)
ZprimeMassPoints = [ 750, 1000, 1250, 1500, 1750, 2000, 2250, 2500 ]
for massPoint in ZprimeMassPoints:
    sampleName = "Zprime%1.0ftoTauTau" % massPoint
    signalSamples.append(sampleName)
WprimeMassPoints = [ 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000, 2100, 2200, 2300, 2400, 2500, 2600, 2700, 2800, 2900, 3000, 3200, 3500, 4000 ]
for massPoint in WprimeMassPoints:
    sampleName = "Wprime%1.0ftoTauTau" % massPoint
    signalSamples.append(sampleName)

backgroundSamples = [
    "PPmuXptGt20Mu15",
    "QCDmuEnrichedPt50to80",
    "QCDmuEnrichedPt80to120",
    "QCDmuEnrichedPt120to170",
    "QCDmuEnrichedPt170to300",
    "QCDmuEnrichedPt300to470",
    "QCDmuEnrichedPt470to600",
    "QCDmuEnrichedPt600to800",
    "QCDmuEnrichedPt800to1000",
    "QCDmuEnrichedPtGt1000",
    "WplusJets_madgraph",    
    "QCDjetsFlatPt15to3000",
    "QCDjetsPt50to80",
    "QCDjetsPt80to120",
    "QCDjetsPt120to170",
    "QCDjetsPt170to300",
    "QCDjetsPt300to470",
    "QCDjetsPt470to600",
    "QCDjetsPt600to800",
    "QCDjetsPt800to1000",
    "QCDjetsPt1000to1400",
    "QCDjetsPt1400to1800",
    "QCDmuEnrichedPtGt1800"
]

allSamples = []
allSamples.extend(signalSamples)
allSamples.extend(backgroundSamples)

inputFileNames = []
for sample in allSamples:
    try:
        inputFileNames.extend([ os.path.join(inputFilePath, sample, file) for file in os.listdir(os.path.join(inputFilePath, sample)) ])
    except OSError:
        print "inputFilePath = %s does not exist --> skipping !!" % os.path.join(inputFilePath, sample)
        continue    
print "inputFileNames = %s" % inputFileNames
process.fwliteInput.fileNames = cms.vstring(inputFileNames)
#----------------------------------------------------------------------------------------------------

process.preselectTreeTauIdMVA = cms.PSet(

    inputTreeName = cms.string('tauIdMVATrainingNtupleProducer/tauIdMVATrainingNtuple'),
    outputTreeName = cms.string('preselectedTauIdMVATrainingNtuple'),

    preselection = cms.string('recTauDecayMode == 0 || recTauDecayMode == 1 || recTauDecayMode == 2 || recTauDecayMode == 10'),

    samples = cms.vstring(signalSamples),
    ##samples = cms.vstring(backgroundSamples),

    branchNamePt = cms.string('recTauPt'),
    branchNameEta = cms.string('recTauEta'),
    
    branchNameEvtWeight = cms.string('evtWeight'),

    applyEventPruning = cms.int32(1),

    keepAllBranches = cms.bool(False),
    checkBranchesForNaNs = cms.bool(True),

    inputVariables = cms.vstring(
        ##'TMath::Log(TMath::Max(1., recTauPt))/F',
        'TMath::Abs(recTauEta)/F',
        'TMath::Log(TMath::Max(1.e-2, tauIsoDeltaR05PtThresholdsLoose3HitsChargedIsoPtSum))/F',
        'TMath::Log(TMath::Max(1.e-2, tauIsoDeltaR05PtThresholdsLoose3HitsNeutralIsoPtSum))/F',
        'TMath::Log(TMath::Max(1.e-2, tauIsoDeltaR08PtThresholdsLoose3HitsPUcorrPtSum))/F',
        'recTauDecayMode/I'
    ),
    spectatorVariables = cms.vstring(
        'recTauPt/F',
        ##'recTauDecayMode/I',
        'leadPFChargedHadrCandPt/F',
        'numOfflinePrimaryVertices/I'
    ),

    outputFileName = cms.string('preselectTreeTauIdMVA.root')
)
