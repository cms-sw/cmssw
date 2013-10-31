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
inputFilePath  = "/data2/veelken/CMSSW_5_3_x/Ntuples/antiMuonDiscrMVATraining/antiMuonDiscr_v1_0/"
inputFilePath += "user/veelken/CMSSW_5_3_x/Ntuples/antiMuonDiscrMVATraining/antiMuonDiscr_v1_0/"

signalSamples = [
    "ZplusJets_madgraph_signal",
    "WplusJets_madgraph_signal",
    "TTplusJets_madgraph_signal"
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
    ##sampleName = "Wprime%1.0ftoTauNu" % massPoint  
    sampleName = "Wprime%1.0ftoTauTau" % massPoint    
    signalSamples.append(sampleName)

backgroundSamples = [
    "ZplusJets_madgraph_background",
    "WplusJets_madgraph_background",
    "TTplusJets_madgraph_background"
]
ZprimeMassPoints = [ 750, 1000, 1250, 1500, 1750, 2000, 2250, 2500, 2750, 3000 ]
for massPoint in ZprimeMassPoints:
    sampleName = "Zprime%1.0ftoMuMu" % massPoint
    backgroundSamples.append(sampleName)
WprimeMassPoints = [ 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000, 2100, 2200, 2300, 2400, 2500, 2600, 2700, 2800, 2900, 3000, 3200, 3500, 4000 ]
for massPoint in WprimeMassPoints:
    sampleName = "Wprime%1.0ftoMuNu" % massPoint
    backgroundSamples.append(sampleName)
mssmHiggsMassPoints = [ 100, 110, 120, 130, 140 ]
for massPoint in mssmHiggsMassPoints:
    ggSampleName = "ggA%1.0ftoMuMu" % massPoint
    backgroundSamples.append(ggSampleName)
    bbSampleName = "bbA%1.0ftoMuMu" % massPoint
    backgroundSamples.append(bbSampleName)
DrellYanMassPoints = [ 120, 200, 400, 500, 700, 800, 1000, 1500, 2000 ]
for massPoint in DrellYanMassPoints:
    sampleName = "DY%1.0ftoMuNu" % massPoint
    backgroundSamples.append(sampleName)

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

    inputTreeName = cms.string('antiMuonDiscrMVATrainingNtupleProducer/antiMuonDiscrMVATrainingNtuple'),
    outputTreeName = cms.string('preselectedAntiMuonDiscrMVATrainingNtuple'),

    preselection = cms.string('recTauDecayMode == 0 || recTauDecayMode == 1 || recTauDecayMode == 2 || recTauDecayMode == 10'),

    samples = cms.vstring(signalSamples),
    ##samples = cms.vstring(backgroundSamples),

    branchNamePt = cms.string('recTauPt'),
    branchNameEta = cms.string('recTauEta'),
    branchNameNumMatches = cms.string('numMatches'),
    
    branchNameEvtWeight = cms.string('evtWeight'),

    applyEventPruning = cms.int32(0),

    keepAllBranches = cms.bool(False),
    checkBranchesForNaNs = cms.bool(True),

    inputVariables = cms.vstring(
        'TMath::Abs(TMath::ACos(TMath::Cos(12.*recTauPhi))/12.)/F',
        'TMath::Abs(recTauEta)/F',
        'TMath::Sqrt(TMath::Max(0., leadPFChargedHadrCandCaloEnECAL))/F',
        'TMath::Sqrt(TMath::Max(0., leadPFChargedHadrCandCaloEnHCAL))/F',
        'numMatches/F',
        'numHitsDT1 + numHitsCSC1 + numHitsRPC1/F',
        'numHitsDT2 + numHitsCSC2 + numHitsRPC2/F',
        'numHitsDT3 + numHitsCSC3 + numHitsRPC3/F',
        'numHitsDT4 + numHitsCSC4 + numHitsRPC4/F'
    ),
    spectatorVariables = cms.vstring(
        'recTauPt/F',
        'numOfflinePrimaryVertices/I'
    ),

    outputFileName = cms.string('preselectTreeAntiMuonDiscrMVA.root')
)
