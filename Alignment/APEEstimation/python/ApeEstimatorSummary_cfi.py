import os

import FWCore.ParameterSet.Config as cms


ApeEstimatorSummary = cms.EDAnalyzer('ApeEstimatorSummary',
    
    # Set baseline or calculate APE
    setBaseline = cms.bool(False),
    
    # Weights for bins in sigmaX used for APE calculation from the values per bin,
    # make sure to use the same configuration for baseline and iterations!!!
    # Currently implemented: "unity", "entries", "entriesOverSigmaX2" 
    apeWeight = cms.string("entries"),
    
    # Define minimum number of hits per interval per sector for use of interval in APE calculation
    minHitsPerInterval = cms.double(100.),
    
    # Sigma factor for second gauss fit (+-2.5 sigma1 around mean1 of first fit)
    sigmaFactorFit = cms.double(2.5),
    
    # Multiplicative APE correction scaling factor (to prevent overestimation, since estimation is iterative process)
    # Also in smmothing mode used for the first iteration
    correctionScaling = cms.double(1.),
    
    # Use smoothing for iterations instead of scaling factor,
    # specify fraction [0,1] of recently calculated value (the rest is taken from previous one)
    smoothIteration = cms.bool(False),
    smoothFraction = cms.double(0.5),
    
    # File name for input file containing normalized residual distributions per sector per error bin (specified in first step's TFileService)
    InputFile = cms.string(os.environ['CMSSW_BASE'] + '/src/Alignment/APEEstimation/hists/inputFile.root'),
    
    # File name for result histograms of: estimated width of normalized residuals, calculated APE value
    ResultsFile = cms.string(os.environ['CMSSW_BASE'] + '/src/Alignment/APEEstimation/hists/resultsFile.root'),
    
    #File name for root file defining the baseline of normalized residual width per sector for design geometry
    BaselineFile = cms.string(os.environ['CMSSW_BASE'] + '/src/Alignment/APEEstimation/hists/baselineApe.root'),
    
    #File name for root file used for iterations where calculated squared APE values are written to
    IterationFile = cms.string(os.environ['CMSSW_BASE'] + '/src/Alignment/APEEstimation/hists/iterationApe.root'),
    
    #File name for root file used for iterations where calculated squared APE values are written to
    DefaultFile = cms.string(os.environ['CMSSW_BASE'] + '/src/Alignment/APEEstimation/hists/defaultApe.root'),
    
    #File name for text file where calculated APE values are written to, used for DBobject creation
    ApeOutputFile = cms.string(os.environ['CMSSW_BASE'] + '/src/Alignment/APEEstimation/hists/apeOutput.txt'),
    
)
