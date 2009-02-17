import FWCore.ParameterSet.Config as cms

demo = cms.EDAnalyzer("TagProbeEDMAnalysis",
    # Efficiency/Fitting variables
    CalculateEffSideBand = cms.untracked.bool(True),
    NameVar2 = cms.untracked.string('eta'),
    # Variables for sideband subtraction
    SBSPeak = cms.untracked.double(91.1876),
    # Variable Specifications for SB subtractions and Roofit
    # Choose binned or unbinned fitter ...
    # Note that the unbinned fit will not fit weighted data,
    # if you wish to use weights, use the binned fit.
    UnbinnedFit = cms.untracked.bool(True),
    # Variables and binning for the eff hists
    # Valid variables names for the eff binning are:
    # "pt","p","px","py","pz","e","et","eta" and "phi"
    # This way of declaring the bin will overide any other
    # If omitted the defaults are var1 = pt and var2 = eta
    NameVar1 = cms.untracked.string('phi'),
    NumBkgPass = cms.untracked.vdouble(1000.0, 0.0, 10000.0),
    NumBinsVar1 = cms.untracked.int32(28),
    NumBinsVar2 = cms.untracked.int32(30),
    # There is also an option to read the variables in 
    # via a file. This allows for much greater binning flexability
    # 
    #      
    # Background variables
    BkgAlpha = cms.untracked.vdouble(62.0, 50.0, 70.0),
    # Root file to eff histograms to
    FitFileName = cms.untracked.string('demo.root'),
    BkgPeak = cms.untracked.vdouble(91.1876),
    # Binning for the above plots 
    XBins = cms.untracked.vuint32(100),
    SignalWidth = cms.untracked.vdouble(2.8),
    MassLow = cms.untracked.double(80.0),
    SBSStanDev = cms.untracked.double(2.0), ## SD from peak for subtraction

    # Mass window for fitting
    # untracked int32 NumBinsMass         = 60
    # untracked double MassLow            = 60.0
    # untracked double MassHigh           = 120.0
    NumBinsMass = cms.untracked.int32(20),
    logY = cms.untracked.vuint32(1),
    # Efficiency variables
    Efficiency = cms.untracked.vdouble(0.9, 0.0, 1.0),
    BkgGamma = cms.untracked.vdouble(0.05, 0.0, 0.1),
    Do2DFit = cms.untracked.bool(True),
    XMax = cms.untracked.vdouble(120.0),
    outputFileNames = cms.untracked.vstring('Zmass_pass.eps'),
    CalculateEffFitter = cms.untracked.bool(True), ## effs from Roofit

    SignalSigma = cms.untracked.vdouble(1.1, 0.5, 4.0),
    # Make some plots of tree variables ...
    quantities = cms.untracked.vstring('TPmass'),
    Var2High = cms.untracked.double(3.0),
    MassHigh = cms.untracked.double(100.0),
    BifurGaussFrac = cms.untracked.vdouble(0.2, 0.01, 0.99),
    useRecoVarsForTruthMatchedCands = cms.untracked.bool(False),
    # Type of Efficiency : 0 => SC-->GsfElectron
    # 1 ==> GsfElectron-->isolation
    # 2 ==> isolation-->id
    # 3 ==> id-->HLT
    TagProbeType = cms.untracked.int32(0),
    # Fitter variables - for the Roofit fitter
    # If you want the variable to float in the fit fill
    # three array elements {default, range_low, range_high}
    # If the variable should be fixed, fill one element {value}
    # Signal variables
    SignalMean = cms.untracked.vdouble(91.1876),
    XMin = cms.untracked.vdouble(50.0),
    Var1High = cms.untracked.double(3.5),
    NumBkgFail = cms.untracked.vdouble(10.0, 0.0, 10000.0),
    BkgBeta = cms.untracked.vdouble(0.001, 0.0, 0.1),
    conditions = cms.untracked.vstring('TPppass==1'),
    SignalWidthL = cms.untracked.vdouble(3.0, 0.0, 20.0),
    SignalWidthR = cms.untracked.vdouble(0.52, 0.0, 10.0),
    Var2Low = cms.untracked.double(-3.0),
    Var1Low = cms.untracked.double(-3.5),
    CalculateEffTruth = cms.untracked.bool(False), ## true effs

    NumSignal = cms.untracked.vdouble(4000.0, 0.0, 100000.0)
)


