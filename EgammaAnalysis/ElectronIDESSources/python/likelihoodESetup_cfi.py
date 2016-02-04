import FWCore.ParameterSet.Config as cms

# ESSource for Electron ID likelihood
ElectronLikelihoodESSource = cms.ESProducer("ElectronLikelihoodESSource",
    useEoverPOut = cms.double(1.0),
    piEBNarrowFracGt15 = cms.double(1.0),
    #! fisher coefficients
    fisherCoeffEBLt15_constant = cms.double(0.693496),
    piWeight = cms.double(1.0),
    fisherCoeffEBLt15_sigmaEtaEta = cms.double(-12.7018),
    piEENarrowFracGt15 = cms.double(1.0),
    eleEBNarrowFracGt15 = cms.double(1.0),
    useShapeFisher = cms.double(0.0),
    piEBShoweringFracGt15 = cms.double(1.0),
    piEBShoweringFracLt15 = cms.double(1.0),
    piEEShoweringFracGt15 = cms.double(1.0),
    piEEGoldenFracGt15 = cms.double(1.0),
    eleEBGoldenFracGt15 = cms.double(1.0),
    #! a priori probabilities (fullclass fractions). Use flat priors for now
    eleEBGoldenFracLt15 = cms.double(1.0),
    piEEBigbremFracLt15 = cms.double(1.0),
    eleEEGoldenFracGt15 = cms.double(1.0),
    fisherCoeffEBLt15_s9s25 = cms.double(1.23863),
    eleEBBigbremFracGt15 = cms.double(1.0),
    eleEBBigbremFracLt15 = cms.double(1.0),
    fisherCoeffEBGt15_constant = cms.double(6.02184),
    fisherCoeffEEGt15_etaLat = cms.double(-9.3025),
    splitBackgroundPdfs = cms.bool(False),
    piEBNarrowFracLt15 = cms.double(1.0),
    fisherCoeffEBGt15_s9s25 = cms.double(2.49634),
    piEBBigbremFracGt15 = cms.double(1.0),
    fisherCoeffEELt15_s9s25 = cms.double(4.51575),
    eleEBShoweringFracGt15 = cms.double(1.0),
    #! a priori probabilies having an electron/hadron (cross sections)
    eleWeight = cms.double(1.0),
    eleEEGoldenFracLt15 = cms.double(1.0),
    useSigmaEtaEta = cms.double(1.0),
    piEENarrowFracLt15 = cms.double(1.0),
    eleEEShoweringFracLt15 = cms.double(1.0),
    #! use dedicated PDF's for each class defined by 
    #! signalWeightSplitting category
    #! for now no splitted PDFs are in the DB for Bkg (lack of statistics)
    splitSignalPdfs = cms.bool(True),
    fisherCoeffEELt15_constant = cms.double(-1.11814),
    piEBBigbremFracLt15 = cms.double(1.0),
    piEBGoldenFracLt15 = cms.double(1.0),
    eleEBNarrowFracLt15 = cms.double(1.0),
    fisherCoeffEBGt15_etaLat = cms.double(-30.1528),
    fisherCoeffEELt15_sigmaEtaEta = cms.double(-5.3288),
    fisherCoeffEELt15_a20 = cms.double(0.0),
    fisherCoeffEBGt15_sigmaEtaEta = cms.double(-49.2656),
    fisherCoeffEEGt15_s9s25 = cms.double(3.61809),
    fisherCoeffEEGt15_sigmaEtaEta = cms.double(-11.7401),
    piEEGoldenFracLt15 = cms.double(1.0),
    backgroundWeightSplitting = cms.string('class'),
    useHoverE = cms.double(1.0),
    fisherCoeffEELt15_etaLat = cms.double(-6.47578),
    eleEEBigbremFracLt15 = cms.double(1.0),
    #! switch the use of one variable ON / OFF
    useDeltaEtaCalo = cms.double(1.0),
    #! PDF's splitting rule
    #! class: split by non-showering / showering+cracks
    #! fullclass: split by golden / bigbrem / narrow / showering+cracks
    #    string signalWeightSplitting = "fullclass"
    #    string backgroundWeightSplitting = "fullclass"
    signalWeightSplitting = cms.string('class'),
    eleEENarrowFracLt15 = cms.double(1.0),
    eleEEBigbremFracGt15 = cms.double(1.0),
    eleEENarrowFracGt15 = cms.double(1.0),
    fisherCoeffEBGt15_a20 = cms.double(0.0),
    piEEShoweringFracLt15 = cms.double(1.0),
    eleEEShoweringFracGt15 = cms.double(1.0),
    eleEBShoweringFracLt15 = cms.double(1.0),
    piEEBigbremFracGt15 = cms.double(1.0),
    fisherCoeffEEGt15_constant = cms.double(0.536351),
    useDeltaPhiIn = cms.double(1.0),
    fisherCoeffEBLt15_etaLat = cms.double(-10.115),
    useE9overE25 = cms.double(1.0),
    fisherCoeffEEGt15_a20 = cms.double(0.0),
    fisherCoeffEBLt15_a20 = cms.double(0.0),
    piEBGoldenFracGt15 = cms.double(1.0)
)


