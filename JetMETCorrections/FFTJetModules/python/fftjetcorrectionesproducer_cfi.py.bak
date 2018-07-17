import FWCore.ParameterSet.Config as cms

class ValidFFTJetCorr:
    """
    A class which contains the info about a valid combination
    of ES record types, ES producer type, and jet type
    """
    def __init__(self, basename, jetType):
        self.basename = basename
        self.dbTag = "FFT" + basename + "CorrectorDBTag"
        self.dbRecord = "FFT" + basename + "CorrectorParametersRcd"
        self.correctorRecord = "FFT" + basename + "CorrectorSequenceRcd"
        self.esProducer = "FFT" + basename + "CorrectionESProducer"
        self.jetType = jetType

# The following dictionary contains valid combinations of ES record
# types, ES producer type, and jet type for FFTJet jet corrections.
#
# The dictionary keys used here correspond to the types defined in
# CondFormats/JetMETObjects/interface/FFTJetCorrTypes.h
#
# The database ES record types are listed in
# CondFormats/DataRecord/interface/FFTJetCorrectorParametersRcdTypes.h
#
# The jet correction sequence types (which depend on the corresponding
# database ES record types) are listed in
# JetMETCorrections/FFTJetObjects/interface/FFTJetCorrectorSequenceRcdTypes.h
#
# The ES producer types are defined in 
# JetMCorrections/FFTJetModules/plugins/FFTJetCorrectionESProducer.cc
#
fftjet_corr_types = {
    "BasicJet"    : ValidFFTJetCorr("BasicJet", "BasicJet"),
    "GenJet"      : ValidFFTJetCorr("GenJet", "GenJet"),
    "CaloJet"     : ValidFFTJetCorr("CaloJet", "CaloJet"),
    "PFJet"       : ValidFFTJetCorr("PFJet", "PFJet"),
    "TrackJet"    : ValidFFTJetCorr("TrackJet", "TrackJet"),
    "JPTJet"      : ValidFFTJetCorr("JPTJet", "JPTJet"),
    "PFCHS0"      : ValidFFTJetCorr("PFCHS0", "PFJet"),
    "PFCHS1"      : ValidFFTJetCorr("PFCHS1", "PFJet"),
    "PFCHS2"      : ValidFFTJetCorr("PFCHS2", "PFJet"),
    "BasicJetSys" : ValidFFTJetCorr("BasicJetSys", "BasicJet"),
    "GenJetSys"   : ValidFFTJetCorr("GenJetSys", "GenJet"),
    "CaloJetSys"  : ValidFFTJetCorr("CaloJetSys", "CaloJet"),
    "PFJetSys"    : ValidFFTJetCorr("PFJetSys", "PFJet"),
    "TrackJetSys" : ValidFFTJetCorr("TrackJetSys", "TrackJet"),
    "JPTJetSys"   : ValidFFTJetCorr("JPTJetSys", "JPTJet"),
    "PFCHS0Sys"   : ValidFFTJetCorr("PFCHS0Sys", "PFJet"),
    "PFCHS1Sys"   : ValidFFTJetCorr("PFCHS1Sys", "PFJet"),
    "PFCHS2Sys"   : ValidFFTJetCorr("PFCHS2Sys", "PFJet"),
    "Gen0"        : ValidFFTJetCorr("Gen0", "GenJet"),
    "Gen1"        : ValidFFTJetCorr("Gen1", "GenJet"),
    "Gen2"        : ValidFFTJetCorr("Gen2", "GenJet"),
    "PF0"         : ValidFFTJetCorr("PF0", "PFJet"),
    "PF1"         : ValidFFTJetCorr("PF1", "PFJet"),
    "PF2"         : ValidFFTJetCorr("PF2", "PFJet"),
    "PF3"         : ValidFFTJetCorr("PF3", "PFJet"),
    "PF4"         : ValidFFTJetCorr("PF4", "PFJet"),
    "Calo0"       : ValidFFTJetCorr("Calo0", "CaloJet"),
    "Calo1"       : ValidFFTJetCorr("Calo1", "CaloJet"),
    "Calo2"       : ValidFFTJetCorr("Calo2", "CaloJet"),
    "Calo3"       : ValidFFTJetCorr("Calo3", "CaloJet"),
    "Calo4"       : ValidFFTJetCorr("Calo4", "CaloJet"),
    "Gen0Sys"     : ValidFFTJetCorr("Gen0Sys", "GenJet"),
    "Gen1Sys"     : ValidFFTJetCorr("Gen1Sys", "GenJet"),
    "Gen2Sys"     : ValidFFTJetCorr("Gen2Sys", "GenJet"),
    "PF0Sys"      : ValidFFTJetCorr("PF0Sys", "PFJet"),
    "PF1Sys"      : ValidFFTJetCorr("PF1Sys", "PFJet"),
    "PF2Sys"      : ValidFFTJetCorr("PF2Sys", "PFJet"),
    "PF3Sys"      : ValidFFTJetCorr("PF3Sys", "PFJet"),
    "PF4Sys"      : ValidFFTJetCorr("PF4Sys", "PFJet"),
    "PF5Sys"      : ValidFFTJetCorr("PF5Sys", "PFJet"),
    "PF6Sys"      : ValidFFTJetCorr("PF6Sys", "PFJet"),
    "PF7Sys"      : ValidFFTJetCorr("PF7Sys", "PFJet"),
    "PF8Sys"      : ValidFFTJetCorr("PF8Sys", "PFJet"),
    "PF9Sys"      : ValidFFTJetCorr("PF9Sys", "PFJet"),
    "Calo0Sys"    : ValidFFTJetCorr("Calo0Sys", "CaloJet"),
    "Calo1Sys"    : ValidFFTJetCorr("Calo1Sys", "CaloJet"),
    "Calo2Sys"    : ValidFFTJetCorr("Calo2Sys", "CaloJet"),
    "Calo3Sys"    : ValidFFTJetCorr("Calo3Sys", "CaloJet"),
    "Calo4Sys"    : ValidFFTJetCorr("Calo4Sys", "CaloJet"),
    "Calo5Sys"    : ValidFFTJetCorr("Calo5Sys", "CaloJet"),
    "Calo6Sys"    : ValidFFTJetCorr("Calo6Sys", "CaloJet"),
    "Calo7Sys"    : ValidFFTJetCorr("Calo7Sys", "CaloJet"),
    "Calo8Sys"    : ValidFFTJetCorr("Calo8Sys", "CaloJet"),
    "Calo9Sys"    : ValidFFTJetCorr("Calo9Sys", "CaloJet"),
    "CHS0Sys"     : ValidFFTJetCorr("CHS0Sys", "PFJet"),
    "CHS1Sys"     : ValidFFTJetCorr("CHS1Sys", "PFJet"),
    "CHS2Sys"     : ValidFFTJetCorr("CHS2Sys", "PFJet"),
    "CHS3Sys"     : ValidFFTJetCorr("CHS3Sys", "PFJet"),
    "CHS4Sys"     : ValidFFTJetCorr("CHS4Sys", "PFJet"),
    "CHS5Sys"     : ValidFFTJetCorr("CHS5Sys", "PFJet"),
    "CHS6Sys"     : ValidFFTJetCorr("CHS6Sys", "PFJet"),
    "CHS7Sys"     : ValidFFTJetCorr("CHS7Sys", "PFJet"),
    "CHS8Sys"     : ValidFFTJetCorr("CHS8Sys", "PFJet"),
    "CHS9Sys"     : ValidFFTJetCorr("CHS9Sys", "PFJet")
}

#
# Procedure for configuring the L2-L3 FFTJet ES producer. This producer
# turns the database record into a sequence of jet corrections.
# The "sequenceTag" argument should be set to one of the keys in the
# "fftjet_corr_types" dictionary.
#
def configure_L2L3_fftjet_esproducer(sequenceTag, tableName, tableCategory):
    #
    # The ES producer name comes from the C++ plugin registration code
    esProducer = fftjet_corr_types[sequenceTag].esProducer
    config = cms.ESProducer(
        esProducer,
        sequence = cms.VPSet(
            cms.PSet(
                level = cms.uint32(2),
                applyTo = cms.string("DataOrMC"),
                adjuster = cms.PSet(
                    Class = cms.string("FFTSimpleScalingAdjuster")
                ),
                scalers = cms.VPSet(
                    cms.PSet(
                        Class = cms.string("auto"),
                        name = cms.string(tableName),
                        nameIsRegex = cms.bool(False),
                        category = cms.string(tableCategory),
                        categoryIsRegex = cms.bool(False)
                    )
                )
            )
        ),
        isArchiveCompressed = cms.bool(False),
        verbose = cms.untracked.bool(False)
    )
    return (config, esProducer)

#
# ES producer for L2 residual corrections
#
def configure_L2Res_fftjet_esproducer(sequenceTag, tableName, tableCategory):
    #
    # The ES producer name comes from the C++ plugin registration code
    esProducer = fftjet_corr_types[sequenceTag].esProducer
    config = cms.ESProducer(
        esProducer,
        sequence = cms.VPSet(
            cms.PSet(
                level = cms.uint32(3),
                applyTo = cms.string("DataOnly"),
                adjuster = cms.PSet(
                    Class = cms.string("FFTSimpleScalingAdjuster")
                ),
                scalers = cms.VPSet(
                    cms.PSet(
                        Class = cms.string("FFTSpecificScaleCalculator"),
                        Subclass = cms.PSet(
                                Class = cms.string("L2ResScaleCalculator"),
                                radiusFactor = cms.double(1.0)
                        ),
                        name = cms.string(tableName),
                        nameIsRegex = cms.bool(False),
                        category = cms.string(tableCategory),
                        categoryIsRegex = cms.bool(False)
                    )
                )
            )
        ),
        isArchiveCompressed = cms.bool(False),
        verbose = cms.untracked.bool(False)
    )
    return (config, esProducer)

#
# ES producer for L3 residual corrections
#
def configure_L3Res_fftjet_esproducer(sequenceTag, tableName, tableCategory):
    #
    # The ES producer name comes from the C++ plugin registration code
    esProducer = fftjet_corr_types[sequenceTag].esProducer
    config = cms.ESProducer(
        esProducer,
        sequence = cms.VPSet(
            cms.PSet(
                level = cms.uint32(4),
                applyTo = cms.string("DataOnly"),
                adjuster = cms.PSet(
                    Class = cms.string("FFTSimpleScalingAdjuster")
                ),
                scalers = cms.VPSet(
                    cms.PSet(
                        Class = cms.string("FFTSpecificScaleCalculator"),
                        Subclass = cms.PSet(
                                Class = cms.string("L2RecoScaleCalculator"),
                                radiusFactor = cms.double(1.0)
                        ),
                        name = cms.string(tableName),
                        nameIsRegex = cms.bool(False),
                        category = cms.string(tableCategory),
                        categoryIsRegex = cms.bool(False)
                    )
                )
            )
        ),
        isArchiveCompressed = cms.bool(False),
        verbose = cms.untracked.bool(False)
    )
    return (config, esProducer)

#
# Helper function for configuring FFTGenericScaleCalculator
#
def configure_FFTGenericScaleCalculator(variables, factorsForTheseVariables):
    if len(variables) == 0:
        raise ValueError("Must have at least one variable mapped")
    if len(variables) != len(factorsForTheseVariables):
        raise ValueError("Incompatible length of the input arguments")
    subclass = cms.PSet(
        Class = cms.string("FFTGenericScaleCalculator"),
        factors = cms.vdouble(factorsForTheseVariables),
        eta=cms.int32(-1),
        phi=cms.int32(-1),
        pt=cms.int32(-1),
        logPt=cms.int32(-1),
        mass=cms.int32(-1),
        logMass=cms.int32(-1),
        energy=cms.int32(-1),
        logEnergy=cms.int32(-1),
        gamma=cms.int32(-1),
        logGamma=cms.int32(-1),
        pileup=cms.int32(-1),
        ncells=cms.int32(-1),
        etSum=cms.int32(-1),
        etaWidth=cms.int32(-1),
        phiWidth=cms.int32(-1),
        averageWidth=cms.int32(-1),
        widthRatio=cms.int32(-1),
        etaPhiCorr=cms.int32(-1),
        fuzziness=cms.int32(-1),
        convergenceDistance=cms.int32(-1),
        recoScale=cms.int32(-1),
        recoScaleRatio=cms.int32(-1),
        membershipFactor=cms.int32(-1),
        magnitude=cms.int32(-1),
        logMagnitude=cms.int32(-1),
        magS1=cms.int32(-1),
        LogMagS1=cms.int32(-1),
        magS2=cms.int32(-1),
        LogMagS2=cms.int32(-1),
        driftSpeed=cms.int32(-1),
        magSpeed=cms.int32(-1),
        lifetime=cms.int32(-1),
        splitTime=cms.int32(-1),
        mergeTime=cms.int32(-1),
        scale=cms.int32(-1),
        logScale=cms.int32(-1),
        nearestNeighborDistance=cms.int32(-1),
        clusterRadius=cms.int32(-1),
        clusterSeparation=cms.int32(-1),
        dRFromJet=cms.int32(-1),
        LaplacianS1=cms.int32(-1),
        LaplacianS2=cms.int32(-1),
        LaplacianS3=cms.int32(-1),
        HessianS2=cms.int32(-1),
        HessianS4=cms.int32(-1),
        HessianS6=cms.int32(-1),
        nConstituents=cms.int32(-1),
        aveConstituentPt=cms.int32(-1),
        logAveConstituentPt=cms.int32(-1),
        constituentPtDistribution=cms.int32(-1),
        constituentEtaPhiSpread=cms.int32(-1),
        chargedHadronEnergyFraction=cms.int32(-1),
        neutralHadronEnergyFraction=cms.int32(-1),
        photonEnergyFraction=cms.int32(-1),
        electronEnergyFraction=cms.int32(-1),
        muonEnergyFraction=cms.int32(-1),
        HFHadronEnergyFraction=cms.int32(-1),
        HFEMEnergyFraction=cms.int32(-1),
        chargedHadronMultiplicity=cms.int32(-1),
        neutralHadronMultiplicity=cms.int32(-1),
        photonMultiplicity=cms.int32(-1),
        electronMultiplicity=cms.int32(-1),
        muonMultiplicity=cms.int32(-1),
        HFHadronMultiplicity=cms.int32(-1),
        HFEMMultiplicity=cms.int32(-1),
        chargedEmEnergyFraction=cms.int32(-1),
        chargedMuEnergyFraction=cms.int32(-1),
        neutralEmEnergyFraction=cms.int32(-1),
        EmEnergyFraction=cms.int32(-1),
        chargedMultiplicity=cms.int32(-1),
        neutralMultiplicity=cms.int32(-1)
    )
    for i, varname in enumerate(variables):
        setattr(subclass, varname, cms.int32(i))
    return subclass

#
# Procedure for configuring the ES source which fetches
# the database record. "process.CondDBCommon" should be
# already defined before calling this procedure.
#
def configure_fftjet_pooldbessource(process, sequenceTag):
    config = cms.ESSource(
        "PoolDBESSource",
        process.CondDBCommon,
        toGet = cms.VPSet(cms.PSet(
            record = cms.string(fftjet_corr_types[sequenceTag].dbRecord),
            tag = cms.string(fftjet_corr_types[sequenceTag].dbTag),
        ))
    )
    sourceName = "FFT" + sequenceTag + "DBESSource"
    setattr(process, sourceName, config)
    return
