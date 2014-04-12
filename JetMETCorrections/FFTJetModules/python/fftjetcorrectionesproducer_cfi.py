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
