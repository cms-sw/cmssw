import FWCore.ParameterSet.Config as cms

class ValidFFTJetLUT:
    """
    A class which contains the info about a valid combination
    of ES record types and ES producer type for FFTJet lookup tables
    """
    def __init__(self, basename):
        self.basename = basename
        self.dbTag = "FFT" + basename + "TableDBTag"
        self.dbRecord = "FFT" + basename + "ParametersRcd"
        self.LUTRecord = "FFT" + basename + "TableRcd"
        self.esProducer = "FFT" + basename + "TableESProducer"

# The following dictionary contains valid combinations of ES record
# types and ES producer type for FFTJet (pileup) lookup tables.
#
# The dictionary keys used here correspond to the types defined in
# CondFormats/JetMETObjects/interface/FFTJetLUTTypes.h
#
# The database ES record types are listed in
# CondFormats/DataRecord/interface/FFTJetCorrectorParametersRcdTypes.h
#
# The jet correction sequence types (which depend on the corresponding
# database ES record types) are listed in
# JetMETCorrections/FFTJetObjects/interface/FFTJetLookupTableRcdTypes.h
#
# The ES producer types are defined in 
# JetMCorrections/FFTJetModules/plugins/FFTJetLookupTableESProducer.cc
#
fftjet_lut_types = {
    "EtaFlatteningFactors"   : ValidFFTJetLUT("EtaFlatteningFactors"),
    "PileupRhoCalibration"   : ValidFFTJetLUT("PileupRhoCalibration"),
    "PileupRhoEtaDependence" : ValidFFTJetLUT("PileupRhoEtaDependence"),
    "LUT0"                   : ValidFFTJetLUT("LUT0"),
    "LUT1"                   : ValidFFTJetLUT("LUT1"),
    "LUT2"                   : ValidFFTJetLUT("LUT2"),
    "LUT3"                   : ValidFFTJetLUT("LUT3"),
    "LUT4"                   : ValidFFTJetLUT("LUT4"),
    "LUT5"                   : ValidFFTJetLUT("LUT5"),
    "LUT6"                   : ValidFFTJetLUT("LUT6"),
    "LUT7"                   : ValidFFTJetLUT("LUT7"),
    "LUT8"                   : ValidFFTJetLUT("LUT8"),
    "LUT9"                   : ValidFFTJetLUT("LUT9"),
    "LUT10"                  : ValidFFTJetLUT("LUT10"),
    "LUT11"                  : ValidFFTJetLUT("LUT11"),
    "LUT12"                  : ValidFFTJetLUT("LUT12"),
    "LUT13"                  : ValidFFTJetLUT("LUT13"),
    "LUT14"                  : ValidFFTJetLUT("LUT14"),
    "LUT15"                  : ValidFFTJetLUT("LUT15")
}

#
# Procedure for configuring the L2-L3 FFTJet ES producer. This
# producer turns the database record into a set of lookup table types.
# The "sequenceTag" argument should be set to one of the keys in the
# "fftjet_lut_types" dictionary.
#
def configure_fftjetlut_esproducer(sequenceTag):
    #
    # The ES producer name comes from the C++ plugin registration code
    esProducer = fftjet_lut_types[sequenceTag].esProducer
    config = cms.ESProducer(
        esProducer,
        tables = cms.VPSet(
	    cms.PSet(
		name = cms.string('.*'),
		nameIsRegex = cms.bool(True),
		category = cms.string('.*'),
		categoryIsRegex = cms.bool(True)
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
def configure_fftjetlut_pooldbessource(process, sequenceTag):
    config = cms.ESSource(
        "PoolDBESSource",
        process.CondDBCommon,
        toGet = cms.VPSet(cms.PSet(
            record = cms.string(fftjet_lut_types[sequenceTag].dbRecord),
            tag = cms.string(fftjet_lut_types[sequenceTag].dbTag),
        ))
    )
    sourceName = "FFT" + sequenceTag + "DBESSource"
    setattr(process, sourceName, config)
    return
