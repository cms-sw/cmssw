import FWCore.ParameterSet.Config as cms

# energy = center of mass energy (beam energy x2), GeV
# [intlumi] = fb-1
# [lumirate] = fb-1/hr
# dose maps only produced for 8 and 14 TeV - used also for 7 and 13 TeV, respectively

HEDarkeningEP = cms.ESSource("HBHEDarkeningEP",
    appendToDataLabel = cms.string("HE"),
    ieta_shift = cms.int32(16),
    drdA = cms.double(4.0),
    drdB = cms.double(0.575),
    dosemaps = cms.VPSet(
        cms.PSet(energy = cms.int32(8), file = cms.FileInPath("CalibCalorimetry/HcalPlugins/data/dosemapHE_4TeV.txt")),
        cms.PSet(energy = cms.int32(14), file = cms.FileInPath("CalibCalorimetry/HcalPlugins/data/dosemapHE_7TeV.txt")),
    ),
    years = cms.VPSet(
        cms.PSet(year = cms.string("2011"), intlumi = cms.double(5.6), lumirate = cms.double(0.005), energy = cms.int32(8)),
        cms.PSet(year = cms.string("2012"), intlumi = cms.double(23.3), lumirate = cms.double(0.013), energy = cms.int32(8)),
        cms.PSet(year = cms.string("2015"), intlumi = cms.double(4.1), lumirate = cms.double(0.009), energy = cms.int32(14)),
        cms.PSet(year = cms.string("2016"), intlumi = cms.double(41.0), lumirate = cms.double(0.026), energy = cms.int32(14)),
        cms.PSet(year = cms.string("2017"), intlumi = cms.double(45.0), lumirate = cms.double(0.043), energy = cms.int32(14)),
        cms.PSet(year = cms.string("2018"), intlumi = cms.double(45.0), lumirate = cms.double(0.043), energy = cms.int32(14)),
        cms.PSet(year = cms.string("2021"), intlumi = cms.double(45.0), lumirate = cms.double(0.05), energy = cms.int32(14)),
        cms.PSet(year = cms.string("2022"), intlumi = cms.double(45.0), lumirate = cms.double(0.05), energy = cms.int32(14)),
        cms.PSet(year = cms.string("2023"), intlumi = cms.double(50.0), lumirate = cms.double(0.05), energy = cms.int32(14)),
        # assume 3000 fb-1 for Phase2
        cms.PSet(year = cms.string("2033"), intlumi = cms.double(3000), lumirate = cms.double(0.15), energy = cms.int32(14)),
    ),
)

HBDarkeningEP = HEDarkeningEP.clone(
    appendToDataLabel = cms.string("HB"),
    ieta_shift = cms.int32(1),
    dosemaps = cms.VPSet(
        cms.PSet(energy = cms.int32(8), file = cms.FileInPath("CalibCalorimetry/HcalPlugins/data/dosemapHB_4TeV.txt")),
        cms.PSet(energy = cms.int32(14), file = cms.FileInPath("CalibCalorimetry/HcalPlugins/data/dosemapHB_7TeV.txt")),
    ),
)
