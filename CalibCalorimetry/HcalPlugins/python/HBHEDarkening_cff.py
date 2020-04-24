import FWCore.ParameterSet.Config as cms

# energy = center of mass energy (beam energy x2), GeV
# [intlumi] = fb-1
# [lumirate] = fb-1/hr
# dose maps only produced for 8 and 14 TeV - used also for 7 and 13 TeV, respectively

# total for phase0/1 ~ 300 fb-1
_years_LHC = cms.VPSet(
    cms.PSet(year = cms.string("2011"), intlumi = cms.double(5.6), lumirate = cms.double(0.005), energy = cms.int32(8)),
    cms.PSet(year = cms.string("2012"), intlumi = cms.double(23.3), lumirate = cms.double(0.013), energy = cms.int32(8)),
    cms.PSet(year = cms.string("2015"), intlumi = cms.double(4.1), lumirate = cms.double(0.009), energy = cms.int32(14)),
    cms.PSet(year = cms.string("2016"), intlumi = cms.double(41.0), lumirate = cms.double(0.026), energy = cms.int32(14)),
    cms.PSet(year = cms.string("2017"), intlumi = cms.double(45.0), lumirate = cms.double(0.043), energy = cms.int32(14)),
    cms.PSet(year = cms.string("2018"), intlumi = cms.double(45.0), lumirate = cms.double(0.043), energy = cms.int32(14)),
    cms.PSet(year = cms.string("2021"), intlumi = cms.double(45.0), lumirate = cms.double(0.05), energy = cms.int32(14)),
    cms.PSet(year = cms.string("2022"), intlumi = cms.double(45.0), lumirate = cms.double(0.05), energy = cms.int32(14)),
    cms.PSet(year = cms.string("2023"), intlumi = cms.double(50.0), lumirate = cms.double(0.05), energy = cms.int32(14)),
)

# total for phase2 nominal = 3000 fb-1 (including phase0/1) @ 5.0E34/cm^2/s
_years_HLLHC_nominal = cms.VPSet(
    cms.PSet(year = cms.string("2038"), intlumi = cms.double(2700), lumirate = cms.double(0.15), energy = cms.int32(14)),
)

# total for phase2 ultimate = 4500 fb-1 (including phase0/1) @ 7.5E34/cm^2/s
_years_HLLHC_ultimate = cms.VPSet(
    cms.PSet(year = cms.string("2029"), intlumi = cms.double(700), lumirate = cms.double(0.15), energy = cms.int32(14)),
    cms.PSet(year = cms.string("2039"), intlumi = cms.double(3500), lumirate = cms.double(0.225), energy = cms.int32(14)),
)

HEDarkeningEP = cms.ESSource("HBHEDarkeningEP",
    appendToDataLabel = cms.string("HE"),
    ieta_shift = cms.int32(16),
    # parameters taken from https://indico.cern.ch/event/641946/contributions/2604357/attachments/1466160/2266650/PlanB_TDR.pdf, slide 4 (brown line)
    drdA = cms.double(5.0),
    drdB = cms.double(0.675),
    dosemaps = cms.VPSet(
        cms.PSet(energy = cms.int32(8), file = cms.FileInPath("CalibCalorimetry/HcalPlugins/data/dosemapHE_4TeV.txt")),
        cms.PSet(energy = cms.int32(14), file = cms.FileInPath("CalibCalorimetry/HcalPlugins/data/dosemapHE_7TeV.txt")),
    ),
    years = _years_LHC + _years_HLLHC_nominal,
)

HBDarkeningEP = HEDarkeningEP.clone(
    appendToDataLabel = cms.string("HB"),
    ieta_shift = cms.int32(1),
    dosemaps = cms.VPSet(
        cms.PSet(energy = cms.int32(8), file = cms.FileInPath("CalibCalorimetry/HcalPlugins/data/dosemapHB_4TeV.txt")),
        cms.PSet(energy = cms.int32(14), file = cms.FileInPath("CalibCalorimetry/HcalPlugins/data/dosemapHB_7TeV.txt")),
    ),
)
