import FWCore.ParameterSet.Config as cms

## HF Recalibration Parameters
from DataFormats.HcalCalibObjects.HFRecalibrationParameters_cff import *

hcal_db_producer = cms.ESProducer("HcalDbProducer",
    dump = cms.untracked.vstring(''),
    file = cms.untracked.string('')
)

#------------------------- HARDCODED conditions

es_hardcode = cms.ESSource("HcalHardcodeCalibrations",
    toGet = cms.untracked.vstring('GainWidths'),
    iLumi = cms.double(-1.),                      # for Upgrade: fb-1
    HERecalibration = cms.bool(False),            # True for Upgrade
    HEreCalibCutoff = cms.double(20.),            # if above is True
    HFRecalibration = cms.bool(False),            # True for Upgrade
    HFRecalParameterBlock = HFRecalParameterBlock,
    GainWidthsForTrigPrims = cms.bool(False),     # True Upgrade
    useHBUpgrade = cms.bool(False),
    useHEUpgrade = cms.bool(False),
    useHFUpgrade = cms.bool(False),
    useHOUpgrade = cms.bool(True),
    testHFQIE10  = cms.bool(False),
    testHEPlan1  = cms.bool(False),
    killHE = cms.bool(False),
    useLayer0Weight = cms.bool(False),
    hb = cms.PSet(
        pedestal      = cms.double(3.285),
        pedestalWidth = cms.double(0.809),
        gain          = cms.vdouble(0.19),
        gainWidth     = cms.vdouble(0.0),
        qieType       = cms.int32(0),
        qieOffset     = cms.vdouble(-0.49,1.8,7.2,37.9),
        qieSlope      = cms.vdouble(0.912,0.917,0.922,0.923),
        mcShape       = cms.int32(125),
        recoShape     = cms.int32(105),
        photoelectronsToAnalog = cms.double(0.3305),
        darkCurrent   = cms.vdouble(0.0),
    ),
    he = cms.PSet(
        pedestal      = cms.double(3.163),
        pedestalWidth = cms.double(0.9698),
        gain          = cms.vdouble(0.23),
        gainWidth     = cms.vdouble(0),
        qieType       = cms.int32(0),
        qieOffset     = cms.vdouble(-0.38,2.0,7.6,39.6),
        qieSlope      = cms.vdouble(0.912,0.916,0.920,0.922),
        mcShape       = cms.int32(125),
        recoShape     = cms.int32(105),
        photoelectronsToAnalog = cms.double(0.3305),
        darkCurrent   = cms.vdouble(0.0),
    ),
    hf = cms.PSet(
        pedestal      = cms.double(9.354),
        pedestalWidth = cms.double(2.516),
        gain          = cms.vdouble(0.14,0.135),
        gainWidth     = cms.vdouble(0.0,0.0),
        qieType       = cms.int32(0),
        qieOffset     = cms.vdouble(-0.87,1.4,7.8,-29.6),
        qieSlope      = cms.vdouble(0.359,0.358,0.360,0.367),
        mcShape       = cms.int32(301),
        recoShape     = cms.int32(301),
        photoelectronsToAnalog = cms.double(0.0),
        darkCurrent   = cms.vdouble(0.0),
    ),
    ho = cms.PSet(
        pedestal      = cms.double(12.06),
        pedestalWidth = cms.double(0.6285),
        gain          = cms.vdouble(0.0060,0.0087),
        gainWidth     = cms.vdouble(0.0,0.0),
        qieType       = cms.int32(0),
        qieOffset     = cms.vdouble(-0.44,1.4,7.1,38.5),
        qieSlope      = cms.vdouble(0.907,0.915,0.920,0.921),
        mcShape       = cms.int32(201),
        recoShape     = cms.int32(201),
        photoelectronsToAnalog = cms.double(4.0),
        darkCurrent   = cms.vdouble(0.0),
    ),
    hbUpgrade = cms.PSet(
        pedestal      = cms.double(17.3),
        pedestalWidth = cms.double(1.5),
        gain          = cms.vdouble(1/2276.), #51.72 is pe/GeV 44.0 is fC/pe.
        gainWidth     = cms.vdouble(0),
        qieType       = cms.int32(2),
        qieOffset     = cms.vdouble(0.,0.,0.,0.),
        qieSlope      = cms.vdouble(0.05376,0.05376,0.05376,0.05376), #1/(3.1*6) where 6 is shunt factor
        mcShape       = cms.int32(203),
        recoShape     = cms.int32(203),
        photoelectronsToAnalog = cms.double(44.0),
        darkCurrent   = cms.vdouble(0.01,0.015),
    ),
    heUpgrade = cms.PSet(
        pedestal      = cms.double(17.3),
        pedestalWidth = cms.double(1.5),
        gain          = cms.vdouble(1/2276.), #51.72 is pe/GeV 44.0 is fC/pe.
        gainWidth     = cms.vdouble(0),
        qieType       = cms.int32(2),
        qieOffset     = cms.vdouble(0.,0.,0.,0.),
        qieSlope      = cms.vdouble(0.05376,0.05376,0.05376,0.05376), #1/(3.1*6) where 6 is shunt factor
        mcShape       = cms.int32(203),
        recoShape     = cms.int32(203),
        photoelectronsToAnalog = cms.double(44.0),
        darkCurrent   = cms.vdouble(0.01,0.015),
    ),
    hfUpgrade = cms.PSet(
        pedestal      = cms.double(13.33),
        pedestalWidth = cms.double(3.33),
        gain          = cms.vdouble(0.14,0.135),
        gainWidth     = cms.vdouble(0.0,0.0),
        qieType       = cms.int32(1),
        qieOffset     = cms.vdouble(0.0697,-0.7405,12.38,-671.9),
        qieSlope      = cms.vdouble(0.297,0.298,0.298,0.313),
        mcShape       = cms.int32(301),
        recoShape     = cms.int32(301),
        photoelectronsToAnalog = cms.double(0.0),
        darkCurrent   = cms.vdouble(0.0),
    ),
    # types (in order): HcalHOZecotek, HcalHOHamamatsu, HcalHEHamamatsu1, HcalHEHamamatsu2, HcalHBHamamatsu1, HcalHBHamamatsu2, HcalHPD
    SiPMCharacteristics = cms.VPSet(
        cms.PSet( pixels = cms.int32(36000), crosstalk = cms.double(0.0), nonlin1 = cms.double(1.0), nonlin2 = cms.double(0.0), nonlin3 = cms.double(0.0) ),
        cms.PSet( pixels = cms.int32(2500), crosstalk = cms.double(0.0), nonlin1 = cms.double(1.0), nonlin2 = cms.double(0.0), nonlin3 = cms.double(0.0) ),
        cms.PSet( pixels = cms.int32(27370), crosstalk = cms.double(0.17), nonlin1 = cms.double(1.00985), nonlin2 = cms.double(7.84089E-6), nonlin3 = cms.double(2.86282E-10) ),
        cms.PSet( pixels = cms.int32(38018), crosstalk = cms.double(0.196), nonlin1 = cms.double(1.00546), nonlin2 = cms.double(6.40239E-6), nonlin3 = cms.double(1.27011E-10) ),
        cms.PSet( pixels = cms.int32(27370), crosstalk = cms.double(0.17), nonlin1 = cms.double(1.00985), nonlin2 = cms.double(7.84089E-6), nonlin3 = cms.double(2.86282E-10) ),
        cms.PSet( pixels = cms.int32(38018), crosstalk = cms.double(0.196), nonlin1 = cms.double(1.00546), nonlin2 = cms.double(6.40239E-6), nonlin3 = cms.double(1.27011E-10) ),
        cms.PSet( pixels = cms.int32(0), crosstalk = cms.double(0.0), nonlin1 = cms.double(1.0), nonlin2 = cms.double(0.0), nonlin3 = cms.double(0.0) ),
    ),
)

es_prefer_hcalHardcode = cms.ESPrefer("HcalHardcodeCalibrations", "es_hardcode")

from Configuration.Eras.Modifier_phase2_hcal_cff import phase2_hcal
phase2_hcal.toModify( es_hardcode,
                             toGet = cms.untracked.vstring(
                                         'GainWidths',
                                         'MCParams',
                                         'RecoParams',
                                         'RespCorrs',
                                         'QIEData',
                                         'QIETypes',
                                         'Gains',
                                         'Pedestals',
                                         'PedestalWidths',
                                         'ChannelQuality',
                                         'ZSThresholds',
                                         'TimeCorrs',
                                         'LUTCorrs',
                                         'LutMetadata',
                                         'L1TriggerObjects',
                                         'PFCorrs',
                                         'ElectronicsMap',
                                         'FrontEndMap',
                                         'CovarianceMatrices',
                                         'SiPMParameters',
                                         'SiPMCharacteristics',
                                         'TPChannelParameters',
                                         'TPParameters',
                                         'FlagHFDigiTimeParams'
                                         ),
                             GainWidthsForTrigPrims = cms.bool(True),
                             HEreCalibCutoff = cms.double(100.),
                             useHBUpgrade = cms.bool(True),
                             useHEUpgrade = cms.bool(True),
                             useHFUpgrade = cms.bool(True),
                             useLayer0Weight = cms.bool(True),
)

from Configuration.Eras.Modifier_phase2_hgcal_cff import phase2_hgcal
phase2_hgcal.toModify( es_hardcode, killHE = cms.bool(True) )
                            
