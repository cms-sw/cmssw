import FWCore.ParameterSet.Config as cms
import SimCalorimetry.HGCalSimProducers.hgcalDigitizer_cfi as digiparam
import RecoLocalCalo.HGCalRecProducers.HGCalUncalibRecHit_cfi as recoparam
import RecoLocalCalo.HGCalRecProducers.HGCalRecHit_cfi as recocalibparam 

# Digitization parameters
adcSaturation_fC = digiparam.hgceeDigitizer.digiCfg.feCfg.adcSaturation_fC
adcNbits = digiparam.hgceeDigitizer.digiCfg.feCfg.adcNbits
tdcSaturation_fC = digiparam.hgceeDigitizer.digiCfg.feCfg.tdcSaturation_fC
tdcNbits = digiparam.hgceeDigitizer.digiCfg.feCfg.tdcNbits
tdcOnset_fC = digiparam.hgceeDigitizer.digiCfg.feCfg.tdcOnset_fC
# Reco calibration parameters
fCPerMIPee = recoparam.HGCalUncalibRecHit.HGCEEConfig.fCPerMIP
fCperMIPfh = recoparam.HGCalUncalibRecHit.HGCHEFConfig.fCPerMIP
layerWeights = recocalibparam.HGCalRecHit.layerWeights
thicknessCorrection = recocalibparam.HGCalRecHit.thicknessCorrection

# Parameters used in several places
triggerCellLsbBeforeCompression = 100./1024.
triggerCellTruncationBits = 0

fe_codec = cms.PSet( CodecName  = cms.string('HGCalTriggerCellThresholdCodec'),
                     CodecIndex = cms.uint32(2),
                     MaxCellsInModule = cms.uint32(116),
                     DataLength = cms.uint32(16),
                     linLSB = cms.double(triggerCellLsbBeforeCompression),
                     triggerCellTruncationBits = cms.uint32(triggerCellTruncationBits),
                     NData = cms.uint32(999),
                     TCThreshold_fC = cms.double(1.),
                     #take the following parameters from the digitization config file
                     adcsaturation = adcSaturation_fC,
                     adcnBits = adcNbits,
                     tdcsaturation = tdcSaturation_fC,
                     tdcnBits = tdcNbits,
                     tdcOnsetfC = tdcOnset_fC
                     )

calib_parValues = cms.PSet( cellLSB =  cms.double( triggerCellLsbBeforeCompression*(2**triggerCellTruncationBits) ),
                             fCperMIPee = fCPerMIPee,
                             fCperMIPfh = fCperMIPfh,
                             dEdXweights = layerWeights,
                             thickCorr = thicknessCorrection
                            )
C2d_parValues = cms.PSet( seeding_threshold = cms.double(5), # MipT
                          clustering_threshold = cms.double(2), # MipT
                          dR_cluster = cms.double(3.), # in cm
                          clusterType = cms.string('NNC2d') # clustering type: dRC2d--> Geometric-dR clustering; NNC2d-->Nearest Neighbors clustering
                          )

C3d_parValues = cms.PSet( dR_multicluster = cms.double(0.01), # dR in normalized plane used to clusterize C2d
                          minPt_multicluster = cms.double(0.5), # minimum pt of the multicluster (GeV)
                          calibSF_multicluster = cms.double(1.084)
                          )
cluster_algo =  cms.PSet( AlgorithmName = cms.string('HGCClusterAlgoThreshold'),
                          FECodec = fe_codec.clone(),
                          HGCalEESensitive_tag = cms.string('HGCalEESensitive'),
                          HGCalHESiliconSensitive_tag = cms.string('HGCalHESiliconSensitive'),
                          calib_parameters = calib_parValues.clone(),
                          C2d_parameters = C2d_parValues.clone(),
                          C3d_parameters = C3d_parValues.clone()
                          )

hgcalTriggerPrimitiveDigiProducer = cms.EDProducer(
    "HGCalTriggerDigiProducer",
    eeDigis = cms.InputTag('mix:HGCDigisEE'),
    fhDigis = cms.InputTag('mix:HGCDigisHEfront'),
    #bhDigis = cms.InputTag('mix:HGCDigisHEback'),
    FECodec = fe_codec.clone(),
    BEConfiguration = cms.PSet( 
        algorithms = cms.VPSet( cluster_algo )
        )
    )

hgcalTriggerPrimitiveDigiFEReproducer = cms.EDProducer(
    "HGCalTriggerDigiFEReproducer",
    feDigis = cms.InputTag('hgcalTriggerPrimitiveDigiProducer'),
    FECodec = fe_codec.clone(),
    BEConfiguration = cms.PSet( 
        algorithms = cms.VPSet( cluster_algo )
        )
    )
