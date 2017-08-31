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
adcSaturationBH_MIP = digiparam.hgchebackDigitizer.digiCfg.feCfg.adcSaturation_fC
adcNbitsBH = digiparam.hgchebackDigitizer.digiCfg.feCfg.adcNbits
# Reco calibration parameters
fCPerMIPee = recoparam.HGCalUncalibRecHit.HGCEEConfig.fCPerMIP
fCPerMIPfh = recoparam.HGCalUncalibRecHit.HGCHEFConfig.fCPerMIP
layerWeights = recocalibparam.HGCalRecHit.layerWeights
thicknessCorrection = recocalibparam.HGCalRecHit.thicknessCorrection

# Parameters used in several places
triggerCellLsbBeforeCompression = 100./1024.
triggerCellTruncationBits = 0

# Equalization in the frontend of the sensor responses to 200um sensors
frontend_thickness_corrections = [1./(c1*c2) for c1,c2 in zip(fCPerMIPee,thicknessCorrection)]
c200 = frontend_thickness_corrections[1]
frontend_thickness_corrections = [c/c200 for c in frontend_thickness_corrections]
fCPerMIP_200 = fCPerMIPee[1]
thicknessCorrection_200 = thicknessCorrection[1]

fe_codec = cms.PSet( CodecName  = cms.string('HGCalTriggerCellThresholdCodec'),
                     CodecIndex = cms.uint32(2),
                     MaxCellsInModule = cms.uint32(288),
                     DataLength = cms.uint32(20),
                     linLSB = cms.double(triggerCellLsbBeforeCompression),
                     linnBits = cms.uint32(16),
                     triggerCellTruncationBits = cms.uint32(triggerCellTruncationBits),
                     NData = cms.uint32(999),
                     TCThreshold_fC = cms.double(1.),
                     TCThresholdBH_MIP = cms.double(1.),
                     #take the following parameters from the digitization config file
                     adcsaturation = adcSaturation_fC,
                     adcnBits = adcNbits,
                     tdcsaturation = tdcSaturation_fC,
                     tdcnBits = tdcNbits,
                     tdcOnsetfC = tdcOnset_fC,
                     adcsaturationBH = adcSaturationBH_MIP,
                     adcnBitsBH = adcNbitsBH,
                     ThicknessCorrections = cms.vdouble(frontend_thickness_corrections)
                     )

calib_parValues = cms.PSet( siliconCellLSB_fC =  cms.double( triggerCellLsbBeforeCompression*(2**triggerCellTruncationBits) ),
                            scintillatorCellLSB_MIP = cms.double(float(adcSaturationBH_MIP.value())/(2**float(adcNbitsBH.value()))),
                            fCperMIP = cms.double(fCPerMIP_200),
                            dEdXweights = layerWeights,
                            thickCorr = cms.double(thicknessCorrection_200)
                            )
C2d_parValues = cms.PSet( seeding_threshold_silicon = cms.double(5), # MipT
                          seeding_threshold_scintillator = cms.double(5), # MipT
                          clustering_threshold_silicon = cms.double(2), # MipT
                          clustering_threshold_scintillator = cms.double(2), # MipT
                          dR_cluster = cms.double(3.), # in cm
                          clusterType = cms.string('NNC2d') # clustering type: dRC2d--> Geometric-dR clustering; NNC2d-->Nearest Neighbors clustering
                          )

C3d_parValues = cms.PSet( dR_multicluster = cms.double(0.01), # dR in normalized plane used to clusterize C2d
                          minPt_multicluster = cms.double(0.5), # minimum pt of the multicluster (GeV)
                          calibSF_multicluster = cms.double(1.084),
                          type_multicluster = cms.string('DBSCAN'), #ConedR for the cone algorithm 
                          dist_dbscan_multicluster = cms.double(0.03),
                          minN_dbscan_multicluster = cms.uint32(3)

                          )
cluster_algo =  cms.PSet( AlgorithmName = cms.string('HGCClusterAlgoThreshold'),
                          FECodec = fe_codec.clone(),
                          calib_parameters = calib_parValues.clone(),
                          C2d_parameters = C2d_parValues.clone(),
                          C3d_parameters = C3d_parValues.clone()
                          )

hgcalTriggerPrimitiveDigiProducer = cms.EDProducer(
    "HGCalTriggerDigiProducer",
    eeDigis = cms.InputTag('mix:HGCDigisEE'),
    fhDigis = cms.InputTag('mix:HGCDigisHEfront'),
    bhDigis = cms.InputTag('mix:HGCDigisHEback'),
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
