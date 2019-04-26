import FWCore.ParameterSet.Config as cms

import SimCalorimetry.HGCalSimProducers.hgcalDigitizer_cfi as digiparam
import RecoLocalCalo.HGCalRecProducers.HGCalUncalibRecHit_cfi as recoparam
import RecoLocalCalo.HGCalRecProducers.HGCalRecHit_cfi as recocalibparam 

from L1Trigger.L1THGCal.egammaIdentification import egamma_identification_drnn_cone
from L1Trigger.L1THGCal.customClustering import binSums, dr_layerbylayer

C3d_parValues = cms.PSet( type_multicluster = cms.string('HistoMaxC3d'),
                          dR_multicluster = cms.double(0.),
                          dR_multicluster_byLayer_coefficientA = cms.vdouble(dr_layerbylayer),
                          dR_multicluster_byLayer_coefficientB = cms.vdouble( [0]*53 ),
                          minPt_multicluster = cms.double(0.5), # minimum pt of the multicluster (GeV)
                          nBins_R_histo_multicluster = cms.uint32(36),
                          nBins_Phi_histo_multicluster = cms.uint32(216),
                          binSumsHisto = binSums,
                          threshold_histo_multicluster = cms.double(10.),
                          cluster_association = cms.string("NearestNeighbour"),
                          EGIdentification=egamma_identification_drnn_cone.clone(),
                          neighbour_weights=cms.vdouble(  0    , 0.25, 0   ,
                                                          0.25 , 0  ,  0.25,
                                                          0    , 0.25, 0
                                                          )
 )

be_proc = cms.PSet( ProcessorName  = cms.string('HGCalBackendLayer2Processor3DClustering'),
                    C3d_parameters = C3d_parValues.clone()
                  )

hgcalBackEndLayer2Producer = cms.EDProducer(
    "HGCalBackendLayer2Producer",
    InputCluster = cms.InputTag('hgcalBackEndLayer1Producer:HGCalBackendLayer1Processor2DClustering'),
    ProcessorParameters = be_proc.clone()
    )
