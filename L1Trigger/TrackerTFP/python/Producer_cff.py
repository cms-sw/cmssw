# Produce L1 tracks with TMTT C++ emulation
import FWCore.ParameterSet.Config as cms

from L1Trigger.TrackerTFP.Setup_cff import TrackerTFPSetup
from L1Trigger.TrackerTFP.Producer_cfi import TrackerTFPProducer_params
from L1Trigger.TrackerTFP.DataFormats_cff import TrackerTFPDataFormats
from L1Trigger.TrackerTFP.LayerEncoding_cff import TrackerTFPLayerEncoding
from L1Trigger.TrackerTFP.KalmanFilterFormats_cfi import TrackerTFPKalmanFilterFormats_params

ProducerPP  = cms.EDProducer( 'trackerTFP::ProducerPP' , TrackerTFPProducer_params )
ProducerGP  = cms.EDProducer( 'trackerTFP::ProducerGP' , TrackerTFPProducer_params )
ProducerHT  = cms.EDProducer( 'trackerTFP::ProducerHT' , TrackerTFPProducer_params )
ProducerCTB = cms.EDProducer( 'trackerTFP::ProducerCTB', TrackerTFPProducer_params )
ProducerKF  = cms.EDProducer( 'trackerTFP::ProducerKF' , TrackerTFPProducer_params, TrackerTFPKalmanFilterFormats_params )
ProducerDR  = cms.EDProducer( 'trackerTFP::ProducerDR' , TrackerTFPProducer_params )
ProducerTQ  = cms.EDProducer( 'trackerTFP::ProducerTQ' , TrackerTFPProducer_params )
ProducerTFP = cms.EDProducer( 'trackerTFP::ProducerTFP', TrackerTFPProducer_params )
