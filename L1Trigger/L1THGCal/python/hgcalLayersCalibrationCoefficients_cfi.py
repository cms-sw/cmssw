import FWCore.ParameterSet.Config as cms

from Configuration.Eras.Modifier_phase2_hgcalV16_cff import phase2_hgcalV16

from RecoLocalCalo.HGCalRecProducers.HGCalRecHit_cfi import dEdX, dEdX_v16
from . import l1tHGCalTriggerGeometryESProducer_cfi as geomparam

AllLayer_weights = cms.vdouble(0.0,
                               0.0158115,
                               0.0286877,
                               0.017707,
                               0.00884719,
                               0.00921472,
                               0.00654193,
                               0.00737344,
                               0.00737619,
                               0.0090818,
                               0.00776757,
                               0.011098,
                               0.00561986,
                               0.012015,
                               0.0105088,
                               0.00834435,
                               0.0113901,
                               0.00995654,
                               0.0120987,
                               0.00708785,
                               0.0101533,
                               0.0108289,
                               0.0143815,
                               0.0145606,
                               0.0133204,
                               0.0137476,
                               0.00911436,
                               0.0275028,
                               0.0338628,
                               0.0674028,
                               0.0546596,
                               0.0634012,
                               0.0613207,
                               0.0653995,
                               0.0450429,
                               0.065412,
                               0.0585679,
                               0.0530456,
                               0.0484401,
                               0.0694564,
                               0.0684182,
                               0.117808,
                               0.126668,
                               0.142361,
                               0.154379,
                               0.102089,
                               0.114404,
                               0.160111,
                               0.178321,
                               0.0964375,
                               0.131446,
                               0.115852,
                               0.326339
                               )

TrgLayer_weights = cms.vdouble(0.0,
                               0.0183664,
                               0.,
                               0.0305622,
                               0.,
                               0.0162589,
                               0.,
                               0.0143918,
                               0.,
                               0.0134475,
                               0.,
                               0.0185754,
                               0.,
                               0.0204934,
                               0.,
                               0.016901,
                               0.,
                               0.0207958,
                               0.,
                               0.0167985,
                               0.,
                               0.0238385,
                               0.,
                               0.0301146,
                               0.,
                               0.0274622,
                               0.,
                               0.0468671,
                               0.,
                               0.078819,
                               0.0555831,
                               0.0609312,
                               0.0610768,
                               0.0657626,
                               0.0465653,
                               0.0629383,
                               0.0610061,
                               0.0517326,
                               0.0492882,
                               0.0699336,
                               0.0673457,
                               0.119896,
                               0.125327,
                               0.143235,
                               0.153295,
                               0.104777,
                               0.109345,
                               0.161386,
                               0.174656,
                               0.108053,
                               0.121674,
                               0.1171,
                               0.328053
                               )



def trigger_dedx_weights(ecal_layers, reco_weights):
    weights = []
    for lid, lw in enumerate(reco_weights):
        if lid > (ecal_layers+1):
            weights.append(lw)
        else:
            # Only half the layers are read in the EE at L1T
            if (lid % 2) == 1:
                weights.append(lw+reco_weights[lid-1])
            else:
                weights.append(0)
    return weights

triggerWeights = cms.PSet(
    weights = cms.vdouble(trigger_dedx_weights(geomparam.CEE_LAYERS, dEdX.weights))
)

phase2_hgcalV16.toModify(triggerWeights,
                         weights = cms.vdouble(trigger_dedx_weights(geomparam.CEE_LAYERS_V16, dEdX_v16.weights))
)
