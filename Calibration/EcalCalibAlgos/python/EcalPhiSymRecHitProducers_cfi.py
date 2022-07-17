import FWCore.ParameterSet.Config as cms

# - thrEEmod is the value in ADC count that rappresent the "5 sigma noise" threshold 
# - A*ring + B is a ad-hoc parametrization of the equivalent noise level, in this way the thr is set in MeV

phisym_params = {
    "barrelHitCollection" : cms.InputTag('ecalRecHit', 'EcalRecHitsEB'),
    "endcapHitCollection" : cms.InputTag('ecalRecHit', 'EcalRecHitsEE'),
    "eThresholds_barrel" : cms.vdouble(
        1.00751, 1.01408, 1.01886, 1.02209, 1.02395, 1.02462, 1.02426, 1.02301, 1.02099, 1.01832,
        1.01508, 1.01139, 1.00731, 1.00292, 0.998285, 0.993462, 0.988504, 0.983455, 0.978355, 0.973239,
        0.968136, 0.963071, 0.958066, 0.953138, 0.948299, 0.943562, 0.938933, 0.934416, 0.930016, 0.925731,
        0.92156, 0.917501, 0.913549, 0.909699, 0.905944, 0.902278, 0.898694, 0.895184, 0.89174, 0.888355,
        0.885023, 0.881736, 0.878489, 0.875276, 0.872093, 0.868936, 0.865802, 0.862691, 0.859601, 0.856534,
        0.853489, 0.850472, 0.847485, 0.844534, 0.841625, 0.838765, 0.835961, 0.833224, 0.830561, 0.827984,
        0.825503, 0.823129, 0.820872, 0.818743, 0.816752, 0.814909, 0.813222, 0.811699, 0.810346, 0.809167,
        0.808164, 0.807336, 0.80668, 0.806189, 0.805852, 0.805656, 0.805582, 0.805606, 0.8057, 0.805828,
        0.805952, 0.806024, 0.80599, 0.805788, 0.805348, 0.805309, 0.80571, 0.805873, 0.805868, 0.805758,
        0.805595, 0.805427, 0.805295, 0.805232, 0.805267, 0.805424, 0.805722, 0.806174, 0.806791, 0.80758,
        0.808543, 0.809683, 0.810995, 0.812478, 0.814124, 0.815926, 0.817876, 0.819964, 0.822179, 0.824511,
        0.82695, 0.829484, 0.832104, 0.834798, 0.837558, 0.840375, 0.84324, 0.846146, 0.849088, 0.852061,
        0.85506, 0.858082, 0.861126, 0.864191, 0.867278, 0.870389, 0.873525, 0.876691, 0.879891, 0.88313,
        0.886414, 0.889751, 0.893146, 0.896607, 0.900142, 0.903758, 0.907462, 0.911262, 0.915162, 0.91917,
        0.923288, 0.92752, 0.931867, 0.936329, 0.940904, 0.945586, 0.950368, 0.95524, 0.960188, 0.965195,
        0.97024, 0.975298, 0.980339, 0.985329, 0.990229, 0.994993, 0.99957, 1.0039, 1.00792, 1.01156,
        1.01474, 1.01737, 1.01934, 1.02054, 1.02085, 1.02014, 1.01824, 1.01499, 1.01018, 1.00359        
    ),    
    "etCut_barrel" : cms.double(1), #this is actually summed to eThr in order to define the upper bound    
    "etCut_endcap" : cms.double(1), #this is actually summed to eThr in order to define the upper bound    
    "A" : cms.vdouble(10, 112.5), # B + A*ring 2016 thr are defined as two linear cut (one for iring<30 and one above)
    "B" : cms.vdouble(150, -2925),
    "thrEEmod" : cms.double(10.),
    "nMisCalib" : cms.int32(10), # <= 10; even; central value does not count
    "misCalibRangeEB" : cms.vdouble(0.95, 1.05),
    "misCalibRangeEE" : cms.vdouble(0.90, 1.10),
}

EcalPhiSymRecHitProducerLumi = cms.EDProducer("EcalPhiSymRecHitProducerLumi")
for par, value in phisym_params.items():
    setattr(EcalPhiSymRecHitProducerLumi, par, value)

EcalPhiSymRecHitProducerRun = cms.EDProducer("EcalPhiSymRecHitProducerRun")
for par, value in phisym_params.items():
    setattr(EcalPhiSymRecHitProducerRun, par, value)
