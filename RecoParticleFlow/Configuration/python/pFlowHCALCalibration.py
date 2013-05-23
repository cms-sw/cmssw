
def pFlowHCALCalibration(process):
    if hasattr(process,'hcalSimParameters'):
        process.hcalSimParameters.hb.samplingFactors = cms.vdouble(125.44, 125.54, 125.32, 125.13, 124.46,
                                                  125.01, 125.22, 125.48, 124.45, 125.9,
                                                  125.83, 127.01, 126.82, 129.73, 131.83,
                                                  143.52)
        process.hcalSimParameters.he.samplingFactors = cms.vdouble(210.55, 197.93, 186.12, 189.64, 189.63,
                                                                   190.28, 189.61, 189.6, 190.12, 191.22,
                                                                   190.9, 193.06, 188.42, 188.42)
        

    if hasattr(process,'particleFlowRecHitHCAL'):
        process.particleFlowRecHitHCAL.HCAL_Calib = False

    return (process)
