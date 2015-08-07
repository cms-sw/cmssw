import FWCore.ParameterSet.Config as cms

def customiseNoCCC(process):

    # apply only in reco step
    if not hasattr(process,'reconstruction'):
        return process
    process.SiStripClusterChargeCutTight.value = -1.
    process.SiStripClusterChargeCutLoose.value = -1.

    return process
