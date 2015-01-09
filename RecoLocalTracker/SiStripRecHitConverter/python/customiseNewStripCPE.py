import FWCore.ParameterSet.Config as cms

def customiseNewStripCPE(process):
    process.StripCPEfromTrackAngleESProducer.parameters.useLegacyError = True
    process.StripCPEfromTrackAngleESProducer.parameters.maxChgOneMIP = -6000.

    return process
