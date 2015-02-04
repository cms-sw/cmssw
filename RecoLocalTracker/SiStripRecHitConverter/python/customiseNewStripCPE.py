import FWCore.ParameterSet.Config as cms

def customiseNewStripCPE(process):
    process.StripCPEfromTrackAngleESProducer.parameters.useLegacyError = False
    process.StripCPEfromTrackAngleESProducer.parameters.maxChgOneMIP = 6000.

    return process
