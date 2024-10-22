import FWCore.ParameterSet.Config as cms

def outputToFlat(process, outputName):
    """ Replace PoolOutputModule by NanoAODOutputModule to get NanoAOD-like (flat tree) output without merging step """
    orig = getattr(process, outputName)
    setattr(process, outputName,
            cms.OutputModule("NanoAODOutputModule", **{
                pn: orig.getParameter(pn) for pn in orig.parameterNames_()
                if pn != "eventAutoFlushCompressedSize"
                })
            )
    return process

def flatSiStripCalCosmicsNano(process):
    return outputToFlat(process, "ALCARECOStreamSiStripCalCosmicsNano")
