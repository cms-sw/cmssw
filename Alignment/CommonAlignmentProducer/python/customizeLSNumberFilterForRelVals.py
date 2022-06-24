import FWCore.ParameterSet.Config as cms

##
## Do not filter out the first 20 LS when
## running the PCL alignment in the RelVal case
##

def doNotFilterLS(process):
    if hasattr(process,'lsNumberFilter'):
        process.lsNumberFilter.minLS = 1
    return process

##
## Required 10 instead of 500 hits per structure
## when running the HG PCL alignment in the RelVal case
##

def lowerHitsPerStructure(process):
    if hasattr(process,'SiPixelAliPedeAlignmentProducerHG'):
        process.SiPixelAliPedeAlignmentProducerHG.algoConfig.pedeSteerer.options = cms.vstring(
            'entries 10',
            'chisqcut  30.0  4.5',
            'threads 1 1',
            'closeandreopen'
        )
    return process
