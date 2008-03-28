import FWCore.ParameterSet.Config as cms

# Default is to merge all FEDRawDataCollections, regardless of origin
# It is possible to merge collections from the current process only
# This switch is needed in case a FEDRawDataCollection is already 
# present but is not wanted 
rawDataCollector = cms.EDFilter("RawDataCollectorModule",
    currentProcessOnly = cms.bool(False)
)


