import FWCore.ParameterSet.Config as cms

from FWCore.Modules.bunchCrossingFilter_cfi import bunchCrossingFilter as _bunchCrossingFilter
selectRange1 = _bunchCrossingFilter.clone(                                                                                                                                                           
    bunches = cms.vuint32(range(757,759)) 
)                                                                                                                                                                                                          

selectSingle = _bunchCrossingFilter.clone(
    bunches = cms.vuint32(755)
)

selectRange2 = _bunchCrossingFilter.clone(
    bunches = cms.vuint32(range(1646,1651)) 
)

BXSeq1 = cms.Sequence( selectRange1 )
BXSeq2 = cms.Sequence( selectSingle )
BXSeq3 = cms.Sequence( selectRange2 )

