import FWCore.ParameterSet.Config as cms

from DPGAnalysis.MuonTools.muDTTPGPhiFlatTableProducer_cfi import muDTTPGPhiFlatTableProducer

muBmtfInFlatTable = muDTTPGPhiFlatTableProducer.clone()
muTwinMuxInFlatTable = muDTTPGPhiFlatTableProducer.clone(tag = 'TM_IN', name = 'ltTwinMuxIn',  src = cms.InputTag('twinMuxStage2Digis','PhIn'))
muTwinMuxOutFlatTable = muDTTPGPhiFlatTableProducer.clone(tag = 'TM_OUT', name = 'ltTwinMuxOut', src = cms.InputTag('twinMuxStage2Digis','PhOut'))

from DPGAnalysis.MuonTools.muDTTPGThetaFlatTableProducer_cfi import muDTTPGThetaFlatTableProducer

muBmtfInThFlatTable = muDTTPGThetaFlatTableProducer.clone()
muTwinMuxInThFlatTable = muDTTPGThetaFlatTableProducer.clone(tag = 'TM_IN', name = 'ltTwinMuxInTh', src = cms.InputTag('twinMuxStage2Digis','ThIn'))

muL1TriggerTables = cms.Sequence(muTwinMuxInFlatTable
                                 + muTwinMuxOutFlatTable
                                 + muBmtfInFlatTable
                                 + muTwinMuxInThFlatTable
                                 + muBmtfInThFlatTable
                                )
