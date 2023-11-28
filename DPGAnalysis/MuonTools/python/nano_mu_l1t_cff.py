import FWCore.ParameterSet.Config as cms

from DPGAnalysis.MuonTools.muDTTPGPhiFlatTableProducer_cfi import muDTTPGPhiFlatTableProducer

muBmtfInFlatTableProducer = muDTTPGPhiFlatTableProducer.clone()
muTwinMuxInFlatTableProducer = muDTTPGPhiFlatTableProducer.clone(tag = 'TM_IN', name = 'ltTwinMuxIn',  src = cms.InputTag('twinMuxStage2Digis','PhIn'))
muTwinMuxOutFlatTableProducer = muDTTPGPhiFlatTableProducer.clone(tag = 'TM_OUT', name = 'ltTwinMuxOut', src = cms.InputTag('twinMuxStage2Digis','PhOut'))

from DPGAnalysis.MuonTools.muDTTPGThetaFlatTableProducer_cfi import muDTTPGThetaFlatTableProducer

muBmtfInThFlatTableProducer = muDTTPGThetaFlatTableProducer.clone()
muTwinMuxInThFlatTableProducer = muDTTPGThetaFlatTableProducer.clone(tag = 'TM_IN', name = 'ltTwinMuxInTh', src = cms.InputTag('twinMuxStage2Digis','ThIn'))

muL1TriggerProducers = cms.Sequence(muTwinMuxInFlatTableProducer
                                    + muTwinMuxOutFlatTableProducer
                                    + muBmtfInFlatTableProducer
                                    + muTwinMuxInThFlatTableProducer
                                    + muBmtfInThFlatTableProducer
                                   )
