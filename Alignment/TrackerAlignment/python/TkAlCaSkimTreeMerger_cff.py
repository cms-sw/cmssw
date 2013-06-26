import FWCore.ParameterSet.Config as cms

AlignmentTreeMerger = cms.EDAnalyzer("TkAlCaSkimTreeMerger",
                                     FileList= cms.string("DQMHitMapsList.txt"),
                                     TreeName= cms.string("AlignmentHitMaps"),#if you change this be sure to be consistent with the rest of your code
                                     OutputFile=cms.string("AlignmentHitMapsMerged.root"),
                                     NhitsMaxLimit=cms.int32(0),#this applies to ALL TK at the same time; no upper limit by default
                                     NhitsMaxSet=cms.PSet(#in this way you can set different thresholds for each subdet; it is ignored if NhitsMaxLimit is higher than -1
                                                   PXBmaxhits=cms.int32(-1),#no upper limit by default 
                                                   PXFmaxhits=cms.int32(-1),
                                                   TIBmaxhits=cms.int32(-1),
                                                   TIDmaxhits=cms.int32(-1),
                                                   TOBmaxhits=cms.int32(-1),
                                                   TECmaxhits=cms.int32(-1)
                                                   )
                                     )
