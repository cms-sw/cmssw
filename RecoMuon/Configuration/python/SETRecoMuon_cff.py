import FWCore.ParameterSet.Config as cms

from RecoMuon.MuonSeedGenerator.SETMuonSeed_cff import *

from RecoMuon.StandAloneMuonProducer.standAloneMuons_cff import standAloneSETMuons

from RecoMuon.GlobalMuonProducer.GlobalMuonProducer_cff import *
globalSETMuons = globalMuons.clone()
globalSETMuons.MuonCollectionLabel = cms.InputTag("standAloneSETMuons","UpdatedAtVtx")

muontracking_with_SET = cms.Sequence(SETMuonSeed*standAloneSETMuons*globalSETMuons)

from RecoMuon.MuonIdentification.muonIdProducerSequence_cff import *
muonsWithSET = muons1stStep.clone()
muonsWithSET.inputCollectionLabels = ['generalTracks', 'globalSETMuons', cms.InputTag('standAloneSETMuons','UpdatedAtVtx')] 
muonsWithSET.inputCollectionTypes = ['inner tracks', 'links', 'outer tracks']   

#muonreco_with_SET = cms.Sequence(muontracking_with_SET*muonsWithSET)
#run only the tracking part for SET, after that it should be merged with the main ones at some point
muonreco_with_SET = cms.Sequence(muontracking_with_SET)
