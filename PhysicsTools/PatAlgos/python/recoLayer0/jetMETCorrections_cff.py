import FWCore.ParameterSet.Config as cms

## produce associated jet correction factors in a valuemap
from PhysicsTools.PatAlgos.recoLayer0.jetCorrFactors_cfi import *
## jet sequence
patJetCorrections = cms.Sequence(jetCorrFactors)

## MET corrections for JES
from JetMETCorrections.Type1MET.MetType1Corrections_cff import *
from JetMETCorrections.Configuration.L2L3Corrections_Summer08_cff import *
corMetType1Icone5.corrector = cms.string('L2L3JetCorrectorIC5Calo')

## MET corrections for muons:
from Geometry.CommonDetUnit.globalTrackingGeometry_cfi import *
from TrackingTools.TrackAssociator.DetIdAssociatorESProducer_cff import *
from TrackPropagation.SteppingHelixPropagator.SteppingHelixPropagatorAny_cfi import *
from JetMETCorrections.Type1MET.MetMuonCorrections_cff import corMetGlobalMuons, goodMuonsforMETCorrection
corMetType1Icone5Muons = corMetGlobalMuons.clone(uncorMETInputTag = cms.InputTag('corMetType1Icone5'),
                                                 muonsInputTag    = cms.InputTag('goodMuonsforMETCorrection'))
corMetType1Icone5Muons.TrackAssociatorParameters.useEcal    = False ## RecoHits
corMetType1Icone5Muons.TrackAssociatorParameters.useHcal    = False ## RecoHits
corMetType1Icone5Muons.TrackAssociatorParameters.useHO      = False ## RecoHits
corMetType1Icone5Muons.TrackAssociatorParameters.useCalo    = True  ## CaloTowers
corMetType1Icone5Muons.TrackAssociatorParameters.useMuon    = False ## RecoHits
corMetType1Icone5Muons.TrackAssociatorParameters.truthMatch = False
## MET sequence
patMETCorrections = cms.Sequence(goodMuonsforMETCorrection *
                                 corMetType1Icone5 *
                                 corMetType1Icone5Muons
                                 )

# JetMET sequence
patJetMETCorrections = cms.Sequence(patJetCorrections +
                                    patMETCorrections
                                    )

